import pprint
import copy
import functools
import logging
import os
import time

import tqdm
import collections
import torch
import torchvision
from torch import nn
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
import numpy as np

from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.structures import ImageList
from detectron2.utils import comm
from detectron2.utils.comm import get_world_size
from detectron2.utils.events import get_event_storage
from detectron2.checkpoint import Checkpointer, DetectionCheckpointer


from template_lib.d2.distributions.fairnas_noise_sample import fairnas_repeat_tensor
from template_lib.d2.distributions import build_d2distributions
from template_lib.d2.models import build_d2model
from template_lib.trainer.base_trainer import Trainer
from template_lib.trainer import get_ddp_attr
from template_lib.gans import inception_utils, gan_utils, gan_losses, GANLosses
from template_lib.gans.networks import build_discriminator, build_generator
from template_lib.gans.evaluation import get_sample_imgs_list_ddp
from template_lib.d2.optimizer import build_optimizer
from template_lib.utils import modelarts_utils
from template_lib.gans.evaluation import build_GAN_metric_dict
from template_lib.gans.evaluation.fid_score import FIDScore
from template_lib.gans.models import build_GAN_model
from template_lib.utils import get_eval_attr, print_number_params, get_attr_kwargs, get_attr_eval, rawgencount, \
  array_eq_in_list
from template_lib.d2template.trainer import TRAINER_REGISTRY, BaseTrainer
from template_lib.models.ema_model import EMA
from template_lib.d2.data import build_dataset_mapper


def sample_func_with_arcs(G, z, y, classes_arcs):
  G.eval()
  with torch.no_grad():
    z_sample = z.sample()
    if hasattr(y, 'sample'):
      y_sample = y.sample()
    else:
      y_sample = y
    batched_arcs = classes_arcs[y_sample]
    G_z = G(z_sample, y=y_sample, batched_arcs=batched_arcs)
  return G_z


@TRAINER_REGISTRY.register()
class TrainerNASGAN(BaseTrainer):

    def __init__(self, cfg, **kwargs):
      super(BaseTrainer, self).__init__()

      self.myargs                        = kwargs['myargs']
      self.iter_every_epoch              = get_attr_kwargs(cfg.trainer, 'iter_every_epoch', **kwargs)
      self.max_iter                      = get_attr_kwargs(cfg.trainer, 'max_iter', **kwargs)
      self.n_classes                     = get_attr_kwargs(cfg.trainer, 'n_classes', **kwargs)
      self.img_size                      = get_attr_kwargs(cfg.trainer, 'img_size', **kwargs)
      self.train_bs                      = get_attr_kwargs(cfg.trainer, 'train_bs', **kwargs)
      self.eval_tf_every_itr             = get_attr_eval(cfg.trainer, 'eval_tf_every_itr', default=float('inf'),
                                                         iter_every_epoch=self.iter_every_epoch)
      self.eval_torch_every_itr          = get_attr_eval(cfg.trainer, 'eval_torch_every_itr', default=float('inf'),
                                                         iter_every_epoch=self.iter_every_epoch)


      self.train_bs = self._get_bs_per_worker(self.train_bs)
      self.cfg = cfg
      self.device = torch.device(f'cuda:{comm.get_rank()}')
      self.logger = logging.getLogger('tl')
      self.distributed = comm.get_world_size() > 1
      self.best_FID_tf = float('inf')
      self.best_IS_mean_tf = float('-inf')

      self.build_GAN_metrics()
      torch.cuda.set_device(self.device)

      self.build_models(cfg=cfg)

      self.noise_create(cfg=cfg)
      # self.backbone = build_backbone(cfg)
      self.to(self.device)

      self.gan_model = build_GAN_model(cfg=cfg.GAN_model, myargs=self.myargs,
                                       D=self.D, G=self.G, D_optim=self.D_optim, G_optim=self.G_optim)

    def build_models(self, cfg):
      self.models = {}
      self.optims = {}

      self.G, self.G_optim = self.build_generator(cfg=cfg)
      self.models.update({'G': self.G})
      self.optims.update({'G_optim': self.G_optim})

      num_branches = get_ddp_attr(self.G, 'num_branches')
      num_layers = get_ddp_attr(self.G, 'num_layers')

      self.controller, self.controller_optim = self.build_controller(
        cfg=cfg, num_layers=num_layers, num_branches=num_branches)
      if self.controller is not None:
        self.models.update({'controller': self.controller})
      if self.controller_optim is not None:
        self.optims.update({'controller_optim': self.controller_optim})

      self.D, self.D_optim = self.build_discriminator(cfg=cfg)
      self.models.update({'D': self.D})
      self.optims.update({'D_optim': self.D_optim})

      self.logger.info("Model:\n{}".format(self))
      self._print_number_params(models_dict=self.models)

    def build_generator(self, cfg):

      G = build_generator(cfg.generator, n_classes=self.n_classes, img_size=self.img_size)
      G_optim = build_optimizer(cfg.generator.optimizer, params=G.parameters())

      #   G = SharedGeneratorNoSkip(
      #     resolution=resolution, no_optim=False, config=cfg.model.generator)
      # G_optim = G.optim

      G = G.to(self.device)
      if self.distributed:
        pg_G = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        G = DistributedDataParallel(
          G, device_ids=[comm.get_local_rank()], broadcast_buffers=True,
          process_group=pg_G, check_reduction=False, find_unused_parameters=True
        ).to(self.device)
      return G, G_optim

    def build_controller(self, cfg, num_layers, num_branches):

      controller = build_d2model(cfg.controller, n_classes=self.n_classes, num_layers=num_layers,
                                 num_branches=num_branches, FID_IS=self.FID_IS,
                                 myargs=self.myargs, cfg_ops=cfg.generator.cfg_ops,
                                 iter_every_epoch=self.iter_every_epoch)
      controller_optim = build_optimizer(cfg.controller.optimizer, params=controller.parameters())

      controller = controller.to(self.device)
      if self.distributed:
        pg_controller = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        controller = DistributedDataParallel(
          controller, device_ids=[comm.get_local_rank()], broadcast_buffers=True,
          process_group=pg_controller, check_reduction=False
        ).to(self.device)
      return controller, controller_optim

    def build_discriminator(self, cfg):
      fixed_arc_file = get_ddp_attr(self.G, 'fixed_arc_file')
      fixed_epoch = get_ddp_attr(self.G, 'fixed_epoch')
      D = build_discriminator(cfg.discriminator, n_classes=self.n_classes, img_size=self.img_size,
                              fixed_arc_file=fixed_arc_file, fixed_epoch=fixed_epoch)
      D_optim = build_optimizer(cfg=cfg.discriminator.optimizer, params=D.parameters())

      D = D.to(self.device)
      if self.distributed:
        pg_D = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        D = DistributedDataParallel(
          D, device_ids=[comm.get_local_rank()], broadcast_buffers=True,
          process_group=pg_D, check_reduction=False, find_unused_parameters=True
        ).to(self.device)
      return D, D_optim

    def build_GAN_metrics(self):

      gan_metrics_dict = build_GAN_metric_dict(self.cfg)
      if 'TFFIDISScore' in gan_metrics_dict:
        self.FID_IS_tf = gan_metrics_dict['TFFIDISScore']
        self.FID_IS = self.FID_IS_tf

      if 'PyTorchFIDISScore' in gan_metrics_dict:
        self.FID_IS_pytorch = gan_metrics_dict['PyTorchFIDISScore']
        self.FID_IS = self.FID_IS_pytorch
      pass

    def noise_create(self, cfg):

      num_branches = get_ddp_attr(self.G, 'num_branches')
      dim_z = get_ddp_attr(self.G, 'dim_z')
      self.z_train = build_d2distributions(cfg.noise.z_train, sample_shape=(self.train_bs, dim_z),
                                           num_ops=num_branches)
      self.y_train = build_d2distributions(cfg.noise.y_train, n_classes=self.n_classes,
                                           sample_shape=self.train_bs, num_ops=num_branches)

      test_bs = cfg.noise.get("test_bs", 64)
      self.z_test = build_d2distributions(cfg.noise.z_test, sample_shape=(test_bs, dim_z))
      self.y_test = build_d2distributions(cfg.noise.y_test, n_classes=self.n_classes,
                                          sample_shape=test_bs)

      pass

    def calculate_FID(self, pool):
      mu, sigma = np.mean(pool, axis=0), np.cov(pool, rowvar=False)
      FID = inception_utils.numpy_calculate_frechet_distance(
        mu, sigma,
        self.data_mu, self.data_sigma)
      return FID

    def calculate_IS(self, logits):
      IS_mean, IS_std = inception_utils.calculate_inception_score(logits)
      return IS_mean

    def get_sample_func(self, classes_arcs, iter):

      self.G.eval()
      if hasattr(self, 'G_ema'):
        G_sample = self.G_ema
      else:
        G_sample = self.G

      self.logger.info('\nEvaluating arcs: \n%s' % classes_arcs.cpu().numpy())
      self.myargs.textlogger.logstr(iter, arcs='\n' + np.array2string(classes_arcs.cpu().numpy(), threshold=np.inf))

      sample_func = functools.partial(
        sample_func_with_arcs, G=G_sample, z=self.z_test, y=self.y_test, classes_arcs=classes_arcs)
      return sample_func

    def _get_eval_iter(self, iteration):
      return iteration // self.iter_every_epoch

    def evaluate_model(self, classes_arcs, iteration):

      eval_iter = self._get_eval_iter(iteration=iteration)

      if iteration % self.eval_torch_every_itr != 0 and iteration % self.eval_tf_every_itr != 0:
        return

      sample_func = self.get_sample_func(classes_arcs=classes_arcs, iter=eval_iter)

      if hasattr(self, 'FID_IS_pytorch') and iteration % self.eval_torch_every_itr == 0:
        IS_mean_torch, IS_std_torch, FID_torch = self.FID_IS_pytorch(
          sample_func=sample_func, stdout=self.myargs.stdout)
        self.logger.info(f'\n\teval_iter {eval_iter}: '
                         f'IS_mean_torch: {IS_mean_torch:.3f} +- {IS_std_torch:.3f}\n\tFID_torch: {FID_torch:.3f}')
        dict_data = dict(IS_mean_torch=IS_mean_torch, IS_std_torch=IS_std_torch, FID_torch=FID_torch)
        Trainer.summary_dict2txtfig(dict_data=dict_data, prefix='evaltorch', step=eval_iter,
                                    textlogger=self.myargs.textlogger)

      if hasattr(self, 'FID_IS_tf') and iteration % self.eval_tf_every_itr == 0:
        try:
          FID_tf, IS_mean_tf, IS_std_tf = self.FID_IS_tf(
            sample_func=sample_func, stdout=self.myargs.stdout)
          self.logger.info(f'\n\teval_iter {eval_iter}: '
                           f'IS_mean_tf:{IS_mean_tf:.3f} +- {IS_std_tf:.3f}\n\tFID_tf: {FID_tf:.3f}')
          dict_data = (dict(FID_tf=FID_tf, IS_mean_tf=IS_mean_tf, IS_std_tf=IS_std_tf))
          Trainer.summary_dict2txtfig(dict_data=dict_data, prefix='evaltf', step=eval_iter,
                                      textlogger=self.myargs.textlogger)

          if FID_tf < self.best_FID_tf or IS_mean_tf > self.best_IS_mean_tf:
            summ_best = {'best_FID_tf': FID_tf, 'best_IS_mean_tf': IS_mean_tf, 'best_IS_std_tf': IS_std_tf}
            Trainer.summary_dict2txtfig(dict_data=summ_best, prefix='evaltf', step=eval_iter,
                                        textlogger=self.myargs.textlogger)
          if FID_tf < self.best_FID_tf:
            self.best_FID_tf = FID_tf
          if IS_mean_tf > self.best_IS_mean_tf:
            self.best_IS_mean_tf = IS_mean_tf
        except:
          self.logger.warning("Error FID_IS_tf.")
          import traceback
          print(traceback.format_exc())

      modelarts_utils.modelarts_sync_results(args=self.myargs.args, myargs=self.myargs, join=False, end=False)
      comm.synchronize()

    def is_repetitive_between_classes(self, sample_arc):
      n_class, n_layer = sample_arc.shape
      for i in range(n_class):
        for j in range(i + 1, n_class):
          if sum(sample_arc[i] == sample_arc[j]) > n_layer - 1:
            return True
      return False

    def revise_num_parameters(self, fixed_arc):
      if get_ddp_attr(self.G, 'layers_para_matrix') is not None:
        param_matrix = get_ddp_attr(self.G, 'layers_para_matrix')
        n_classes, n_layers = fixed_arc.shape
        for layer_id in range(n_layers):
          for class_id in range(n_classes):
            op_id = fixed_arc[class_id, layer_id]
            param_matrix[op_id, layer_id] = 0
        del_num_para = sum(sum(param_matrix))
        all_num_para = sum([p.data.nelement() for p in self.G.parameters()])
        Trainer.print_number_params(dict(G=self.G, D=self.D))
        self.logger.info('Revised number of params in G:\t {}M'.format(
          (all_num_para - del_num_para) / 1e6))

    def train_func(self, data, iteration, pbar):
      """Perform architecture search by training a controller and shared_cnn.
      """
      if comm.is_main_process() and iteration % self.iter_every_epoch == 0:
        pbar.set_postfix_str(s="Supernet ")

      images, labels = self.preprocess_image(data)
      images = images.tensor

      # FairNAS architecture sampling
      batched_arcs = self.controller(bs=self.train_bs)
      # sync arcs
      batched_arcs = self.get_tensor_of_main_processing(batched_arcs)

      # Prepare data for FairNAS training
      images = get_ddp_attr(self.controller, 'fairnas_repeat_tensor')(sample=images)
      labels = get_ddp_attr(self.controller, 'fairnas_repeat_tensor')(sample=labels)

      self.gan_model(images=images, labels=labels, z=self.z_train, iteration=iteration, batched_arcs=batched_arcs)

      # Just for monitoring the training processing
      classes_arcs = batched_arcs[[0,], ].repeat(self.n_classes, 1)
      self.evaluate_model(classes_arcs=classes_arcs, iteration=iteration)
      comm.synchronize()

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        labels = torch.LongTensor([x["label"] for x in batched_inputs]).to(self.device)
        images = ImageList.from_tensors(images)
        return images, labels

    def _load_model(self, eval_ckpt):
      model = self.get_saved_model()
      checkpointer = Checkpointer(model, save_to_disk=False)
      checkpointer.resume_or_load(eval_ckpt)
      pass

    @staticmethod
    def _get_arc_from_file(fixed_arc_file, fixed_epoch, nrows=1):
      if os.path.isfile(fixed_arc_file):
        print(f'Using fixed_arc_file: {fixed_arc_file}, \tfixed_epoch: {fixed_epoch}')
        with open(fixed_arc_file) as f:
          while True:
            epoch_str = f.readline().strip(': \n')
            sample_arc = []
            for _ in range(nrows):
              class_arc = f.readline().strip('[\n ]')
              sample_arc.append(np.fromstring(class_arc, dtype=int, sep=' '))
            if fixed_epoch == int(epoch_str):
              break
        sample_arc = np.array(sample_arc)
      else:
        raise NotImplemented
      print('fixed arcs: \n%s' % sample_arc)
      return sample_arc

    def get_fixed_arc(self, fixed_arc_file, fixed_epoch):
      self.logger.info(f'\n\tUsing fixed_arc: {fixed_arc_file}, \n\tfixed_epoch: {fixed_epoch}')

      if os.path.isfile(fixed_arc_file):
        n_classes = self.n_classes
        if fixed_epoch < 0:
          with open(fixed_arc_file) as f:
            sample_arc = []
            for _ in range(n_classes):
              class_arc = f.readline().strip('[\n ]')
              sample_arc.append(np.fromstring(class_arc, dtype=int, sep=' '))
        else:
          with open(fixed_arc_file) as f:
            while True:
              epoch_str = f.readline().strip(': \n')
              sample_arc = []
              for _ in range(n_classes):
                class_arc = f.readline().strip('[\n ]')
                sample_arc.append(np.fromstring(class_arc, dtype=int, sep=' '))
              if fixed_epoch == int(epoch_str):
                break
        sample_arc = np.array(sample_arc)

      elif fixed_arc_file == 'random':
        while True:
          sample_arc = np.random.randint(0, len(get_ddp_attr(self.G, 'ops')),
                                         (self.n_classes, get_ddp_attr(self.G, 'num_layers')),
                                         dtype=int)
          if not self.is_repetitive_between_classes(sample_arc=sample_arc):
            break

      elif fixed_arc_file in get_ddp_attr(self.G, 'cfg_ops'):
        # single op
        ops = list(get_ddp_attr(self.G, 'cfg_ops').keys())
        op_idx = ops.index(fixed_arc_file)
        sample_arc = \
          np.ones((self.n_classes, get_ddp_attr(self.G, 'num_layers')), dtype=int) * op_idx
      else:
        raise NotImplemented

      self.logger.info(f'Sample_arc: \n{sample_arc}')
      self.revise_num_parameters(fixed_arc=sample_arc)
      fixed_arc = torch.from_numpy(sample_arc).cuda()
      return fixed_arc

    def after_resume(self):
      pass

@TRAINER_REGISTRY.register()
class TrainerSupernetFair(TrainerNASGAN):

  def __init__(self, cfg, **kwargs):
    super().__init__(cfg=cfg, **kwargs)
    pass

  def _get_eval_iter(self, iteration):
    return iteration // self.iter_every_epoch

  def train_func(self, data, iteration, pbar):
    images, labels = self.preprocess_image(data)
    images = images.tensor

    bs = len(images)

    batched_arcs, fair_arcs = get_ddp_attr(self.controller, 'get_fair_path')(bs=bs, iteration=iteration)

    batched_arcs = self._get_tensor_of_main_processing(batched_arcs)

    # Prepare data for FairNAS training
    images = get_ddp_attr(self.controller, 'fairnas_repeat_tensor')(sample=images)
    labels = get_ddp_attr(self.controller, 'fairnas_repeat_tensor')(sample=labels)

    self.gan_model(images=images, labels=labels, z=self.z_train, iteration=iteration, batched_arcs=batched_arcs)

    # Just for monitoring the training processing
    classes_arcs = batched_arcs[[0, ],].repeat(self.n_classes, 1)
    self.evaluate_model(classes_arcs=classes_arcs, iteration=iteration)
    comm.synchronize()


@TRAINER_REGISTRY.register()
class TrainerSupernetController(TrainerNASGAN):

  def __init__(self, cfg, **kwargs):
    super().__init__(cfg=cfg, **kwargs)

    self.train_controller_every_iter = get_attr_kwargs(cfg.trainer, 'train_controller_every_iter', default=50, **kwargs)
    self.ckpt_dir                    = get_attr_kwargs(cfg.trainer, 'ckpt_dir', default=None, **kwargs)
    self.ckpt_epoch                  = get_attr_kwargs(cfg.trainer, 'ckpt_epoch', default=0, **kwargs)
    self.ckpt_iter_every_epoch       = get_attr_kwargs(cfg.trainer, 'ckpt_iter_every_epoch', default=0, **kwargs)

    if self.ckpt_dir is not None:
      ckpt_path = self._get_ckpt_path(ckpt_dir=self.ckpt_dir, ckpt_epoch=self.ckpt_epoch,
                                      iter_every_epoch=self.ckpt_iter_every_epoch)
      self._load_model(ckpt_path)
    pass

  def _get_eval_iter(self, iteration):
    return iteration // self.iter_every_epoch

  def train_func(self, data, iteration, pbar):
    images, labels = self.preprocess_image(data)
    images = images.tensor

    bs = len(images)

    batched_arcs, fair_arcs = get_ddp_attr(self.controller, 'get_fair_path')(bs=bs, iteration=iteration)

    batched_arcs = self._get_tensor_of_main_processing(batched_arcs)
    fair_arcs = self._get_tensor_of_main_processing(fair_arcs)

    # Prepare data for FairNAS training
    images = get_ddp_attr(self.controller, 'fairnas_repeat_tensor')(sample=images)
    labels = get_ddp_attr(self.controller, 'fairnas_repeat_tensor')(sample=labels)

    self.gan_model(images=images, labels=labels, z=self.z_train, iteration=iteration, batched_arcs=batched_arcs)

    if iteration % self.train_controller_every_iter == 0:
      get_ddp_attr(self.controller, 'train_controller')(
        G=self.G, z=self.z_test, y=self.y_test,
        controller=self.controller, controller_optim=self.controller_optim,
        iteration=iteration, pbar=pbar)

    # Just for monitoring the training processing
    sampled_arc = get_ddp_attr(self.controller, 'get_sampled_arc')(iteration=iteration)
    sampled_arc = self._get_tensor_of_main_processing(sampled_arc)

    classes_arcs = sampled_arc[[0, ],].repeat(self.n_classes, 1)
    self.evaluate_model(classes_arcs=classes_arcs, iteration=iteration)
    comm.synchronize()


@TRAINER_REGISTRY.register()
class TrainerRetrain(TrainerNASGAN):

  def __init__(self, cfg, **kwargs):
    super().__init__(cfg=cfg, **kwargs)

    num_layers = get_ddp_attr(self.G, 'num_layers')
    self.arcs = torch.zeros((1, num_layers), dtype=torch.int64).cuda()

    self.G_ema = copy.deepcopy(self.G)
    self.ema = EMA(source=self.G, target=self.G_ema, decay=0.999, start_itr=0)
    self.models.update({'G_ema': self.G_ema})

    pass

  def train_func(self, data, iteration, pbar):
    images, labels = self.preprocess_image(data)
    images = images.tensor

    bs = len(images)
    batched_arcs = self.arcs.repeat(bs, 1)

    self.gan_model(images=images, labels=labels, z=self.z_train, iteration=iteration, batched_arcs=batched_arcs,
                   ema=self.ema)

    # Just for monitoring the training processing
    classes_arcs = batched_arcs[[0, ],].repeat(self.n_classes, 1)
    self.evaluate_model(classes_arcs=classes_arcs, iteration=iteration)
    comm.synchronize()


@TRAINER_REGISTRY.register()
class TrainerEvaluate(TrainerNASGAN):

  def __init__(self, cfg, **kwargs):
    super().__init__(cfg=cfg, **kwargs)

    num_layers                      = get_ddp_attr(self.G, 'num_layers')
    self.ckpt_dir                   = get_attr_kwargs(cfg.trainer, 'ckpt_dir', default=None, **kwargs)
    self.ckpt_epoch                 = get_attr_kwargs(cfg.trainer, 'ckpt_epoch', default=0, **kwargs)
    self.ckpt_iter_every_epoch      = get_attr_kwargs(cfg.trainer, 'ckpt_iter_every_epoch', default=0, **kwargs)

    self.arcs = torch.zeros((1, num_layers), dtype=torch.int64).cuda()

    self.G_ema = copy.deepcopy(self.G)
    self.ema = EMA(source=self.G, target=self.G_ema, decay=0.999, start_itr=0)
    self.models.update({'G_ema': self.G_ema})

    if self.ckpt_dir is not None:
      ckpt_path = self._get_ckpt_path(ckpt_dir=self.ckpt_dir, ckpt_epoch=self.ckpt_epoch,
                                      iter_every_epoch=self.ckpt_iter_every_epoch)
      self._load_model(ckpt_path)
    pass

  def train_func(self, data, iteration, pbar):
    # images, labels = self.preprocess_image(data)
    # images = images.tensor

    # Just for monitoring the training processing
    classes_arcs = self.arcs.repeat(self.n_classes, 1)
    self.evaluate_model(classes_arcs=classes_arcs, iteration=iteration)
    comm.synchronize()
    exit(-1)


@TRAINER_REGISTRY.register()
class TrainerSupernetCondController(TrainerNASGAN):

  def __init__(self, cfg, **kwargs):
    super().__init__(cfg=cfg, **kwargs)

    self.train_controller_every_iter = get_attr_kwargs(cfg.trainer, 'train_controller_every_iter', default=50, **kwargs)
    self.ckpt_dir = get_attr_kwargs(cfg.trainer, 'ckpt_dir', default=None, **kwargs)
    self.ckpt_epoch = get_attr_kwargs(cfg.trainer, 'ckpt_epoch', default=0, **kwargs)
    self.ckpt_iter_every_epoch = get_attr_kwargs(cfg.trainer, 'ckpt_iter_every_epoch', default=0, **kwargs)

    if self.ckpt_dir is not None:
      ckpt_path = self._get_ckpt_path(ckpt_dir=self.ckpt_dir, ckpt_epoch=self.ckpt_epoch,
                                      iter_every_epoch=self.ckpt_iter_every_epoch)
      self._load_model(ckpt_path)
    pass

  def _get_eval_iter(self, iteration):
    return iteration // self.iter_every_epoch

  def train_func(self, data, iteration, pbar):
    images, labels = self.preprocess_image(data)
    images = images.tensor

    bs = len(images)

    batched_arcs, _ = get_ddp_attr(self.controller, 'get_fair_path')(bs=bs, labels=labels, iteration=iteration)
    batched_arcs = self._get_tensor_of_main_processing(batched_arcs)

    # Prepare data for FairNAS training
    images = get_ddp_attr(self.controller, 'fairnas_repeat_tensor')(sample=images)
    labels = get_ddp_attr(self.controller, 'fairnas_repeat_tensor')(sample=labels)

    self.gan_model(images=images, labels=labels, z=self.z_train, iteration=iteration, batched_arcs=batched_arcs)

    if iteration % self.train_controller_every_iter == 0:
      get_ddp_attr(self.controller, 'train_controller')(
        G=self.G, z=self.z_test, y=self.y_test,
        controller=self.controller, controller_optim=self.controller_optim,
        iteration=iteration, pbar=pbar)

    # Just for monitoring the training processing
    sampled_arc = get_ddp_attr(self.controller, 'get_sampled_arc')(iteration=iteration)
    sampled_arc = self._get_tensor_of_main_processing(sampled_arc)
    if len(sampled_arc) == 1:
      sampled_arc = sampled_arc.repeat(self.n_classes, 1)
    classes_arcs = sampled_arc
    self.evaluate_model(classes_arcs=classes_arcs, iteration=iteration)
    comm.synchronize()


@TRAINER_REGISTRY.register()
class TrainerCondController(TrainerSupernetCondController):

  def __init__(self, cfg, **kwargs):
    super().__init__(cfg=cfg, **kwargs)

    pass

  def _load_model(self, eval_ckpt):
    model = self.get_saved_model()
    delattr(model, 'controller')
    checkpointer = Checkpointer(model, save_to_disk=False)
    checkpointer.resume_or_load(eval_ckpt, resume=False)
    pass

  def train_func(self, data, iteration, pbar):

    if iteration % self.train_controller_every_iter == 0:
      get_ddp_attr(self.controller, 'train_controller')(
        G=self.G, z=self.z_test, y=self.y_test,
        controller=self.controller, controller_optim=self.controller_optim,
        iteration=iteration, pbar=pbar)

    # Just for monitoring the training processing
    sampled_arc = get_ddp_attr(self.controller, 'get_sampled_arc')(iteration=iteration)
    sampled_arc = self._get_tensor_of_main_processing(sampled_arc)
    if len(sampled_arc) == 1:
      sampled_arc = sampled_arc.repeat(self.n_classes, 1)
    classes_arcs = sampled_arc
    self.evaluate_model(classes_arcs=classes_arcs, iteration=iteration)
    comm.synchronize()



@TRAINER_REGISTRY.register()
class TrainerRetrainConditional(TrainerNASGAN):

  def __init__(self, cfg, **kwargs):
    super().__init__(cfg=cfg, **kwargs)

    self.fixed_arc_file             = get_attr_kwargs(cfg.trainer, 'fixed_arc_file', default=None, **kwargs)
    self.fixed_epoch                = get_attr_kwargs(cfg.trainer, 'fixed_epoch', default=0, **kwargs)
    self.fixed_rows                 = get_attr_kwargs(cfg.trainer, 'fixed_rows', default=self.n_classes, **kwargs)
    self.use_ema                    = get_attr_kwargs(cfg.trainer, 'use_ema', default=True, **kwargs)
    self.load_model                 = get_attr_kwargs(cfg.trainer, 'load_model', default=False, **kwargs)
    self.ckpt_dir                   = get_attr_kwargs(cfg.trainer, 'ckpt_dir', default=None, **kwargs)
    self.ckpt_epoch                 = get_attr_kwargs(cfg.trainer, 'ckpt_epoch', default=0, **kwargs)
    self.ckpt_iter_every_epoch      = get_attr_kwargs(cfg.trainer, 'ckpt_iter_every_epoch', default=0, **kwargs)

    sample_arc = self._get_arc_from_file(
      fixed_arc_file=self.fixed_arc_file, fixed_epoch=self.fixed_epoch, nrows=self.fixed_rows)
    self.arcs = torch.from_numpy(sample_arc)
    if self.fixed_rows == 1:
      self.arcs = self.arcs.repeat(self.n_classes, 1)

    if self.use_ema:
      self.G_ema = copy.deepcopy(self.G)
      # self.G_ema = build_generator(cfg.generator, n_classes=self.n_classes, img_size=self.img_size)
      self.ema = EMA(source=self.G, target=self.G_ema, decay=0.999, start_itr=0)
      self.models.update({'G_ema': self.G_ema})
    else:
      self.ema = None

    if self.load_model:
      ckpt_path = self._get_ckpt_path(ckpt_dir=self.ckpt_dir, ckpt_epoch=self.ckpt_epoch,
                                      iter_every_epoch=self.ckpt_iter_every_epoch)
      self._load_model(ckpt_path)
    pass

  def train_func(self, data, iteration, pbar):
    images, labels = self.preprocess_image(data)
    images = images.tensor

    bs = len(images)
    batched_arcs = self.arcs[labels]

    self.gan_model(images=images, labels=labels, z=self.z_train, iteration=iteration, batched_arcs=batched_arcs,
                   ema=self.ema, max_iter=self.max_iter)

    # Just for monitoring the training processing
    classes_arcs = self.arcs
    self.evaluate_model(classes_arcs=classes_arcs, iteration=iteration)
    comm.synchronize()

  def eval_FID_IS(self):
    cfg = self.cfg.eval_FID_IS

    ckpt_dir                = get_attr_kwargs(cfg, 'ckpt_dir')
    ckpt_epoch              = get_attr_kwargs(cfg, 'ckpt_epoch')
    end_ckpt_epoch          = get_attr_kwargs(cfg, 'end_ckpt_epoch', default=ckpt_epoch)
    ckpt_iter_every_epoch   = get_attr_kwargs(cfg, 'ckpt_iter_every_epoch')

    classes_arcs = self.arcs

    for epoch in tqdm.tqdm(range(ckpt_epoch, end_ckpt_epoch + 1), file=self.myargs.stdout,
                           desc=f'{self.myargs.args.time_str_suffix}'):
      if ckpt_dir is not None:
        ckpt_path = self._get_ckpt_path(ckpt_dir=ckpt_dir, ckpt_epoch=epoch,
                                        iter_every_epoch=ckpt_iter_every_epoch)
        self._load_model(ckpt_path)

      self.evaluate_model(classes_arcs=classes_arcs, iteration=epoch*self.iter_every_epoch)
    pass

  def evaluation(self):
    cfg = self.cfg.evaluation

    ckpt_path             = get_attr_kwargs(cfg, 'ckpt_dir')

    classes_arcs = self.arcs

    self._load_model(ckpt_path)

    self.evaluate_model(classes_arcs=classes_arcs, iteration=0)
    pass

  def save_figure(self):
    cfg = self.cfg.save_figure

    ckpt_dir                 = get_attr_kwargs(cfg, 'ckpt_dir')
    ckpt_epoch               = get_attr_kwargs(cfg, 'ckpt_epoch')
    ckpt_iter_every_epoch    = get_attr_kwargs(cfg, 'ckpt_iter_every_epoch')
    num_every_row            = get_attr_kwargs(cfg, 'num_every_row')
    saved_img_path           = get_attr_kwargs(cfg, 'saved_img_path')
    num_samples              = get_attr_kwargs(cfg, 'num_samples')
    n_classes                = get_attr_kwargs(cfg, 'n_classes')

    classes_arcs = self.arcs

    if ckpt_dir is not None:
      ckpt_path = self._get_ckpt_path(ckpt_dir=ckpt_dir, ckpt_epoch=ckpt_epoch,
                                      iter_every_epoch=ckpt_iter_every_epoch)
      self._load_model(ckpt_path)

    with torch.no_grad():
      for i in tqdm.tqdm(range(num_samples), file=self.myargs.stdout):
        imgs = []
        for idx in range(n_classes):
          y_test = self.y_test.sample().fill_(idx)[:num_every_row]
          z_sample = self.z_test.sample()[:num_every_row]
          z_sample = z_sample.to(self.device)

          img = self.G_ema(z_sample, y=y_test, batched_arcs=classes_arcs[y_test])
          imgs.append(img)

        imgs = torch.cat(imgs, 0)
        merged_img = torchvision.utils.make_grid(imgs, nrow=num_every_row, normalize=True)
        torchvision.utils.save_image(merged_img, os.path.join(self.myargs.args.outdir, f'{i}_{saved_img_path}'))
    pass

  def count_flops(self):
    # from thop import profile
    from fvcore.nn.flop_count import flop_count
    import pickle

    # ckpt_iter_every_epoch          = get_attr_kwargs(self.cfg.count_flops, 'ckpt_iter_every_epoch')

    z_sample = self.z_train.sample()
    z_sample = z_sample[[0]].to(self.device)

    flops_dict = {}
    classes_arcs = self.arcs
    for class_idx in range(len(classes_arcs)):
      y = torch.empty(1, dtype=torch.int64).fill_(class_idx)
      batched_arcs = self.arcs[y]

      final_count, skipped_ops = flop_count(self.G, inputs=(z_sample, y, batched_arcs))
      flops = sum(final_count.values())
      flops_dict[class_idx] = flops
    print(pprint.pformat(flops_dict), file=self.myargs.stdout)
    mean_flops = sum(flops_dict.values())/len(flops_dict)
    print(f"mean flops: {mean_flops}")

    saved_data = []
    for k, v in flops_dict.items():
      saved_data.append([k, v])
    saved_data = np.array(saved_data)

    saved_file = os.path.join(self.myargs.args.outdir, 'flops_dict.pkl')
    with open(saved_file, 'wb') as f:
      pickle.dump(saved_data, f)
      print(f'Saved to {saved_file}')
    pass

  def compute_intra_FID(self, ):
    from template_lib.d2.data import build_cifar10_per_class

    cfg = self.cfg.compute_intra_FID

    ckpt_path                      = get_attr_kwargs(cfg, 'ckpt_path', default=None)
    ckpt_dir                       = get_attr_kwargs(cfg, 'ckpt_dir', default='')
    ckpt_epoch                     = get_attr_kwargs(cfg, 'ckpt_epoch', default=0)
    ckpt_iter_every_epoch          = get_attr_kwargs(cfg, 'ckpt_iter_every_epoch', default=0)
    registed_name                  = get_attr_kwargs(cfg, 'registed_name')
    fid_stats_dir                  = get_attr_kwargs(cfg, 'fid_stats_dir')
    num_inception_images           = get_attr_kwargs(cfg, 'num_inception_images')
    intra_FID_file                 = get_attr_kwargs(cfg, 'intra_FID_file')
    eval_total_FID                 = get_attr_kwargs(cfg, 'eval_total_FID', default=True)

    if not ckpt_path:
      ckpt_path = self._get_ckpt_path(ckpt_dir=ckpt_dir, ckpt_epoch=ckpt_epoch,
                                      iter_every_epoch=ckpt_iter_every_epoch)
    self._load_model(ckpt_path)

    classes_arcs = self.arcs
    if eval_total_FID:
      self.evaluate_model(classes_arcs=classes_arcs, iteration=0)

    class_to_idx = MetadataCatalog.get(registed_name).get('class_to_idx')
    self.logger.info('\n' + pprint.pformat(class_to_idx))

    intra_FIDs = []
    pbar = tqdm.tqdm(class_to_idx.items(), desc=f"compute intra FID {self.myargs.args.time_str_suffix}", file=self.myargs.stdout)
    pbar.set_postfix_str(self.myargs.args.time_str_suffix)
    for class_dir, idx in pbar:
      mu_sigma = np.load(os.path.join(fid_stats_dir, f'{idx}.npz'))
      class_mu, class_sigma = mu_sigma['mu'], mu_sigma['sigma']

      y_test = self.y_test.sample().fill_(idx)
      sample_func = functools.partial(
        sample_func_with_arcs, G=self.G_ema, z=self.z_test, y=y_test, classes_arcs=classes_arcs)

      while True:
        try:
          mu, sigma = self.FID_IS(sample_func, return_fid_stat=True, num_inception_images=num_inception_images,
                                  stdout=self.myargs.stdout)
          intra_FID = self.FID_IS._calculate_frechet_distance(mu, sigma, class_mu, class_sigma)
          break
        except KeyboardInterrupt:
          exit(-1)
        except:
          import traceback
          print(traceback.format_exc())
          continue
      self.logger.info(f'{class_dir} intra FID: {intra_FID}')
      intra_FIDs.append([idx, intra_FID])
      Trainer.summary_dict2txtfig(dict_data=dict(intra_FID_tf=intra_FID), prefix='intraFID',
                                  step=idx, textlogger=self.myargs.textlogger)

    if len(intra_FIDs) > 0:
      intra_FIDs = np.array(intra_FIDs)
      intra_FID_file = os.path.join(self.myargs.args.outdir, intra_FID_file)
      np.savez(intra_FID_file, intra_FID=intra_FIDs)
      self.logger.info(f'Saved intra FID to {intra_FID_file}')
    comm.synchronize()
    return

  def finetune_all_class(self, ):

    registed_name                          = get_attr_kwargs(self.cfg.finetune_all_class, 'registed_name')
    fid_stats_dir                          = get_attr_kwargs(self.cfg.finetune_all_class, 'fid_stats_dir')
    num_inception_images                   = get_attr_kwargs(self.cfg.finetune_all_class, 'num_inception_images')
    dataset_name_prefix                    = get_attr_kwargs(self.cfg.finetune_all_class, 'dataset_name_prefix')
    max_epoch                              = get_attr_kwargs(self.cfg.finetune_all_class, 'max_epoch')
    resume_ckpt_dir                        = get_attr_kwargs(self.cfg.finetune_all_class, 'resume_ckpt_dir', default=None)
    resume_ckpt_epoch                      = get_attr_kwargs(self.cfg.finetune_all_class, 'resume_ckpt_epoch', default=0)
    resume_ckpt_iter_every_epoch           = get_attr_kwargs(self.cfg.finetune_all_class, 'resume_ckpt_iter_every_epoch', default=0)
    class_idx_list                         = get_attr_kwargs(self.cfg.finetune_all_class, 'class_idx_list',
                                                             default=list(range(self.n_classes)))
    use_torch                              = get_attr_kwargs(self.cfg.finetune_all_class, 'use_torch', default=False)
    eval_every_epoch                       = get_attr_kwargs(self.cfg.finetune_all_class, 'eval_every_epoch', default=1)

    class_to_idx = MetadataCatalog.get(registed_name).get('class_to_idx')
    self.logger.info('\n' + pprint.pformat(class_to_idx))

    pbar_class = tqdm.tqdm(class_to_idx.items(), desc=f"finetune_all_class {self.myargs.args.time_str_suffix}",
                     file=self.myargs.stdout)
    pbar_class.set_postfix_str(self.myargs.args.time_str_suffix)
    fid_logits_dict = {}
    checkpointer = DetectionCheckpointer(self.get_saved_model(),
                                         os.path.join(self.cfg.OUTPUT_DIR), **self.build_optimizer())
    resume_ckpt_model = self._get_ckpt_path(ckpt_dir=resume_ckpt_dir, ckpt_epoch=resume_ckpt_epoch,
                                            iter_every_epoch=resume_ckpt_iter_every_epoch)

    checkpointer.resume_or_load(resume_ckpt_model, resume=False)
    # To avoid cuda out of memory error.
    with torch.no_grad():
      bs = 256
      z = self.z_train.sample()
      z = self.z_train.sample([bs, ] + list(z.shape)[1:])
      z = z.cuda()
      y = torch.randint(0, self.n_classes, (bs, ))
      batched_arcs = self.arcs[y]
      G_z = self.G(z, y, batched_arcs)

    for class_name, idx in pbar_class:
      if not idx in class_idx_list:
        continue
      checkpointer.resume_or_load(resume_ckpt_model, resume=False)

      best_intra_FID = float('inf')
      mu_sigma = np.load(os.path.join(fid_stats_dir, f'{idx}.npz'))
      class_mu, class_sigma = mu_sigma['mu'], mu_sigma['sigma']

      self.cfg.defrost()
      dataset_name = dataset_name_prefix + '_' + class_name
      self.cfg.DATASETS.TRAIN = (dataset_name,)
      self.cfg.freeze()

      mapper = build_dataset_mapper(self.cfg.start.dataset_mapper)
      data_loader = build_detection_train_loader(self.cfg, mapper=mapper)

      num_images = MetadataCatalog.get(dataset_name).get('num_images')
      iter_every_epoch = num_images // self.cfg.start.IMS_PER_BATCH
      max_iter = iter_every_epoch * max_epoch
      pbar = zip(data_loader, range(0, max_iter))
      pbar = tqdm.tqdm(
        pbar, desc=f'do_train, {self.myargs.args.time_str_suffix}, '
                   f'iters {iter_every_epoch} * bs {self.cfg.start.IMS_PER_BATCH}'
                   f' = imgs {iter_every_epoch * self.cfg.start.IMS_PER_BATCH}',
        file=self.myargs.stdout, initial=0, total=max_iter)

      try:
        for data, iteration in pbar:
          cur_epoch = (iteration + 1) // iter_every_epoch
          pbar.set_postfix_str(f'{idx} {class_name} epoch={cur_epoch}')
          images, labels = self.preprocess_image(data)
          images = images.tensor

          batched_arcs = self.arcs[labels]

          self.gan_model(images=images, labels=labels, z=self.z_train, iteration=iteration, batched_arcs=batched_arcs,
                         ema=self.ema, max_iter=self.max_iter)
          if (iteration + 1) % (iter_every_epoch * eval_every_epoch) == 0 or iteration == 0:
            intra_FID, fid_logits = self._compute_intra_FID(class_id=idx, num_inception_images=num_inception_images,
                                                            class_mu=class_mu, class_sigma=class_sigma,
                                                            use_torch=use_torch)

            summary_d = {f'{idx}_{class_name}_FID': intra_FID}
            Trainer.summary_dict2txtfig(dict_data=summary_d, prefix='finetune', step=cur_epoch,
                                        textlogger=self.myargs.textlogger, save_fig_sec=1)

            if best_intra_FID > intra_FID:
              best_intra_FID = intra_FID
              fid_logits_dict[class_name] = fid_logits
              model_name = f'best_model_{idx}_{class_name}'
              checkpointer.save(model_name)
              class_fid_logits_file = os.path.join(self.myargs.args.outdir, 'fid_logits',
                                                   f'{idx}_{class_name}_fid_logits')
              os.makedirs(os.path.dirname(class_fid_logits_file), exist_ok=True)
              np.savez(class_fid_logits_file, fid_logits=fid_logits)
      except:
        print('Sleeping')
        import traceback
        print(traceback.format_exc())
        time.sleep(5)
        pass
      summary_d = {f'intra_FID': best_intra_FID}
      Trainer.summary_dict2txtfig(dict_data=summary_d, prefix='finetune', step=idx,
                                  textlogger=self.myargs.textlogger, save_fig_sec=1)
      del data_loader
      torch.cuda.empty_cache()
      pass
    fid_logits = np.concatenate(list(fid_logits_dict.values()), axis=0)
    fid_logits_file = os.path.join(self.myargs.args.outdir, "fid_logits")
    np.savez(fid_logits_file, fid_logits_dict=fid_logits_dict)
    all_mu, all_sigma = self.FID_IS._calculate_fid_stat(pred_FIDs=fid_logits)
    all_FID = self.FID_IS._calculate_frechet_distance(all_mu, all_sigma,
                                                      self.FID_IS.mu_data, self.FID_IS.sigma_data)
    self.logger.info(f"Final FID: {all_FID}")
    pass

  def _compute_intra_FID(self, class_id, num_inception_images, class_mu, class_sigma, use_torch=False):
    y_test = self.y_test.sample().fill_(class_id)
    sample_func = functools.partial(
      sample_func_with_arcs, G=self.G_ema, z=self.z_test, y=y_test, classes_arcs=self.arcs)

    while True:
      try:
        mu, sigma, fid_logits = self.FID_IS(sample_func, return_fid_stat=True, return_fid_logit=True,
                                            num_inception_images=num_inception_images, stdout=self.myargs.stdout)
        intra_FID = self.FID_IS._calculate_frechet_distance(mu, sigma, class_mu, class_sigma, use_torch=use_torch)
        break
      except KeyboardInterrupt:
        raise KeyboardInterrupt
      except:
        import traceback
        print(traceback.format_exc())
        continue
    self.logger.info(f'class_id {class_id} intra FID: {intra_FID}')
    return intra_FID, fid_logits

  def eval_finetune_all_class(self, ):

    registed_name                = get_attr_kwargs(self.cfg.eval_finetune_all_class, 'registed_name')
    fid_stats_dir                = get_attr_kwargs(self.cfg.eval_finetune_all_class, 'fid_stats_dir')
    num_inception_images         = get_attr_kwargs(self.cfg.eval_finetune_all_class, 'num_inception_images')
    dataset_name_prefix          = get_attr_kwargs(self.cfg.eval_finetune_all_class, 'dataset_name_prefix')
    fid_logits                   = get_attr_kwargs(self.cfg.eval_finetune_all_class, 'fid_logits')


    class_to_idx = MetadataCatalog.get(registed_name).get('class_to_idx')
    classes = MetadataCatalog.get(registed_name).get('classes')
    self.logger.info('\n' + pprint.pformat(class_to_idx))

    pbar_class = tqdm.tqdm(class_to_idx.items(), desc=f"finetune_all_class {self.myargs.args.time_str_suffix}",
                           file=self.myargs.stdout)
    pbar_class.set_postfix_str(self.myargs.args.time_str_suffix)
    logits_dict = {}
    for fid_logits_dir, class_idx_list in fid_logits.items():
      for class_idx in class_idx_list:
        fid_file = os.path.join(fid_logits_dir, 'fid_logits', f'{class_idx}_{classes[class_idx]}_fid_logits.npz')
        logits = np.load(fid_file, allow_pickle=True)['fid_logits']
        logits_dict[class_idx] = logits
    intra_FID_dict = {}
    for class_idx in sorted(logits_dict):
      logits = logits_dict[class_idx]
      mu_sigma = np.load(os.path.join(fid_stats_dir, f'{class_idx}.npz'))
      class_mu, class_sigma = mu_sigma['mu'], mu_sigma['sigma']

      mu, sigma = self.FID_IS._calculate_fid_stat(pred_FIDs=logits)
      intra_FID = self.FID_IS._calculate_frechet_distance(mu, sigma, class_mu, class_sigma)
      intra_FID_dict[class_idx] = intra_FID
      summary_d = {f'intra_FID': intra_FID}
      Trainer.summary_dict2txtfig(dict_data=summary_d, prefix='finetune', step=class_idx,
                                  textlogger=self.myargs.textlogger, save_fig_sec=1)
      pass

    self.logger.info('\n' + pprint.pformat(intra_FID_dict))

    fid_logits = np.concatenate(list(logits_dict.values()), axis=0)
    fid_logits_file = os.path.join(self.myargs.args.outdir, "fid_logits")
    np.savez(fid_logits_file, logits_dict=logits_dict)
    all_mu, all_sigma = self.FID_IS._calculate_fid_stat(pred_FIDs=fid_logits)
    all_FID = self.FID_IS._calculate_frechet_distance(all_mu, all_sigma,
                                                      self.FID_IS.mu_data, self.FID_IS.sigma_data)
    self.logger.info(f"Final FID: {all_FID}")
    pass

  def eval_calibrated_model(self, ):
    from template_lib.d2.data import build_cifar10_per_class

    registed_name = get_attr_kwargs(self.cfg.eval_calibrated_model, 'registed_name')
    fid_stats_dir = get_attr_kwargs(self.cfg.eval_calibrated_model, 'fid_stats_dir')
    fid_logits_file = get_attr_kwargs(self.cfg.eval_calibrated_model, 'fid_logits_file')

    class_to_idx = MetadataCatalog.get(registed_name).get('class_to_idx')
    self.logger.info('\n' + pprint.pformat(class_to_idx))

    class2logits_dict = (np.load(fid_logits_file, allow_pickle=True)['fid_logits_dict']).item()
    index2logits_dict = {class_to_idx[k]: v for k, v in class2logits_dict.items()}
    intra_FID_dict = {}
    for class_idx in sorted(index2logits_dict):
      mu_sigma = np.load(os.path.join(fid_stats_dir, f'{class_idx}.npz'))
      class_mu, class_sigma = mu_sigma['mu'], mu_sigma['sigma']

      logits = index2logits_dict[class_idx]
      mu, sigma = self.FID_IS._calculate_fid_stat(pred_FIDs=logits)
      intra_FID = self.FID_IS._calculate_frechet_distance(mu, sigma, class_mu, class_sigma)
      intra_FID_dict[class_idx] = intra_FID
      summary_d = {f'intra_FID': intra_FID}
      Trainer.summary_dict2txtfig(dict_data=summary_d, prefix='finetune', step=class_idx,
                                  textlogger=self.myargs.textlogger, save_fig_sec=1)
      pass

    self.logger.info('\n' + pprint.pformat(intra_FID_dict))

    fid_logits = np.concatenate(list(class2logits_dict.values()), axis=0)
    all_mu, all_sigma = self.FID_IS._calculate_fid_stat(pred_FIDs=fid_logits)
    all_FID = self.FID_IS._calculate_frechet_distance(all_mu, all_sigma,
                                                      self.FID_IS.mu_data, self.FID_IS.sigma_data)
    self.logger.info(f"Calibrated FID: {all_FID}")
    pass


@TRAINER_REGISTRY.register()
class TrainerRetrainConditionalFinetuneClass(TrainerRetrainConditional):

  def __init__(self, cfg, **kwargs):
    super().__init__(cfg=cfg, **kwargs)

    self.finetune_class        = get_attr_kwargs(cfg.trainer, 'finetune_class', default=0, **kwargs)
    self.reset_G_ema           = get_attr_kwargs(cfg.trainer, 'reset_G_ema', default=False, **kwargs)

    self.arcs = self.arcs[self.finetune_class].repeat(self.n_classes, 1)
    pass

  def after_resume(self):
    if self.reset_G_ema:
      with torch.no_grad():
        source_dict = self.G.state_dict()
        target_dict = self.G_ema.state_dict()
        for key in source_dict:
          target_dict[key].data.copy_(source_dict[key].data)
    pass


