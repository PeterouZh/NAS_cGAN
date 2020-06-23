import copy
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from detectron2.utils import comm
from detectron2.checkpoint import DetectionCheckpointer, Checkpointer

from template_lib.utils import get_ddp_attr, get_ddp_module, get_attr_eval, get_attr_kwargs
from template_lib.gans.models import build_GAN_model
from template_lib.d2.models import build_d2model
from template_lib.d2.optimizer import build_optimizer
from template_lib.d2.distributions.fairnas_noise_sample import fairnas_repeat_tensor
from template_lib.models.ema_model import EMA

from .build import TRAINER_REGISTRY
from . import PAGANSearch


class DumpModule(nn.Module):
  def __init__(self, ):
    super(DumpModule, self).__init__()
    pass


@TRAINER_REGISTRY.register()
class PAGANTrainControllerSuperNet(PAGANSearch):

  def __init__(self, cfg, **kwargs):
    super(PAGANTrainControllerSuperNet, self).__init__(cfg=cfg, **kwargs)

    self.train_controller_every_iter        = get_attr_kwargs(cfg.trainer, 'train_controller_every_iter', default=50, **kwargs)
    pass

  def build_controller(self, cfg, num_layers, num_branches):

    controller = build_d2model(cfg.model.controller, n_classes=self.n_classes, num_layers=num_layers,
                               num_branches=num_branches, FID_IS=self.FID_IS,
                               myargs=self.myargs, cfg_ops=cfg.model.generator.cfg_ops)
    controller_optim = build_optimizer(cfg.model.controller.optimizer, params=controller.parameters())

    controller = controller.to(self.device)
    if self.distributed:
      pg_controller = torch.distributed.new_group(range(torch.distributed.get_world_size()))
      controller = DistributedDataParallel(
        controller, device_ids=[comm.get_local_rank()], broadcast_buffers=True,
        process_group=pg_controller, check_reduction=False
      ).to(self.device)
    return controller, controller_optim

  def train_func(self, data, iteration, pbar):
    images, labels = self.preprocess_image(data)
    images = images.tensor

    bs = len(images)

    batched_arcs, fair_arcs = get_ddp_attr(self.controller, 'get_fair_path')(bs=bs)

    batched_arcs = self.get_tensor_of_main_processing(batched_arcs)
    fair_arcs = self.get_tensor_of_main_processing(fair_arcs)

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
    sampled_arc = get_ddp_attr(self.controller, 'get_sampled_arc')()
    sampled_arc = self.get_tensor_of_main_processing(sampled_arc)

    classes_arcs = sampled_arc[[0,],].repeat(self.n_classes, 1)
    self.evaluate_model(classes_arcs=classes_arcs, iteration=iteration)
    comm.synchronize()

  def _get_iter_per_worker(self, iteration):
    iter_per_worker = iteration
    return iter_per_worker

  def _get_eval_iter(self, iteration):
    return iteration // self.iter_every_epoch


@TRAINER_REGISTRY.register()
class TrainerDenseGANSupernetController(PAGANTrainControllerSuperNet):

  def __init__(self, cfg, **kwargs):
    super(TrainerDenseGANSupernetController, self).__init__(cfg=cfg, **kwargs)

    self.train_controller_every_iter = get_attr_kwargs(cfg.trainer, 'train_controller_every_iter', default=50, **kwargs)

    pass

  def build_controller(self, cfg, num_layers, num_branches):

    controller = build_d2model(cfg.model.controller, n_classes=self.n_classes, num_layers=num_layers,
                               num_branches=num_branches, FID_IS=self.FID_IS,
                               myargs=self.myargs, cfg_ops=cfg.model.generator.cfg_ops)
    controller_optim = build_optimizer(cfg.model.controller.optimizer, params=controller.parameters())

    controller = controller.to(self.device)
    if self.distributed:
      pg_controller = torch.distributed.new_group(range(torch.distributed.get_world_size()))
      controller = DistributedDataParallel(
        controller, device_ids=[comm.get_local_rank()], broadcast_buffers=True,
        process_group=pg_controller, check_reduction=False
      ).to(self.device)
    return controller, controller_optim

  def _get_eval_iter(self, iteration):
    return iteration // self.iter_every_epoch

  def train_func(self, data, iteration, pbar):
    images, labels = self.preprocess_image(data)
    images = images.tensor

    bs = len(images)

    batched_arcs, fair_arcs = get_ddp_attr(self.controller, 'get_fair_path')(bs=bs)

    batched_arcs = self.get_tensor_of_main_processing(batched_arcs)
    fair_arcs = self.get_tensor_of_main_processing(fair_arcs)

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
    sampled_arc = get_ddp_attr(self.controller, 'get_sampled_arc')()
    sampled_arc = self.get_tensor_of_main_processing(sampled_arc)

    classes_arcs = sampled_arc[[0, ],].repeat(self.n_classes, 1)
    self.evaluate_model(classes_arcs=classes_arcs, iteration=iteration)
    comm.synchronize()


@TRAINER_REGISTRY.register()
class TrainerDenseGANSupernetControllerExplore(PAGANTrainControllerSuperNet):

  def __init__(self, cfg, **kwargs):
    super(TrainerDenseGANSupernetControllerExplore, self).__init__(cfg=cfg, **kwargs)

    pass

  def train_func(self, data, iteration, pbar):
    images, labels = self.preprocess_image(data)
    images = images.tensor

    bs = len(images)

    batched_arcs = get_ddp_attr(self.controller, 'get_sampled_arc')(bs=bs)

    self.gan_model(images=images, labels=labels, z=self.z_train, iteration=iteration, batched_arcs=batched_arcs)

    if iteration % self.train_controller_every_iter == 0:
      get_ddp_attr(self.controller, 'train_controller')(
        G=self.G, z=self.z_train, y=self.y_train,
        controller=self.controller, controller_optim=self.controller_optim,
        iteration=iteration, pbar=pbar)

    # Just for monitoring the training processing
    sampled_arc = get_ddp_attr(self.controller, 'get_sampled_arc')()
    sampled_arc = self.get_tensor_of_main_processing(sampled_arc)

    classes_arcs = sampled_arc[[0, ],].repeat(self.n_classes, 1)
    self.evaluate_model(classes_arcs=classes_arcs, iteration=iteration)
    comm.synchronize()



@TRAINER_REGISTRY.register()
class TrainerDenseGANRetrain(TrainerDenseGANSupernetController):

  def __init__(self, cfg, **kwargs):
    super(TrainerDenseGANRetrain, self).__init__(cfg=cfg, **kwargs)

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
class TrainerDenseGANEvaluate(TrainerDenseGANSupernetController):

  def __init__(self, cfg, **kwargs):
    super(TrainerDenseGANEvaluate, self).__init__(cfg=cfg, **kwargs)

    self.ckpt_dir                    = get_attr_kwargs(cfg.trainer, 'ckpt_dir', **kwargs)
    self.ckpt_epoch                  = get_attr_kwargs(cfg.trainer, 'ckpt_epoch', **kwargs)
    self.iter_every_epoch            = get_attr_kwargs(cfg.trainer, 'iter_every_epoch', **kwargs)

    self.G_ema = copy.deepcopy(self.G)
    self.models.update({'G_ema': self.G_ema})
    eval_ckpt = self._get_ckpt_path(ckpt_dir=self.ckpt_dir, ckpt_epoch=self.ckpt_epoch, iter_every_epoch=self.iter_every_epoch)
    self._load_G(eval_ckpt)

    num_layers = get_ddp_attr(self.G, 'num_layers')
    self.arcs = torch.zeros((1, num_layers), dtype=torch.int64).cuda()

    pass

  def train_func(self, data, iteration, pbar):

    classes_arcs = self.arcs.repeat(self.n_classes, 1)
    self.evaluate_model(classes_arcs=classes_arcs, iteration=iteration)

    comm.synchronize()
    exit(-1)

  def _load_G(self, eval_ckpt):
    checkpointer = Checkpointer(self.get_saved_model(), save_to_disk=False)
    checkpointer.resume_or_load(eval_ckpt)

    pass
