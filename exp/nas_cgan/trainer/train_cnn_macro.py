import collections, tqdm, os, copy
import functools

import numpy as np
import torch, torchvision
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data.dataset import Subset
import torch.nn.functional as F

from template_lib.models import ema_model
from template_lib.trainer import base_trainer
from template_lib.gans import inception_utils, gan_utils, gan_losses, GANLosses
from template_lib.gans.evaluation.inception_score import InceptionScore
from template_lib.gans.evaluation.fid_score import FIDScore
from template_lib.utils import modelarts_utils

from ..models.controller import Controller
from ..models.shared_cnn import SharedCNN, SharedGenerator
from ..models.supernet import SharedGeneratorNoSkip, SharedGeneratorCBN
from ..utils.utils import AverageMeter
from ..utils.cutout import Cutout
from . import imsize_dict, nclass_dict, activation_dict
from ..models import BigGAN
from ..models.controller import G_Controller


class Trainer(base_trainer.Trainer):
  def __init__(self, args, myargs):
    self.args = args
    self.myargs = myargs
    self.init_static_method()
    self.config = myargs.config
    self.logger = myargs.logger
    self.train_dict = self.init_train_dict()

    # self.dataset_load()
    self.model_create()
    self.optimizer_create()
    self.scheduler_create()

    self.noise_create()
    # load inception network
    self.inception_metrics_func_create_tf()
    self.inception_metrics_func_create_pytorch()

  def init_train_dict(self, ):
    train_dict = collections.OrderedDict()
    train_dict['epoch_shared_generator'] = 0
    train_dict['batch_shared_generator'] = 0
    train_dict['batch_controller'] = 0
    train_dict['best_FID'] = 9999
    train_dict['best_IS'] = 0
    self.myargs.train_dict = train_dict
    self.myargs.checkpoint_dict['train_dict'] = train_dict
    return train_dict

  def dataset_load(self):
    args = self.config.dataset
    if args.dataset.lower() == 'c10_deprecated':
      self.dataset_load_cifar10(args)
    elif args.dataset.lower() in ['c10', 'stl10', 'mnist', 'mnist_svhn',
                                  'celeba_attr']:
      import TOOLS.datasets
      imgdataset = TOOLS.datasets.ImageDataset(args=args)
      self.data_loaders = {}
      self.data_loaders['train_dataset'] = imgdataset.train
    else:
      assert 0
    pass

  def dataset_load_cifar10(self, args):
    """Create data loaders for the CIFAR-10 dataset.

    Returns: Dict containing data loaders.
    """
    args = self.config.dataset
    normalize = transforms.Normalize(
      mean=[0.5, 0.5, 0.5],
      std=[0.5, 0.5, 0.5])

    train_transform = transforms.Compose([
      transforms.ToTensor(),
      normalize])

    if args.cutout > 0:
      train_transform.transforms.append(Cutout(length=args.cutout))

    valid_transform = transforms.Compose([
      transforms.ToTensor(),
      normalize])

    test_transform = transforms.Compose([
      transforms.ToTensor(),
      normalize])

    args.data_path = os.path.expanduser(args.data_path)
    train_dataset = datasets.CIFAR10(root=args.data_path,
                                     train=True,
                                     transform=train_transform,
                                     download=True)

    valid_dataset = datasets.CIFAR10(root=args.data_path,
                                     train=True,
                                     transform=valid_transform,
                                     download=True)

    test_dataset = datasets.CIFAR10(root=args.data_path,
                                    train=False,
                                    transform=test_transform,
                                    download=True)

    train_indices = list(range(0, 45000))
    valid_indices = list(range(45000, 50000))
    train_subset = Subset(train_dataset, train_indices)
    valid_subset = Subset(valid_dataset, valid_indices)

    data_loaders = {}
    data_loaders['train_subset'] = torch.utils.data.DataLoader(
      dataset=train_subset,
      batch_size=args.batch_size,
      shuffle=True,
      pin_memory=True,
      num_workers=args.num_worker_train)

    data_loaders['valid_subset'] = torch.utils.data.DataLoader(
      dataset=valid_subset,
      batch_size=args.batch_size,
      shuffle=True,
      pin_memory=True,
      num_workers=args.num_worker_val,
      drop_last=True)

    data_loaders['train_dataset'] = torch.utils.data.DataLoader(
      dataset=train_dataset,
      batch_size=args.batch_size,
      shuffle=True,
      pin_memory=True,
      num_workers=args.num_worker_train)

    data_loaders['test_dataset'] = torch.utils.data.DataLoader(
      dataset=test_dataset,
      batch_size=args.batch_size,
      shuffle=False,
      pin_memory=True,
      num_workers=args.num_worker_test)

    self.data_loaders = data_loaders

  def val_batch_data(self):
    if not hasattr(self, "valid_subset_itr"):
      self.valid_subset_itr = iter(self.data_loaders['valid_subset'])
    try:
      images, labels = next(self.valid_subset_itr)
    except:
      self.valid_subset_itr = iter(self.data_loaders['valid_subset'])
      images, labels = next(self.valid_subset_itr)
    images = images.cuda()
    labels = labels.cuda()
    return images, labels

  def model_create(self):
    config = self.config.model
    resolution = imsize_dict[self.config.dataset.dataset]
    n_classes = nclass_dict[self.config.dataset.dataset]
    self.n_classes = n_classes
    if hasattr(self.config.dataset, 'attr'):
      assert n_classes == 2 ** len(self.config.dataset.attr)
    if getattr(config, 'use_cbn', False):
      self.G = SharedGeneratorCBN(
        resolution=resolution, n_classes=n_classes,
        no_optim=False, config=config.generator).cuda()
    else:
      self.G = SharedGeneratorNoSkip(
        resolution=resolution, no_optim=False, config=config.generator).cuda()

    self.G_optim = self.G.optim
    self.myargs.checkpoint_dict['G'] = self.G
    self.myargs.checkpoint_dict['G_optim'] = self.G.optim
    if getattr(config, 'use_ema', False):
      self.G_ema = copy.deepcopy(self.G)
      self.myargs.checkpoint_dict['G_ema'] = self.G_ema
      self.ema = ema_model.EMA(self.G, self.G_ema,
                               decay=config.ema_decay,
                               start_itr=config.ema_start)

    # Create controller
    controller_c = config.controller
    self.controller = Controller(
      n_classes=n_classes,
      num_layers=(self.G.num_layers + 1 if self.G.output_sample_arc
                  else self.G.num_layers),
      num_branches=len(self.G.ops),
      config=controller_c).cuda()
    self.controller_optim = self.controller.optim
    self.myargs.checkpoint_dict['controller'] = self.controller
    self.myargs.checkpoint_dict['C_optim'] = self.controller.optim

    self.G_C = G_Controller(self.G, self.controller)
    # self.G_C = torch.nn.DataParallel(self.G_C)

    if getattr(config, 'use_cdisc', False):
      disc_c = config.discriminator_cond
      D_activation = activation_dict[disc_c.D_activation]
      self.D = BigGAN.Discriminator(
        **{**disc_c,
           'resolution': resolution,
           'n_classes': n_classes,
           'D_activation': D_activation},
        **disc_c.optimizer).cuda()
      self.D_optim = self.D.optim
    else:
      from ..models.autogan_cifar10_a import Discriminator
      from ..models import optimizer_dict
      disc_c = config.discriminator
      disc_optim_c = disc_c.optimizer
      self.D = Discriminator(args=disc_c).cuda()
      adam_eps = getattr(disc_optim_c, 'adam_eps', 1.e-8)
      self.D_optim = self.optim = optimizer_dict[disc_optim_c.type](
        params=self.D.parameters(), lr=disc_optim_c.D_lr,
        betas=(disc_optim_c.D_B1, disc_optim_c.D_B2),
        eps=adam_eps)

    self.myargs.checkpoint_dict['D'] = self.D
    self.myargs.checkpoint_dict['D_optim'] = self.D_optim
    self.D = torch.nn.DataParallel(self.D)

    self.models = {'controller': self.controller,
                   'G': self.G,
                   'D': self.D}
    self.print_number_params(models=self.models)

  def optimizer_create(self):
    return
    # args = self.config.optimizer
    # # https://github.com/melodyguan/enas/blob/master/src/utils.py#L218
    # controller_optimizer = torch.optim.Adam(params=self.controller.parameters(),
    #                                         lr=args.controller_lr,
    #                                         betas=(0.0, 0.999),
    #                                         eps=1e-3)
    # self.controller_optimizer = controller_optimizer
    # self.myargs.checkpoint_dict['controller_optimizer'] = controller_optimizer
    #
    # # https://github.com/melodyguan/enas/blob/master/src/utils.py#L213
    # shared_cnn_optimizer = torch.optim.SGD(params=self.shared_cnn.parameters(),
    #                                        lr=args.child_lr_max,
    #                                        momentum=0.9,
    #                                        nesterov=True,
    #                                        weight_decay=args.child_l2_reg)
    # self.shared_cnn_optimizer = shared_cnn_optimizer
    # self.myargs.checkpoint_dict['shared_cnn_optimizer'] = shared_cnn_optimizer
    #
    # # https://github.com/melodyguan/enas/blob/master/src/utils.py#L154
    # shared_cnn_scheduler = CosineAnnealingLR(optimizer=shared_cnn_optimizer,
    #                                          T_max=args.child_lr_T,
    #                                          eta_min=args.child_lr_min)
    # self.shared_cnn_scheduler = shared_cnn_scheduler

  def noise_create(self):
    config = self.config.noise
    batch_size = self.config.dataset.batch_size
    self.z_ = gan_utils.z_normal(
      batch_size=batch_size,
      dim_z=self.config.model.generator.dim_z,
      z_mean=config.z_mean, z_var=config.z_var)
    self.y_ = gan_utils.y_categorical(batch_size=batch_size,
                                      nclasses=self.n_classes)
    config.gen_batch_size = getattr(config, 'gen_batch_size', batch_size)
    self.z_gen = gan_utils.z_normal(
      batch_size=config.gen_batch_size,
      dim_z=self.config.model.generator.dim_z,
      z_mean=config.z_mean, z_var=config.z_var)
    self.y_gen = gan_utils.y_categorical(batch_size=config.gen_batch_size,
                                         nclasses=self.n_classes)

    test_batch_size = self.config.noise.test_batch_size
    self.z_test = gan_utils.z_normal(
      batch_size=test_batch_size,
      dim_z=self.config.model.generator.dim_z,
      z_mean=config.z_mean, z_var=config.z_var)
    self.y_test = gan_utils.y_categorical(batch_size=test_batch_size,
                                          nclasses=self.n_classes)

    # Prepare a fixed z & y to see individual sample evolution throghout training
    self.fixed_z = gan_utils.z_normal(
      batch_size=200,
      dim_z=self.config.model.generator.dim_z,
      z_mean=config.z_mean, z_var=config.z_var)
    self.fixed_y = gan_utils.y_categorical(batch_size=200,
                                           nclasses=self.n_classes)
    self.fixed_z.sample_()
    self.fixed_y.sample_()

  def inception_metrics_func_create_pytorch(self):
    config = self.config.inception_metric
    if not hasattr(config, 'saved_inception_moments'):
      return
    self.logger.info_msg('Load inception moments: %s',
                     config.saved_inception_moments)
    # copy fid stat
    # modelarts_utils.modelarts_copy_data(
    #   datapath=config.saved_inception_moments)
    # assert os.path.exists(os.path.expanduser(config.saved_inception_moments))
    # # copy inception_v3 model
    # modelarts_utils.modelarts_copy_data(
    #   datapath=config.torch_inception_v3, datapath_obs=config.torch_inception_v3_obs)
    inception_metrics = inception_utils.InceptionMetrics(
      saved_inception_moments=config.saved_inception_moments)
    self.inception_net = inception_metrics.net
    self.data_mu = inception_metrics.data_mu
    self.data_sigma = inception_metrics.data_sigma

    inception_metrics = functools.partial(
      inception_metrics,
      num_inception_images=config.num_inception_images,
      num_splits=10, prints=True)
    self.inception_metrics = inception_metrics

  def inception_metrics_func_create_tf(self):
    config = self.config.inception_metric
    if not hasattr(config, 'tf_inception_model_dir'):
      return
    self.logger.info_msg('Load tf_inception_mode: %s',
                         config.tf_inception_model_dir)
    modelarts_utils.modelarts_copy_data(
      datapath=config.tf_inception_model_dir)
    eval_IS = getattr(config, 'eval_IS', True)
    if eval_IS:
      self.IS = InceptionScore(
        tf_inception_model_dir=config.tf_inception_model_dir)
    self.FID = FIDScore(tf_inception_model_dir=config.tf_inception_model_dir)

    self.logger.info_msg('Load fid_stat: %s', config.fid_stat)
    modelarts_utils.modelarts_copy_data(datapath=config.fid_stat)
    assert os.path.exists(os.path.expanduser(config.fid_stat))
    self.fid_stat = config.fid_stat
    self.num_inception_images = config.num_inception_images

  def train(self):
    args = self.config.train
    if args.type == 'train_enas':
      self.train_enas()
    elif args.type == 'retrain_cgan':
      self.retrain_cgan()
    elif args.type == 'sample_images':
      self.sample_images_for_fixed_z()
    else:
      assert 0

  def train_enas(self, ):
    """Perform architecture search by training a controller and shared_cnn.
    """
    args = self.config.train_enas
    baseline = None
    self.modelarts(join=True)
    for epoch in range(self.train_dict['epoch_shared_generator'], args.num_epochs):
      self.logger.info('Searching epoch: [%d/%d]', epoch ,args.num_epochs)

      if args.gan_type == 'hingeloss':
        self.train_shared_generator_hingeloss()
      elif args.gan_type == 'wgangp':
        self.train_shared_generator_wgangp()
      else:
        assert 0
      self.train_dict['epoch_shared_generator'] += 1

      random_search = getattr(args, 'random_search', False)
      if epoch + 1 >= args.train_controller_start and (not random_search):
        baseline = self.train_controller(baseline)

      if epoch % args.eval_every_epochs == 0 or epoch == args.num_epochs - 1:
        self.evaluate_model_pytorch()
      self.modelarts(join=False)
    self.modelarts(join=True, end=True)

  def retrain_cgan(self, ):
    """Perform architecture search by training a controller and shared_cnn.
    """
    args = self.config.retrain_cgan
    self.finetune()
    self.modelarts(join=True)
    fixed_arc = self.get_fixed_arc(args.fixed_arc_epoch)
    self.logger.info('Using fixed_arc: %s'%args.fixed_arc_epoch)
    self.logger.info_msg('\n' + str(fixed_arc))
    self.revise_num_parameters(fixed_arc=fixed_arc)

    for epoch in range(self.train_dict['epoch_shared_generator'],
                       args.num_epochs):
      self.logger.info(
        'retrain_cgan epoch: [%d/%d]', epoch, args.num_epochs)
      if args.gan_type == 'hingeloss':
        self.train_shared_generator_hingeloss(fixed_arc=fixed_arc)
      elif args.gan_type == 'wgangp':
        self.train_shared_generator_wgangp(fixed_arc=fixed_arc)
      elif args.gan_type == 'relhinge':
        self.train_shared_generator_relhinge(fixed_arc=fixed_arc)
      else:
        assert 0
      self.train_dict['epoch_shared_generator'] += 1

      if epoch % args.eval_every_epochs == 0 or epoch == args.num_epochs - 1:
        if args.eval_pytorch:
          self.evaluate_model_pytorch(fixed_arc=fixed_arc)
        if args.eval_tf:
          self.evaluate_model_tf(fixed_arc=fixed_arc)

      if getattr(args, 'sample_images', False):
        self.sample_images_for_fixed_z_(fixed_arc=fixed_arc)

      self.modelarts(join=False)
    self.modelarts(join=True, end=True)

  def train_shared_generator_wgangp(self, fixed_arc=None):
    """Train shared_cnn by sampling architectures from the controller.

    Args:
        fixed_arc: Architecture to train, overrides the controller sample
    Returns: Nothing.
    """
    args = self.config.train_shared_generator
    if args.dummy:
      self.logger.info_msg("Dummy train_shared_generator_wgangp.")
      return

    self.controller.eval()
    # Use the full training set when training a fixed architecture
    train_loader = self.data_loaders['train_dataset']
    batch_size = self.config.dataset.batch_size
    pbar = tqdm.tqdm(train_loader, file=self.myargs.stdout,
                     desc='train_shared_generator_wgangp')
    for i, (images, labels) in enumerate(pbar):
      if len(images) != batch_size:
        return
      summary_d = collections.defaultdict(dict)

      real = images.cuda()
      dy = labels.cuda()
      gy = dy

      self.G.train()
      if hasattr(self, 'G_ema'):
        self.G_ema.train()
      self.controller.eval()
      self.D.train()
      self.D.zero_grad()

      d_real = self.D(real, dy)
      d_real_mean = d_real.mean()
      summary_d['d_logit_mean']['d_real_mean'] = d_real_mean.item()

      self.z_.sample_()
      cbn = getattr(self.config.model, 'use_cbn', False)
      fake = self.G_C(self.z_, gy, train_C=False, train_G=False,
                      fixed_arc=fixed_arc, same_in_batch=False, cbn=cbn)
      d_fake = self.D(fake.detach(), gy)
      d_fake_mean = d_fake.mean()
      summary_d['d_logit_mean']['d_fake_mean'] = d_fake_mean.item()

      gp = gan_losses.wgan_gp_gradient_penalty_cond(
        x=real, G_z=fake, gy=gy, f=self.D, backward=True, gp_lambda=10.)
      summary_d['scalars']['gp'] = gp.item()

      wd = d_real_mean - d_fake_mean
      summary_d['wd']['wd'] = wd.item()
      d_loss = -wd

      d_loss.backward()
      self.D_optim.step()
      summary_d['scalars']['d_loss'] = d_loss.item()

      ############################
      # (2) Update G network
      ###########################
      if i % args.n_critic == 0:
        self.G_C.zero_grad()
        self.z_.sample_()
        gy = dy
        # self.y_.sample_()

        if fixed_arc is None:
          fake = nn.parallel.data_parallel(
            self.G, (self.z_, self.G_C.sample_arc))
          # fake = self.G(self.z_, self.G_C.sample_arc)
        else:
          fake = self.G_C(self.z_, gy=gy, train_G=True,
                          fixed_arc=fixed_arc, cbn=cbn)
        d_fake_g = self.D(fake, gy)
        d_fake_g_mean = d_fake_g.mean()
        summary_d['d_logit_mean']['d_fake_g_mean'] = d_fake_g_mean.item()
        g_loss = -d_fake_g_mean
        g_loss.backward()
        summary_d['scalars']['g_loss'] = g_loss.item()

        if getattr(args, 'child_grad_bound', 0) > 0:
          grad_norm = torch.nn.utils.clip_grad_norm_(self.G.parameters(),
                                                     args.child_grad_bound)
          summary_d['scalars']['grad_norm'] = grad_norm

        self.G_optim.step()
        if hasattr(self, 'ema'):
          self.ema.update(self.train_dict['batch_shared_generator'])

      self.train_dict['batch_shared_generator'] += 1

      if i % args.log_every == 0:
        # save checkpoint
        # self.save_checkpoint()
        self.summary_dicts(summary_dicts=summary_d,
                           prefix='train_shared_generator',
                           step=self.train_dict['batch_shared_generator'])

  def train_shared_generator_hingeloss(self, fixed_arc=None):
    """Train shared_cnn by sampling architectures from the controller.

    Args:
        fixed_arc: Architecture to train, overrides the controller sample
    Returns: Nothing.
    """
    args = self.config.train_shared_generator
    if args.dummy:
      self.logger.info("Dummy train_shared_cnn.")
      return

    self.controller.eval()
    # Use the full training set when training a fixed architecture
    train_loader = self.data_loaders['train_dataset']
    batch_size = self.config.dataset.batch_size
    pbar = tqdm.tqdm(train_loader, file=self.myargs.stdout,
                     desc='train_shared_generator_hingeloss')
    for i, (images, labels) in enumerate(pbar):
      if len(images) != batch_size:
        return
      summary_d = collections.defaultdict(dict)

      real = images.cuda()
      dy = labels.cuda()
      gy = dy

      self.G.train()
      self.controller.eval()
      self.D.train()
      self.D.zero_grad()

      d_real = self.D(real, dy)

      self.z_.sample_()
      fake = self.G_C(self.z_, gy, train_C=False, train_G=False,
                      fixed_arc=fixed_arc, same_in_batch=False)
      d_fake = self.D(fake.detach(), gy)

      r_logit_mean, f_logit_mean, D_loss = gan_losses.hinge_loss_discriminator(
        r_logit=d_real, f_logit=d_fake)
      summary_d['d_logit_mean']['r_logit_mean'] = r_logit_mean.item()
      summary_d['d_logit_mean']['f_logit_mean'] = f_logit_mean.item()
      summary_d['scalars']['D_loss'] = D_loss.item()

      D_loss.backward()
      self.D_optim.step()

      ############################
      # (2) Update G network
      ###########################
      if i % args.n_critic == 0:
        self.G_C.zero_grad()
        self.z_.sample_()
        z = self.z_
        # self.z_gen.sample_()
        # self.y_gen.sample_()
        # gy = self.y_gen
        gy = dy
        # fake = self.G_C(self.z_, gy, train_C=False, fixed_arc=fixed_arc)
        if fixed_arc is None:
          fake = self.G(z, self.G_C.sample_arc)
        else:
          fake = self.G_C(z, gy=gy, train_G=True, fixed_arc=fixed_arc)
        d_fake_g = self.D(fake, gy)
        G_f_logit_mean, G_loss = \
          gan_losses.hinge_loss_generator(f_logit=d_fake_g)
        summary_d['d_logit_mean']['G_f_logit_mean'] = G_f_logit_mean.item()
        summary_d['scalars']['G_loss'] = G_loss.item()
        G_loss.backward()
        if getattr(args, 'child_grad_bound', 0) > 0:
          grad_norm = torch.nn.utils.clip_grad_norm_(self.G.parameters(),
                                                     args.child_grad_bound)
          summary_d['scalars']['grad_norm'] = grad_norm
        self.G_optim.step()

        if hasattr(self, 'ema'):
          self.ema.update(self.train_dict['batch_shared_generator'])

      self.train_dict['batch_shared_generator'] += 1

      if i % args.log_every == 0:
        # save checkpoint
        # self.save_checkpoint()
        self.summary_dicts(summary_dicts=summary_d,
                           prefix='train_shared_generator',
                           step=self.train_dict['batch_shared_generator'])

  def train_shared_generator_relhinge(self, fixed_arc=None):
    """Train shared_cnn by sampling architectures from the controller.

    Args:
        fixed_arc: Architecture to train, overrides the controller sample
    Returns: Nothing.
    """
    args = self.config.train_shared_generator
    if args.dummy:
      self.logger.info("Dummy train_shared_cnn.")
      return

    self.controller.eval()
    # Use the full training set when training a fixed architecture
    train_loader = self.data_loaders['train_dataset']
    loss_func = GANLosses.RelativisticAverageHingeGAN(dis=self.D)
    batch_size = self.config.dataset.batch_size
    pbar = tqdm.tqdm(train_loader, file=self.myargs.stdout)
    for i, (images, labels) in enumerate(pbar):
      if len(images) != batch_size:
        return
      summary_d = collections.defaultdict(dict)

      real = images.cuda()
      dy = labels.cuda()
      gy = dy

      self.G.train()
      self.controller.eval()
      self.D.train()
      self.D.zero_grad()

      self.z_.sample_()
      fake = self.G_C(self.z_, gy, train_C=False, train_G=False,
                      fixed_arc=fixed_arc, same_in_batch=False)

      dis_loss, dis_sum_dict = loss_func.dis_loss_cond(
        real_samps=real, ry=dy, fake_samps=fake.detach(), fy=gy)

      summary_d['logit_mean']['dis_real_logit_mean'] = \
        dis_sum_dict['dis_real_logit_mean']
      summary_d['logit_mean']['dis_fake_logit_mean'] = \
        dis_sum_dict['dis_fake_logit_mean']
      summary_d['scalars']['dis_loss'] = dis_loss.item()

      dis_loss.backward()
      self.D_optim.step()

      ############################
      # (2) Update G network
      ###########################
      if i % args.n_critic == 0:
        self.G_C.zero_grad()
        self.z_.sample_()
        gy = dy
        # self.y_.sample_()
        # fake = self.G_C(self.z_, gy, train_C=False, fixed_arc=fixed_arc)
        if fixed_arc is None:
          fake = self.G(self.z_, self.G_C.module.sample_arc)
        else:
          fake = self.G_C(self.z_, gy=gy, train_G=True,
                          fixed_arc=fixed_arc)
        gen_loss, gen_sum_dict = loss_func.gen_loss_cond(
          real_samps=real, ry=dy, fake_samps=fake, fy=gy)

        summary_d['logit_mean']['gen_real_logit_mean'] = \
          gen_sum_dict['gen_real_logit_mean']
        summary_d['logit_mean']['gen_fake_logit_mean'] = \
          gen_sum_dict['gen_fake_logit_mean']
        summary_d['scalars']['gen_loss'] = gen_loss.item()

        gen_loss.backward()
        if getattr(args, 'child_grad_bound', 0) > 0:
          grad_norm = torch.nn.utils.clip_grad_norm_(self.G.parameters(),
                                                     args.child_grad_bound)
          summary_d['scalars']['grad_norm'] = grad_norm
        self.G_optim.step()

      self.train_dict['batch_shared_generator'] += 1

      if i % args.log_every == 0:
        # save checkpoint
        self.save_checkpoint()
        self.summary_dicts(summary_dicts=summary_d,
                           prefix='train_shared_generator',
                           step=self.train_dict['batch_shared_generator'])

  def train_controller(self, baseline=None):
    """Train controller to optimizer validation accuracy using REINFORCE.

    Args:
        baseline: The baseline score (i.e. average val_acc) from the
        previous epoch

    Returns:
        baseline: The baseline score (i.e. average val_acc) for the
        current epoch

    For more stable training we perform weight updates using the average of
    many gradient estimates. controller_num_aggregate indicates how many samples
    we want to average over (default = 20). By default PyTorch will sum gradients
    each time .backward() is called (as long as an optimizer step is not taken),
    so each iteration we divide the loss by controller_num_aggregate to get the
    average.

    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py#L270
    """
    args = self.config.train_controller
    if args.dummy:
      self.logger.info("Dummy train_controller")
      return

    meter_dict = collections.defaultdict(AverageMeter)

    self.G.eval()
    self.controller.train()

    self.controller.zero_grad()
    pbar = tqdm.tqdm(range(args.train_steps * args.num_aggregate),
                     desc='train_controller', file=self.myargs.stdout)
    for i in pbar:
      self.z_test.sample_()
      self.y_test.sample_()
      x, sample_entropy, sample_log_prob = self.G_C(
        self.z_test, self.y_test, train_C=True, train_G=False,
        same_in_batch=True,
        return_sample_entropy=True, return_sample_log_prob=True)
      pool, logits, labels = [], [], []
      with torch.no_grad():
        pool_val, logits_val = self.inception_net(x)
        pool += [np.asarray(pool_val.cpu())]
        logits += [np.asarray(F.softmax(logits_val, 1).cpu())]
        labels += [np.asarray(self.y_.cpu())]
      pool, logits, labels = [np.concatenate(item, 0) for item in \
                              [pool, logits, labels]]
      reward_g = self.calculate_IS(logits)
      meter_dict['reward_g'].update(reward_g)

      # detach to make sure that gradients aren't backpropped through the reward
      reward = torch.tensor(reward_g).cuda()
      sample_entropy_mean = sample_entropy.mean()
      meter_dict['sample_entropy'].update(sample_entropy_mean.item())
      reward += args.entropy_weight * sample_entropy_mean

      if baseline is None:
        baseline = reward_g
      else:
        baseline -= (1 - args.bl_dec) * (baseline - reward)
        # detach to make sure that gradients are not backpropped through the baseline
        baseline = baseline.detach()

      sample_log_prob_mean = sample_log_prob.mean()
      meter_dict['sample_log_prob'].update(sample_log_prob_mean.item())
      loss = -1 * sample_log_prob_mean * (reward - baseline)
      meter_dict['reward'].update(reward.item())
      meter_dict['baseline'].update(baseline.item())

      meter_dict['loss'].update(loss.item())

      # Average gradient over controller_num_aggregate samples
      loss = loss / args.num_aggregate

      loss.backward(retain_graph=False)

      # Aggregate gradients for controller_num_aggregate iterationa, then update weights
      if (i + 1) % args.num_aggregate == 0:
        self.train_dict['batch_controller'] += 1
        grad_norm = torch.nn.utils.clip_grad_norm_(self.controller.parameters(),
                                                   args.child_grad_bound)
        self.controller.optim.step()
        self.controller.zero_grad()

        summary_scalars = {'grad_norm': grad_norm}
        summary_reward_baseline = {}
        for meter_k, meter in meter_dict.items():
          if meter_k in ['reward', 'baseline']:
            summary_reward_baseline[meter_k] = meter.avg
          else:
            summary_scalars[meter_k] = meter.avg
          meter.reset()
        self.summary_scalars(summary=summary_scalars,
                             prefix='train_controller',
                             step=self.train_dict['batch_controller'])
        self.summary_scalars_together(
          summary=summary_reward_baseline,
          prefix='train_controller',
          step=self.train_dict['batch_controller'])
    # end for

    self.G.train()
    return baseline

  @staticmethod
  def sample(G, z, y, arcs, parallel=True):
    G.eval()
    with torch.no_grad():
      z.sample_()
      y.sample_()
      sample_arcs = arcs[y]
      if parallel:
        G_z = nn.parallel.data_parallel(G, (z, sample_arcs))
      else:
        G_z = G(z, sample_arcs=sample_arcs)
    return G_z

  @staticmethod
  def sample_cbn(G, z, y, arcs, parallel=False):
    G.eval()
    with torch.no_grad():
      z.sample_()
      y.sample_()
      sample_arcs = arcs[y]
      G_z = G(z, y, sample_arcs=sample_arcs)
    return G_z

  def evaluate_model_pytorch(self, fixed_arc=None):
    """Print the validation and test accuracy for a controller and shared_cnn.
    """
    if not hasattr(self, 'inception_metrics'):
      return
    self.G_C.eval()
    self.controller.eval()
    self.G.eval()
    G_sample = self.G
    if hasattr(self, 'G_ema'):
      self.G_ema.eval()
      G_sample = self.G_ema

    # Print sample arcs
    if fixed_arc is not None:
      arcs = torch.from_numpy(fixed_arc).cuda()
    else:
      y = torch.arange(0, self.n_classes, dtype=torch.int)
      self.controller(y, determine_sample=False)
      arcs = self.controller.sample_arc

    self.myargs.logger.info_msg('evaluating pytorch arcs: \n%s'%arcs.cpu().numpy())
    self.myargs.textlogger.logstr(
      self.train_dict['epoch_shared_generator'],
      arcs='\n' + np.array2string(arcs.cpu().numpy(), threshold=np.inf))

    sample_func = functools.partial(
      self.sample, G=G_sample, z=self.z_test, y=self.y_test, arcs=arcs,
      parallel=True)

    IS_mean, IS_std, FID = self.inception_metrics(
      use_torch=False, sample_func=sample_func,
      show_process=True, stdout=self.myargs.stdout)
    self.logger.info_msg('IS_mean: %f +- %f, FID: %f', IS_mean, IS_std, FID)
    summary = {'IS_mean': IS_mean, 'IS_std': IS_std, 'FID': FID}
    self.summary_scalars(summary=summary, prefix='evaluate_model_pytorch',
                         step=self.train_dict['epoch_shared_generator'])

    self.controller.train()
    self.G.train()

  def evaluate_model_tf(self, fixed_arc=None):
    """Print the validation and test accuracy for a controller and shared_cnn.
    """

    self.G_C.eval()
    self.controller.eval()
    self.G.eval()
    G_sample = self.G
    if hasattr(self, 'G_ema'):
      self.G_ema.eval()
      G_sample = self.G_ema

    # Print sample arcs
    if fixed_arc is not None:
      arcs = torch.from_numpy(fixed_arc).cuda()
    else:
      y = torch.range(0, self.n_classes - 1, dtype=torch.int)
      self.controller(y, determine_sample=False)
      arcs = self.controller.sample_arc

    self.myargs.logger.info_msg('evaluating tensorflow arcs: \n%s' % arcs.cpu().numpy())
    self.myargs.textlogger.logstr(
      self.train_dict['epoch_shared_generator'],
      arcs='\n' + np.array2string(arcs.cpu().numpy(), threshold=np.inf))

    cbn = getattr(self.config.model, 'use_cbn', False)
    if cbn:
      sample_func = functools.partial(
        self.sample_cbn, G=G_sample, z=self.z_test, y=self.y_test, arcs=arcs,
        parallel=True)
    else:
      sample_func = functools.partial(
        self.sample, G=G_sample, z=self.z_test, y=self.y_test, arcs=arcs,
        parallel=True)

    img_list = self.FID.get_sample_imgs_list(
      sample_func=sample_func, num_imgs=self.num_inception_images,
      stdout=self.myargs.stdout)
    summary = {}
    if hasattr(self, 'IS'):
      # get inception score
      IS_mean, IS_std = self.IS.get_inception_score(
        img_list, bs=50, stdout=self.myargs.stdout)
      print(f"Inception score: mean={IS_mean}, std={IS_std}")
      summary = {'IS_mean': IS_mean, 'IS_std': IS_std}

    # get fid score
    FID = self.FID.calculate_fid_given_paths(
      fid_buffer=img_list, fid_stat=self.fid_stat, stdout=self.myargs.stdout)
    print(f"FID score: {FID}")
    summary.update({'FID': FID})
    self.summary_scalars(summary, prefix='evaluate_model_tf',
                         step=self.train_dict['epoch_shared_generator'])

    if FID < self.train_dict['best_FID']:
      self.train_dict['best_FID'] = FID
      self.summary_scalars(summary={'best_FID': FID},
                           prefix='evaluate_model_tf',
                           step=self.train_dict['epoch_shared_generator'])
      self.save_checkpoint(
        filename=f'ckpt_epoch_%d.tar'%(self.train_dict['epoch_shared_generator']))
    if hasattr(self, 'IS'):
      if IS_mean > self.train_dict['best_IS']:
        self.train_dict['best_IS'] = IS_mean
        self.summary_scalars(summary={'best_IS': IS_mean},
                             prefix='evaluate_model_tf',
                             step=self.train_dict['epoch_shared_generator'])
        self.save_checkpoint(
          filename=f'ckpt_epoch_%d.tar' % (
          self.train_dict['epoch_shared_generator']))

    self.controller.train()
    self.G.train()

  def is_repetitive_between_classes(self, sample_arc):
    n_class, n_layer = sample_arc.shape
    for i in range(n_class):
      for j in range(i+1, n_class):
        if sum(sample_arc[i] == sample_arc[j]) > n_layer - 1:
          return True
    return False

  def revise_num_parameters(self, fixed_arc):
    if hasattr(self.G, 'layers_para_matrix'):
      param_matrix = self.G.layers_para_matrix
      n_classes, n_layers = fixed_arc.shape
      for layer_id in range(n_layers):
        for class_id in range(n_classes):
          op_id = fixed_arc[class_id, layer_id]
          param_matrix[op_id, layer_id] = 0
      del_num_para = sum(sum(param_matrix))
      all_num_para = sum([p.data.nelement() for p in self.G.parameters()])
      self.print_number_params(dict(G=self.G, D=self.D))
      self.logger.info('Revised number of params in G:\t {}M'.format(
        (all_num_para - del_num_para) / 1e6))


  def get_fixed_arc(self, fixed_arc_epoch, finetune_path=None):
    if isinstance(fixed_arc_epoch, str):
      if fixed_arc_epoch == 'random':
        while True:
          sample_arc = np.random.randint(0, len(self.G.ops),
                                         (self.n_classes, self.G.num_layers),
                                         dtype=int)
          if not self.is_repetitive_between_classes(sample_arc=sample_arc):
            break
      elif fixed_arc_epoch in self.G.ops:
        # single op
        op_idx = self.G.ops.index(fixed_arc_epoch)
        sample_arc = \
          np.ones((self.n_classes, self.G.num_layers), dtype=int) * op_idx
      else:
        import json
        arc_list = json.loads(
          fixed_arc_epoch.replace(' ', ', ').replace('][', '], [').strip())
        sample_arc = np.array(arc_list)
      return sample_arc
    if finetune_path is None:
      finetune_path = self.args.finetune_path
    arcs_file = os.path.join(finetune_path, 'textlog/arcs.log')
    n_classes = self.n_classes
    with open(arcs_file) as f:
      while True:
        epoch_str = f.readline().strip(': \n')
        sample_arc = []
        for _ in range(n_classes):
          class_arc = f.readline().strip('[\n ]')
          sample_arc.append(np.fromstring(class_arc, dtype=int, sep=' '))
        if fixed_arc_epoch == int(epoch_str):
          break
    sample_arc = np.array(sample_arc)
    return sample_arc

  def calculate_FID(self, pool):
    mu, sigma = np.mean(pool, axis=0), np.cov(pool, rowvar=False)
    FID = inception_utils.numpy_calculate_frechet_distance(
      mu, sigma,
      self.data_mu, self.data_sigma)
    return FID

  def calculate_IS(self, logits):
    IS_mean, IS_std = inception_utils.calculate_inception_score(logits)
    return IS_mean

  def sample_images_for_fixed_z_(self, fixed_arc=None):
    if not hasattr(self.config, 'sample_images_for_fixed_z'):
      return
    config = self.config.sample_images_for_fixed_z

    y = torch.arange(0, self.n_classes).view(-1, 1)
    y = y.repeat(config.num_z, 1).view(-1).cuda()

    z = self.fixed_z[0:len(y)]

    cbn = getattr(self.config.model, 'use_cbn', False)
    fake = self.G_C(z, y, train_C=False, train_G=False, fixed_arc=fixed_arc,
                    cbn=cbn)
    fake_merge = torchvision.utils.make_grid(
      fake, nrow=self.n_classes, pad_value=1, normalize=True)
    self.myargs.writer.add_image('fixed_samples', fake_merge, self.myargs.train_dict['epoch_shared_generator'])
    torchvision.utils.save_image(
      fake, os.path.join(self.args.imgdir,
                         'sampel_images_epoch_%d.png'%self.myargs.train_dict['epoch_shared_generator']),
      nrow=self.n_classes, normalize=True, pad_value=1)
    pass

  def sample_images_for_fixed_z(self):
    finetune_path = 'results/enas_cgan/test_train_cgan_gen_mnist_v4_20191030-11_58_54'
    resume_model_path = os.path.join(finetune_path, 'models/ckpt_epoch_61.tar')

    self.resume(myargs=self.myargs, resume_path=resume_model_path)
    fixed_arc = self.get_fixed_arc(fixed_arc_epoch=1, finetune_path=finetune_path)
    self.logger.info('Using fixed_arc: \n%s' % finetune_path)
    self.logger.info_msg('\n' + str(fixed_arc))
    self.sample_images_for_fixed_z_(fixed_arc=fixed_arc)
    pass

