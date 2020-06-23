import collections, tqdm, os
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from template_lib.trainer import base_trainer
from template_lib.gans import inception_utils, gan_utils, gan_losses

from ..models.controller import Controller
from ..models.shared_cnn import SharedCNN, SharedGenerator
from ..models.supernet import \
  (SharedGeneratorNoSkip,SharedGeneratorCBN)
from ..utils.utils import AverageMeter
from . import imsize_dict, nclass_dict, activation_dict
from ..models import BigGAN
from ..models.controller import G_Controller
from . import train_cnn_macro

class Trainer(train_cnn_macro.Trainer):

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
      fake = self.G_C(self.z_, gy, train_C=False, train_G=False,
                      fixed_arc=fixed_arc, same_in_batch=False)
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
                          fixed_arc=fixed_arc)
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


class Trainer_deprecated(train_cnn_macro.Trainer):

  def train_shared_generator_wgangp(self, fixed_arc=None):
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

      d_real = self.D(real, dy)
      d_real_mean = d_real.mean()
      summary_d['d_logit_mean']['d_real_mean'] = d_real_mean.item()

      self.z_.sample_()
      fake = self.G_C(self.z_, gy, train_C=False, train_G=False,
                      fixed_arc=fixed_arc, same_in_batch=False,
                      cbn=True)
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
          fake = self.G(self.z_, gy, self.G_C.module.sample_arc)
        else:
          fake = self.G_C(self.z_, gy=gy, train_G=True,
                          fixed_arc=fixed_arc, cbn=True)
        d_fake_g = self.D(fake, gy)
        d_fake_g_mean = d_fake_g.mean()
        summary_d['d_logit_mean']['d_fake_g_mean'] = d_fake_g_mean.item()
        g_loss = -d_fake_g_mean
        g_loss.backward()

        if getattr(args, 'child_grad_bound', 0) > 0:
          grad_norm = torch.nn.utils.clip_grad_norm_(self.G.parameters(),
                                                     args.child_grad_bound)
          summary_d['scalars']['grad_norm'] = grad_norm

        self.G_optim.step()
        summary_d['scalars']['g_loss'] = g_loss.item()

      self.train_dict['batch_shared_generator'] += 1

      if i % args.log_every == 0:
        # save checkpoint
        self.save_checkpoint()
        self.summary_dicts(summary_dicts=summary_d,
                           prefix='train_shared_generator',
                           step=self.train_dict['batch_shared_generator'])

      if (i + 1) % (len(train_loader) // 2) == 0:
        self.summary_figures(summary_dicts=summary_d,
                             prefix='train_shared_generator')

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

      d_real = self.D(real, dy)

      self.z_.sample_()
      fake = self.G_C(self.z_, gy, train_C=False, train_G=False,
                      fixed_arc=fixed_arc, same_in_batch=False,
                      cbn=True)
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
        gy = dy
        # self.y_.sample_()
        # fake = self.G_C(self.z_, gy, train_C=False, fixed_arc=fixed_arc)
        if fixed_arc is None:
          fake = self.G(self.z_, gy, self.G_C.module.sample_arc)
        else:
          fake = self.G_C(self.z_, gy=gy, train_G=True,
                          fixed_arc=fixed_arc, cbn=True)
        d_fake_g = self.D(fake, gy)
        G_f_logit_mean, G_loss = \
          gan_losses.hinge_loss_generator(f_logit=d_fake_g)
        summary_d['d_logit_mean']['G_f_logit_mean'] = G_f_logit_mean.item()

        G_loss.backward()
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

      if (i + 1) % (len(train_loader) // 2) == 0:
        self.summary_figures(summary_dicts=summary_d,
                             prefix='train_shared_generator')

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
    pbar = tqdm.tqdm(
      range(args.train_steps * args.num_aggregate),
      file=self.myargs.stdout)
    for i in pbar:
      self.z_test.sample_()
      self.y_test.sample_()
      x, sample_entropy, sample_log_prob = self.G_C(
        self.z_test, self.y_test, train_C=True, train_G=False,
        same_in_batch=True,
        return_sample_entropy=True, return_sample_log_prob=True,
        cbn=True)
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

        summary_scalars = {'tc.grad_norm': grad_norm}
        summary_reward_baseline = {}
        for meter_k, meter in meter_dict.items():
          if meter_k in ['reward', 'baseline']:
            summary_reward_baseline['tc.' + meter_k] = meter.avg
          else:
            summary_scalars['tc.' + meter_k] = meter.avg
          meter.reset()
        self.summary_scalars(summary=summary_scalars,
                             prefix='train_controller',
                             step=self.train_dict['batch_controller'])
        self.summary_scalars_together(
          summary=summary_reward_baseline,
          prefix='train_controller',
          step=self.train_dict['batch_controller'])
    # end for
    self.myargs.textlogger.log_axes(**summary_scalars)
    self.myargs.textlogger.log_ax(**summary_reward_baseline)

    self.G.train()
    return baseline

  @staticmethod
  def sample(G, z, y, arcs):
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
    args = self.config.evaluate_model_pytorch

    self.G_C.eval()
    self.controller.eval()
    self.G.eval()

    # Print sample arcs
    if fixed_arc is not None:
      arcs = torch.from_numpy(fixed_arc).cuda()
    else:
      y = torch.range(0, self.n_classes - 1, dtype=torch.int)
      self.controller(y, determine_sample=False)
      arcs = self.controller.sample_arc

    self.myargs.logger.info_msg('arcs: \n%s'%arcs.cpu().numpy())
    self.myargs.textlogger.logstr(
      self.train_dict['epoch_shared_generator'],
      arcs='\n' + str(arcs.cpu().numpy()))
    self.save_checkpoint()

    sample_func = functools.partial(
      self.sample, G=self.G, z=self.z_test, y=self.y_test, arcs=arcs)

    IS_mean, IS_std, FID = self.inception_metrics(
      show_process=False, use_torch=False, parallel=True,
      sample_func=sample_func)
    self.logger.info_msg('IS_mean: %f +- %f, FID: %f', IS_mean, IS_std, FID)
    summary = {'IS_mean': IS_mean, 'IS_std': IS_std, 'FID': FID}
    self.summary_scalars(summary=summary, prefix='evaluate_mode',
                         step=self.train_dict['epoch_shared_generator'])
    self.myargs.textlogger.log_axes(**summary)
    if FID < self.train_dict['best_FID']:
      self.train_dict['best_FID'] = FID
      self.summary_scalars(summary={'best_FID': FID},
                           prefix='evaluate_mode',
                           step=self.train_dict['epoch_shared_generator'])
    if IS_mean > self.train_dict['best_IS']:
      self.train_dict['best_IS'] = IS_mean
      self.summary_scalars(summary={'best_IS': IS_mean},
                           prefix='evaluate_mode',
                           step=self.train_dict['epoch_shared_generator'])
    self.controller.train()
    self.G.train()







