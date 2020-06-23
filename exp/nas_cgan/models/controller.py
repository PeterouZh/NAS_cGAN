import os
import unittest

import yaml
from easydict import EasyDict
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions.categorical import Categorical

from template_lib.trainer import get_ddp_attr


class Controller(nn.Module):
  '''
  https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py
  '''

  def __init__(self, n_classes, num_layers, num_branches,
               search_whole_channels=True, config=None):
    super(Controller, self).__init__()

    self.n_classes = n_classes
    self.search_whole_channels = search_whole_channels
    self.num_layers = num_layers
    self.num_branches = num_branches

    self.lstm_size = config.lstm_size
    self.lstm_num_layers = config.lstm_num_layers
    self.tanh_constant = config.tanh_constant
    self.temperature = config.temperature

    self._create_params()
    self._reset_params()

    optim_c = config.optimizer
    self.optim = torch.optim.Adam(
      params=self.parameters(),
      lr=optim_c.lr,
      betas=(optim_c.B1, optim_c.B2),
      eps=optim_c.adam_eps)

  def _create_params(self):
    '''
    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py#L83
    '''
    self.w_lstm = nn.LSTM(input_size=self.lstm_size,
                          hidden_size=self.lstm_size,
                          num_layers=self.lstm_num_layers)
    # Learn the starting input
    self.g_emb = nn.Embedding(self.n_classes, self.lstm_size)

    if self.search_whole_channels:
      self.w_emb = nn.Embedding(self.num_branches, self.lstm_size)
      self.w_soft = nn.Linear(self.lstm_size, self.num_branches, bias=False)
    else:
      assert False, "Not implemented error: search_whole_channels = False"

    # self.w_attn_1 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
    # self.w_attn_2 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
    # self.v_attn = nn.Linear(self.lstm_size, 1, bias=False)

  def _reset_params(self):
    for m in self.modules():
      if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.1, 0.1)

    nn.init.uniform_(self.w_lstm.weight_hh_l0, -0.1, 0.1)
    nn.init.uniform_(self.w_lstm.weight_ih_l0, -0.1, 0.1)

  def forward(self, class_ids, determine_sample=False):
    '''
    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py#L126
    '''
    h0 = None  # setting h0 to None will initialize LSTM state with 0s
    arc_seq = []
    entropys = []
    log_probs = []
    if isinstance(class_ids, int):
      class_ids = [class_ids]
    if isinstance(class_ids, list):
      class_ids = torch.tensor(class_ids, dtype=torch.int64)
    class_ids = class_ids.type(torch.int64)
    inputs = self.g_emb.weight[class_ids]

    for layer_id in range(self.num_layers):
      if self.search_whole_channels:
        inputs = inputs.unsqueeze(dim=0)
        output, hn = self.w_lstm(inputs, h0)
        output = output.squeeze(dim=0)
        h0 = hn

        logit = self.w_soft(output)
        if self.temperature > 0:
          logit /= self.temperature
        if self.tanh_constant is not None:
          logit = self.tanh_constant * torch.tanh(logit)

        branch_id_dist = Categorical(logits=logit)
        if determine_sample:
          branch_id = logit.argmax(dim=1)
        else:
          branch_id = branch_id_dist.sample()

        arc_seq.append(branch_id)

        log_prob = branch_id_dist.log_prob(branch_id)
        log_probs.append(log_prob.view(-1))
        entropy = branch_id_dist.entropy()
        entropys.append(entropy.view(-1))

      else:
        # https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py#L171
        assert False, "Not implemented error: search_whole_channels = False"

      # Calculate average of class and branch embedding
      # and use it as input for next step
      inputs = self.w_emb(branch_id) + self.g_emb.weight[class_ids]
      inputs /= 2


    self.sample_arc = torch.stack(arc_seq, dim=1)

    self.sample_entropy = torch.stack(entropys, dim=1)

    self.sample_log_prob = torch.stack(log_probs, dim=1)
    self.sample_prob = self.sample_log_prob.exp()


class G_Controller(nn.Module):
  def __init__(self, G, controller):
    super(G_Controller, self).__init__()
    self.G = G
    self.controller = controller

  def forward(self, z, gy, train_C=False, train_G=True,
              fixed_arc=None, same_in_batch=True,
              return_sample_entropy=False,
              return_sample_log_prob=False,
              determine_sample=False,
              return_sample_arc=False,
              cbn=False):

    if fixed_arc is not None:
      fixed_arc = torch.from_numpy(fixed_arc).cuda()
      self.sample_arc = fixed_arc[gy]
    else:
      with torch.set_grad_enabled(train_C):
        if same_in_batch:
          y_range = torch.arange(
            0, get_ddp_attr(self.controller, 'n_classes'), dtype=torch.int64)
          self.controller(y_range, determine_sample=determine_sample)
          self.sample_arc = get_ddp_attr(self.controller, 'sample_arc')[gy]
        else:
          self.controller(gy, determine_sample=determine_sample)
          self.sample_arc = get_ddp_attr(self.controller, 'sample_arc')

    with torch.set_grad_enabled(train_G):
      if cbn:
        x = self.G(z, gy, self.sample_arc)
      else:
        # x = nn.parallel.data_parallel(self.G, (z, self.sample_arc))
        x = self.G(z, self.sample_arc)

    out = x
    ret_out = (out, )
    if return_sample_entropy:
      sample_entropy = get_ddp_attr(self.controller, 'sample_entropy')
      ret_out = ret_out + (sample_entropy, )
    if return_sample_log_prob:
      sample_log_prob = get_ddp_attr(self.controller, 'sample_log_prob')
      ret_out = ret_out + (sample_log_prob, )
    if return_sample_arc:
      ret_out = ret_out + (self.sample_arc, )
    if len(ret_out) == 1:
      return ret_out[0]
    return ret_out


def test_Controller(args, myargs):
  yaml_file = 'enas_cgan/configs/enas_cgan.yaml'
  with open(yaml_file, 'r') as f:
    configs = EasyDict(yaml.load(f))
  config = configs.train_cgan_cnn_macro_cifar10.model
  resolution = 32
  gen_c = config.generator
  from .supernet import SharedGeneratorNoSkip
  generator = SharedGeneratorNoSkip(
    resolution=resolution, config=gen_c).cuda()

  from enas_cgan.models.controller import Controller
  controller_c = configs.train_cgan_cnn_macro_cifar10.model.controller
  n_classes = 10
  batchsize = 128
  controller = Controller(n_classes=n_classes,
                          num_layers=generator.num_layers,
                          num_branches=len(generator.ops),
                          config=controller_c)
  controller = controller.cuda()
  class_ids = torch.randint(0, n_classes, (batchsize, ))
  with torch.no_grad():
    controller(class_ids)  # perform forward pass to generate a new architecture
  sample_arc = controller.sample_arc
  # sample_arc = torch.randint(0, len(generator.ops), (128, generator.num_layers))

  dummy_data = torch.rand(batchsize, 120).cuda()
  x = generator(dummy_data, sample_arc)

  pass


def test_G_Controller(args, myargs):
  yaml_file = 'enas_cgan/configs/enas_cgan.yaml'
  with open(yaml_file, 'r') as f:
    configs = EasyDict(yaml.load(f))
  config = configs.train_cgan_cnn_macro_cifar10.model
  resolution = 32
  gen_c = config.generator
  from .supernet import SharedGeneratorNoSkip
  generator = SharedGeneratorNoSkip(
    resolution=resolution, config=gen_c).cuda()

  from enas_cgan.models.controller import Controller
  controller_c = configs.train_cgan_cnn_macro_cifar10.model.controller
  n_classes = 10
  batchsize = 128
  controller = Controller(n_classes=n_classes,
                          num_layers=generator.num_layers,
                          num_branches=len(generator.ops),
                          config=controller_c)
  controller = controller.cuda()
  class_ids = torch.randint(0, n_classes, (batchsize,))
  with torch.no_grad():
    controller(class_ids)  # perform forward pass to generate a new architecture
  sample_arc = controller.sample_arc

  G_C = G_Controller(generator, controller)

  from template_lib.gans import gan_utils
  z = gan_utils.z_normal(
    batch_size=64, dim_z=120, z_mean=0, z_var=1)
  y = gan_utils.y_categorical(batch_size=64, nclasses=n_classes)
  z.sample_()
  y.sample_()
  x0, a0 = G_C(z, y, train_C=False, determine_sample=False,
               return_sample_arc=True)
  x1, a1 = G_C(z, y, train_C=False, determine_sample=False,
               return_sample_arc=True)
  x2, a2 = G_C(z, y, train_C=False, determine_sample=True,
               return_sample_arc=True)
  x3, a3 = G_C(z, y, train_C=False, determine_sample=True,
               return_sample_arc=True)

  temp = torch.all(a3 == a3)
  pass

