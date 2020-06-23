from collections import OrderedDict
from easydict import EasyDict
import yaml
import functools
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from . import layers
from .BigGAN import G_arch, D_arch
from .ops import \
  (MixedLayer, MixedLayerCond, UpSample, DownSample, Identity,
   MixedLayerSharedWeights, MixedLayerCondSharedWeights,
   SinglePathLayer)


class SharedGeneratorNoSkip(nn.Module):
  def __init__(self, resolution, no_optim=False,
               config=None):
    super(SharedGeneratorNoSkip, self).__init__()
    self.config = config
    for k, v in config.items():
      assert not hasattr(self, k)
      setattr(self, k, v)

    self.arch = G_arch(config.ch, config.attention)[resolution]

    if self.use_sn:
      self.which_linear = functools.partial(
        layers.SNLinear, num_svs=1, num_itrs=1, eps=1e-6)
      self.which_conv = functools.partial(
        layers.SNConv2d, kernel_size=3, padding=1,
        num_svs=1, num_itrs=1, eps=1e-6)
      self.which_conv_1x1 = functools.partial(
        layers.SNConv2d, kernel_size=1, padding=0,
        num_svs=1, num_itrs=1, eps=1e-6)
    else:
      self.which_linear = nn.Linear
      self.which_conv = functools.partial(
        nn.Conv2d, kernel_size=3, padding=1)
      self.which_conv_1x1 = functools.partial(
        nn.Conv2d, kernel_size=1, padding=0)

    # First linear layer
    self.linear = self.which_linear(
      config.dim_z, self.arch['in_channels'][0] * (config.bottom_width ** 2))

    num_conv_in_block = 2
    self.num_layers = len(self.arch['in_channels']) * num_conv_in_block
    self.upsample_layer_idx = \
      [num_conv_in_block * l
        for l in range(0, self.num_layers//num_conv_in_block)]

    self.layers = nn.ModuleList([])
    self.layers_para_list = []
    self.skip_layers = nn.ModuleList([])
    bn_type = getattr(config, 'bn_type', 'bn')

    for layer_id in range(self.num_layers):
      block_in = self.arch['in_channels'][layer_id//num_conv_in_block]
      block_out = self.arch['out_channels'][layer_id//num_conv_in_block]
      if layer_id % num_conv_in_block == 0:
        in_channels = block_in
        out_channels = block_out
      else:
        in_channels = block_out
        out_channels = block_out
      upsample = (UpSample()
                  if (self.arch['upsample'][layer_id//num_conv_in_block] and
                      layer_id in self.upsample_layer_idx)
                  else None)
      if getattr(self, 'share_conv_weights', False):
        if getattr(self, 'single_path_layer', False):
          layer = SinglePathLayer(
            layer_id=layer_id, in_planes=in_channels, out_planes=out_channels,
            ops=self.ops, track_running_stats=self.track_running_stats,
            scalesample=upsample, bn_type=bn_type, share_bias=self.share_bias)
        else:
          layer = MixedLayerSharedWeights(
            layer_id=layer_id, in_planes=in_channels, out_planes=out_channels,
            ops=self.ops, track_running_stats=self.track_running_stats,
            scalesample=upsample, bn_type=bn_type)
      else:
        layer = MixedLayer(
          layer_id, in_channels, out_channels,
          ops=self.ops, track_running_stats=self.track_running_stats,
          scalesample=upsample, bn_type=bn_type)
      self.layers.append(layer)
      self.layers_para_list.append(layer.num_para_list)
      if layer_id in self.upsample_layer_idx:
        skip_layers = []
        if self.arch['upsample'][layer_id//num_conv_in_block]:
          skip_layers.append(('upsample_%d'%layer_id, UpSample()))
        # if in_channels != out_channels:
          conv_1x1 = self.which_conv_1x1(in_channels, out_channels,
                                         kernel_size=1, padding=0)
          skip_layers.append(('upsample_%d_conv_1x1'%layer_id, conv_1x1))
        else:
          identity = Identity()
          skip_layers.append(('skip_%d_identity' % layer_id, identity))
        skip_layers = nn.Sequential(OrderedDict(skip_layers))
        self.skip_layers.append(skip_layers)

    self.layers_para_matrix = np.array(self.layers_para_list).T
    # output layer
    self.output_type = getattr(config, 'output_type', 'snconv')
    self.output_sample_arc = False
    if self.output_type == 'snconv':
      self.output_layer = nn.Sequential(
        nn.BatchNorm2d(
          self.arch['out_channels'][-1],
          affine=True, track_running_stats=self.track_running_stats),
        nn.ReLU(),
        self.which_conv(self.arch['out_channels'][-1], 3))
    elif self.output_type == 'MixedLayer':
      self.output_sample_arc = True
      if getattr(self, 'share_conv_weights', False):
        self.output_conv = MixedLayerSharedWeights(
          layer_id + 1, self.arch['out_channels'][-1], 3, ops=self.ops,
          track_running_stats=self.track_running_stats, scalesample=None,
          bn_type=bn_type)
      else:
        self.output_conv = MixedLayer(
          layer_id + 1, self.arch['out_channels'][-1], 3, ops=self.ops,
          track_running_stats=self.track_running_stats, scalesample=None,
          bn_type=bn_type)
    else:
      assert 0

    self.init_weights()

    if no_optim:
      return
    optim_c = config.optimizer
    self.lr, self.B1, self.B2 = optim_c.G_lr, optim_c.G_B1, optim_c.G_B2
    adam_eps = getattr(optim_c, 'adam_eps', 1.e-8)
    self.optim = optimizer_dict[optim_c.type](
      params=self.parameters(), lr=self.lr, betas=(self.B1, self.B2),
      eps=adam_eps)
    pass

  def forward(self, x, sample_arcs):
    """

    :param x:
    :param sample_arcs: (b, num_layers)
    :return:
    """

    x = self.linear(x)
    x = x.view(x.size(0), -1, self.bottom_width, self.bottom_width)

    upsample_layer = 0
    for layer_id in range(self.num_layers):
      if layer_id == 0:
        prev_layer = x
      sample_arc = sample_arcs[:, layer_id]
      x = self.layers[layer_id](x, sample_arc)

      if layer_id - 1 in self.upsample_layer_idx:
        x_up = self.skip_layers[upsample_layer](prev_layer)
        upsample_layer += 1
        x = x + x_up
        prev_layer = x

    if self.output_type == 'snconv':
      x = self.output_layer(x)
    elif self.output_type == 'MixedLayer':
      sample_arc = sample_arcs[:, -1]
      x = self.output_conv(x, sample_arc)
    else:
      assert 0
    x = torch.tanh(x)
    return x

  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv2d)
              or isinstance(module, nn.Linear)
              or isinstance(module, nn.Embedding)):
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        else:
          print('Init style not recognized...')
      if (isinstance(module, MixedLayerSharedWeights)):
        if self.init == 'ortho':
          for k, w in module.conv_weights.items():
            init.orthogonal_(w)
        else:
          assert 0
      if (isinstance(module, SinglePathLayer)):
        if self.init == 'ortho':
          init.orthogonal_(module.conv_weights_space)
        else:
          assert 0
      pass



class SharedDiscriminatorNoSkip(nn.Module):
  def __init__(self, resolution, no_optim=False,
               config=None):
    super(SharedDiscriminatorNoSkip, self).__init__()
    self.config = config
    self.init = self.config.init
    self.ops = self.config.ops
    self.track_running_stats = self.config.track_running_stats
    self.bottom_width = self.config.bottom_width

    self.arch = D_arch(config.ch, config.attention)[resolution]

    self.which_linear = functools.partial(
      layers.SNLinear, num_svs=1, num_itrs=1, eps=1e-6)
    self.which_conv = functools.partial(
      layers.SNConv2d, kernel_size=3, padding=1,
      num_svs=1, num_itrs=1, eps=1e-6)
    self.which_conv_1x1 = functools.partial(
      layers.SNConv2d, kernel_size=1, padding=0,
      num_svs=1, num_itrs=1, eps=1e-6)

    num_conv_in_block = 2
    self.num_layers = len(self.arch['in_channels']) * num_conv_in_block
    self.skip_layer_idx = \
      [num_conv_in_block * l
        for l in range(0, self.num_layers//num_conv_in_block)]

    self.layers = nn.ModuleList([])
    self.skip_layers = nn.ModuleList([])

    for layer_id in range(self.num_layers):
      block_in = self.arch['in_channels'][layer_id//num_conv_in_block]
      block_out = self.arch['out_channels'][layer_id//num_conv_in_block]
      if layer_id % num_conv_in_block == 0:
        in_channels = block_in
        out_channels = block_out
      else:
        in_channels = block_out
        out_channels = block_out
      downsample = (DownSample()
                  if (self.arch['downsample'][layer_id//num_conv_in_block] and
                      layer_id in self.skip_layer_idx)
                  else None)
      layer = MixedLayer(layer_id, in_channels, out_channels, ops=self.ops,
                         track_running_stats=self.track_running_stats,
                         scalesample=downsample)
      self.layers.append(layer)
      if layer_id in self.skip_layer_idx:
        skip_layers = []
        if self.arch['downsample'][layer_id//num_conv_in_block]:
          skip_layers.append(('downsample_%d'%layer_id, DownSample()))
        if in_channels != out_channels:
          conv_1x1 = self.which_conv_1x1(in_channels, out_channels,
                                         kernel_size=1, padding=0)
          skip_layers.append(('skip_%d_conv_1x1'%layer_id, conv_1x1))
        else:
          identity = Identity()
          skip_layers.append(('skip_%d_identity'%layer_id, identity))
        skip_layers = nn.Sequential(OrderedDict(skip_layers))
        self.skip_layers.append(skip_layers)

    self.linear = self.which_linear(
      self.arch['out_channels'][-1], 1)

    self.init_weights()

    if no_optim:
      return
    optim_c = config.optimizer
    self.lr, self.B1, self.B2, self.adam_eps = \
      optim_c.G_lr, optim_c.G_B1, optim_c.G_B2, optim_c.adam_eps

    self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                            betas=(self.B1, self.B2), weight_decay=0,
                            eps=self.adam_eps)

    pass

  def forward(self, x, sample_arcs):
    """

    :param x:
    :param sample_arcs: (b, num_layers)
    :return:
    """

    downsample_layer = 0
    for layer_id in range(self.num_layers):
      if layer_id == 0:
        prev_layer = x
      sample_arc = sample_arcs[:, layer_id]
      x = self.layers[layer_id](x, sample_arc)

      if layer_id - 1 in self.skip_layer_idx:
        x_up = self.skip_layers[downsample_layer](prev_layer)
        downsample_layer += 1
        x = x + x_up
        prev_layer = x

    x = torch.sum(torch.relu(x), [2, 3])
    x = self.linear(x)

    return x

  def init_weights(self):
    for module in self.modules():
      if (isinstance(module, nn.Conv2d)
              or isinstance(module, nn.Linear)
              or isinstance(module, nn.Embedding)):
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        else:
          print('Init style not recognized...')


class SharedGeneratorCBN(nn.Module):
  def __init__(self, resolution, n_classes, no_optim=False,
               config=None):
    super(SharedGeneratorCBN, self).__init__()
    self.n_classes = n_classes
    self.config = config
    for k, v in config.items():
      assert not hasattr(self, k)
      setattr(self, k, v)

    self.arch = G_arch(config.ch, config.attention)[resolution]

    if config.hier:
      # Number of places z slots into
      self.num_slots = len(self.arch['in_channels']) + 1
      self.z_chunk_size = (self.dim_z // self.num_slots)
      # Recalculate latent dimensionality for even splitting into chunks
      self.dim_z = self.z_chunk_size * self.num_slots
    else:
      self.num_slots = 1
      self.z_chunk_size = 0

    if self.use_sn:
      self.which_linear = functools.partial(
        layers.SNLinear, num_svs=1, num_itrs=1, eps=1e-6)
      self.which_conv = functools.partial(
        layers.SNConv2d, kernel_size=3, padding=1,
        num_svs=1, num_itrs=1, eps=1e-6)
      self.which_conv_1x1 = functools.partial(
        layers.SNConv2d, kernel_size=1, padding=0,
        num_svs=1, num_itrs=1, eps=1e-6)
    else:
      self.which_linear = nn.Linear
      self.which_conv = functools.partial(
        nn.Conv2d, kernel_size=3, padding=1)
      self.which_conv_1x1 = functools.partial(
        nn.Conv2d, kernel_size=1, padding=0)

    self.which_embedding = nn.Embedding
    bn_linear = (
      functools.partial(self.which_linear, bias=False) if self.G_shared
      else self.which_embedding)
    self.which_bn = functools.partial(
      layers.ccbn,
      which_linear=bn_linear,
      cross_replica=False,
      mybn=False,
      input_size=(self.shared_dim + self.z_chunk_size if self.G_shared
                  else self.n_classes),
      norm_style=self.norm_style,
      eps=self.BN_eps)

    # Prepare model
    # If not using shared embeddings, self.shared is just a passthrough
    self.shared = (self.which_embedding(self.n_classes, self.shared_dim) \
                     if self.G_shared else layers.identity())

    # First linear layer
    self.linear = self.which_linear(
      self.dim_z // self.num_slots, self.arch['in_channels'][0] * (config.bottom_width ** 2))

    num_conv_in_block = 2
    self.num_conv_in_block = num_conv_in_block
    self.num_layers = len(self.arch['in_channels']) * num_conv_in_block
    self.upsample_layer_idx = \
      [num_conv_in_block * l
        for l in range(0, self.num_layers//num_conv_in_block)]

    self.layers = nn.ModuleList([])
    self.layers_para_list = []
    self.skip_layers = nn.ModuleList([])

    for layer_id in range(self.num_layers):
      block_in = self.arch['in_channels'][layer_id//num_conv_in_block]
      block_out = self.arch['out_channels'][layer_id//num_conv_in_block]
      if layer_id % num_conv_in_block == 0:
        in_channels = block_in
        out_channels = block_out
      else:
        in_channels = block_out
        out_channels = block_out
      upsample = (UpSample()
                  if (self.arch['upsample'][layer_id//num_conv_in_block] and
                      layer_id in self.upsample_layer_idx)
                  else None)
      if getattr(self, 'share_conv_weights', False):
        layer = MixedLayerCondSharedWeights(
          layer_id, in_channels, out_channels, ops=self.ops,
          track_running_stats=self.track_running_stats,
          scalesample=upsample, which_bn=self.which_bn)
      else:
        layer = MixedLayerCond(
          layer_id, in_channels, out_channels, ops=self.ops,
          track_running_stats=self.track_running_stats,
          scalesample=upsample, which_bn=self.which_bn)
      self.layers.append(layer)
      self.layers_para_list.append(layer.num_para_list)
      if layer_id in self.upsample_layer_idx:
        skip_layers = []
        if self.arch['upsample'][layer_id//num_conv_in_block]:
          skip_layers.append(('upsample_%d'%layer_id, UpSample()))
        # if in_channels != out_channels:
          conv_1x1 = self.which_conv_1x1(in_channels, out_channels,
                                         kernel_size=1, padding=0)
          skip_layers.append(('upsample_%d_conv_1x1'%layer_id, conv_1x1))
        else:
          identity = Identity()
          skip_layers.append(('skip_%d_identity' % layer_id, identity))
        skip_layers = nn.Sequential(OrderedDict(skip_layers))
        self.skip_layers.append(skip_layers)

    self.layers_para_matrix = np.array(self.layers_para_list).T
    self.output_type = getattr(config, 'output_type', 'snconv')
    self.output_sample_arc = False
    if self.output_type == 'snconv':
      self.output_layer = nn.Sequential(
        nn.BatchNorm2d(
          self.arch['out_channels'][-1],
          affine=True, track_running_stats=self.track_running_stats),
        nn.ReLU(),
        self.which_conv(self.arch['out_channels'][-1], 3))
    elif self.output_type == 'MixedLayer':
      self.output_sample_arc = True
      self.output_conv = MixedLayer(
        layer_id + 1, self.arch['out_channels'][-1], 3, ops=self.ops,
        track_running_stats=self.track_running_stats, scalesample=None)
    else:
      assert 0
    self.init_weights()

    if no_optim:
      return
    optim_c = config.optimizer
    self.lr, self.B1, self.B2, self.adam_eps = \
      optim_c.G_lr, optim_c.G_B1, optim_c.G_B2, optim_c.adam_eps

    self.optim = optimizer_dict[optim_c.type](
      params=self.parameters(), lr=self.lr, betas=(self.B1, self.B2),
      weight_decay=0, eps=self.adam_eps)

    pass

  def forward(self, z, y, sample_arcs):
    """

    :param sample_arcs: (b, num_layers)
    :return:
    """
    y = self.shared(y)
    if self.hier:
      zs = torch.split(z, self.z_chunk_size, 1)
      z = zs[0]
      ys = [torch.cat([y, item], 1) for item in zs[1:]]
    else:
      ys = [y] * len(self.arch['in_channels'])

    x = self.linear(z)
    x = x.view(x.size(0), -1, self.bottom_width, self.bottom_width)

    upsample_layer = 0
    for layer_id in range(self.num_layers):
      if layer_id == 0:
        prev_layer = x
      sample_arc = sample_arcs[:, layer_id]
      x = self.layers[layer_id](
        x=x, y=ys[layer_id // self.num_conv_in_block], sample_arc=sample_arc)

      if layer_id - 1 in self.upsample_layer_idx:
        x_up = self.skip_layers[upsample_layer](prev_layer)
        upsample_layer += 1
        x = x + x_up
        prev_layer = x

    if self.output_type == 'snconv':
      x = self.output_layer(x)
    elif self.output_type == 'MixedLayer':
      sample_arc = sample_arcs[:, -1]
      x = self.output_conv(x, sample_arc)
    else:
      assert 0
    x = torch.tanh(x)
    return x

  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv2d)
              or isinstance(module, nn.Linear)
              or isinstance(module, nn.Embedding)):
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        else:
          print('Init style not recognized...')

      if (isinstance(module, (MixedLayerCondSharedWeights,
                              MixedLayerSharedWeights))):
        if self.init == 'ortho':
          for k, w in module.conv_weights.items():
            init.orthogonal_(w)
        else:
          assert 0


class AlphaDiscriminator(nn.Module):
  def __init__(self, resolution, no_optim=False,
               config=None):
    super(AlphaDiscriminator, self).__init__()
    self.config = config
    self.init = self.config.init
    self.ops = self.config.ops
    self.track_running_stats = self.config.track_running_stats
    self.bottom_width = self.config.bottom_width

    self.arch = D_arch(config.ch, config.attention)[resolution]

    self.which_linear = functools.partial(
      layers.SNLinear, num_svs=1, num_itrs=1, eps=1e-6)
    self.which_conv = functools.partial(
      layers.SNConv2d, kernel_size=3, padding=1,
      num_svs=1, num_itrs=1, eps=1e-6)
    self.which_conv_1x1 = functools.partial(
      layers.SNConv2d, kernel_size=1, padding=0,
      num_svs=1, num_itrs=1, eps=1e-6)

    num_conv_in_block = 2
    self.num_layers = len(self.arch['in_channels']) * num_conv_in_block
    self.skip_layer_idx = \
      [num_conv_in_block * l
        for l in range(0, self.num_layers//num_conv_in_block)]

    self.layers = nn.ModuleList([])
    self.skip_layers = nn.ModuleList([])

    for layer_id in range(self.num_layers):
      block_in = self.arch['in_channels'][layer_id//num_conv_in_block]
      block_out = self.arch['out_channels'][layer_id//num_conv_in_block]
      if layer_id % num_conv_in_block == 0:
        in_channels = block_in
        out_channels = block_out
      else:
        in_channels = block_out
        out_channels = block_out
      downsample = (DownSample()
                  if (self.arch['downsample'][layer_id//num_conv_in_block] and
                      layer_id in self.skip_layer_idx)
                  else None)
      layer = MixedLayer(layer_id, in_channels, out_channels, ops=self.ops,
                         track_running_stats=self.track_running_stats,
                         scalesample=downsample)
      self.layers.append(layer)
      if layer_id in self.skip_layer_idx:
        skip_layers = []
        if self.arch['downsample'][layer_id//num_conv_in_block]:
          skip_layers.append(('downsample_%d'%layer_id, DownSample()))
        if in_channels != out_channels:
          conv_1x1 = self.which_conv_1x1(in_channels, out_channels,
                                         kernel_size=1, padding=0)
          skip_layers.append(('skip_%d_conv_1x1'%layer_id, conv_1x1))
        else:
          identity = Identity()
          skip_layers.append(('skip_%d_identity'%layer_id, identity))
        skip_layers = nn.Sequential(OrderedDict(skip_layers))
        self.skip_layers.append(skip_layers)

    self.linear = self.which_linear(
      self.arch['out_channels'][-1], 1)

    self.init_weights()

    if no_optim:
      return
    optim_c = config.optimizer
    self.lr, self.B1, self.B2, self.adam_eps = \
      optim_c.G_lr, optim_c.G_B1, optim_c.G_B2, optim_c.adam_eps

    self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                            betas=(self.B1, self.B2), weight_decay=0,
                            eps=self.adam_eps)

    pass

  def forward(self, x, sample_arcs):
    """

    :param x:
    :param sample_arcs: (b, num_layers)
    :return:
    """

    downsample_layer = 0
    for layer_id in range(self.num_layers):
      if layer_id == 0:
        prev_layer = x
      sample_arc = sample_arcs[:, layer_id]
      x = self.layers[layer_id](x, sample_arc)

      if layer_id - 1 in self.skip_layer_idx:
        x_up = self.skip_layers[downsample_layer](prev_layer)
        downsample_layer += 1
        x = x + x_up
        prev_layer = x

    x = torch.sum(torch.relu(x), [2, 3])
    x = self.linear(x)

    return x

  def init_weights(self):
    for module in self.modules():
      if (isinstance(module, nn.Conv2d)
              or isinstance(module, nn.Linear)
              or isinstance(module, nn.Embedding)):
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        else:
          print('Init style not recognized...')


def test_SharedGeneratorNoSkip(args, myargs):
  yaml_file = 'enas_cgan/configs/enas_cgan.yaml'
  with open(yaml_file, 'r') as f:
    configs = EasyDict(yaml.load(f))
  config = configs.search_cgan_gen_cifar10_v1.model
  resolution = 32
  gen_c = config.generator
  generator = SharedGeneratorNoSkip(
    resolution=resolution, config=gen_c).cuda()

  from enas_cgan.models.controller import Controller
  controller_c = configs.search_cgan_gen_cifar10_v1.model.controller
  n_classes = 10
  controller = Controller(n_classes=n_classes,
                          num_layers=generator.num_layers,
                          num_branches=len(generator.ops),
                          config=controller_c)
  controller = controller.cuda()
  class_id = 0
  with torch.no_grad():
    controller(class_id)  # perform forward pass to generate a new architecture
  # sample_arc = controller.sample_arc
  sample_arc = torch.randint(0, len(generator.ops), (128, generator.num_layers))
  dummy_data = torch.rand(128, gen_c.dim_z).cuda()
  x = generator(dummy_data, sample_arc)

  from torchsummary import summary
  summary(generator, [(120, ), (generator.num_layers, )])

  import torchviz
  g = torchviz.make_dot(x)
  g.view()

  pass


def test_SharedDiscriminatorNoSkip(args, myargs):
  yaml_file = 'enas_cgan/configs/enas_cgan.yaml'
  with open(yaml_file, 'r') as f:
    configs = EasyDict(yaml.load(f))
  config = configs.search_cgan_gen_dis_cifar10.model
  resolution = 32
  # gen_c = config.generator
  # generator = SharedGeneratorNoSkip(
  #   resolution=resolution, config=gen_c).cuda()
  dis_c = config.discriminator
  discriminator = SharedDiscriminatorNoSkip(
    resolution=resolution, config=dis_c).cuda()

  from enas_cgan.models.controller import Controller
  controller_c = configs.search_cgan_gen_dis_cifar10.model.controller
  n_classes = 10
  controller = Controller(n_classes=n_classes,
                          num_layers=discriminator.num_layers,
                          num_branches=len(discriminator.ops),
                          config=controller_c)
  controller = controller.cuda()
  class_id = 0
  with torch.no_grad():
    controller(class_id)  # perform forward pass to generate a new architecture
  # sample_arc = controller.sample_arc
  sample_arc = torch.randint(
    0, len(discriminator.ops), (128, discriminator.num_layers))
  dummy_data = torch.rand(128, 3, 32, 32).cuda()
  x = discriminator(dummy_data, sample_arc)

  # from torchsummary import summary
  # summary(discriminator, [[3, 32, 32], [discriminator.num_layers, ]])

  import torchviz
  g = torchviz.make_dot(x)
  g.view()

  pass

def test_AlphaDiscriminator(args, myargs):
  yaml_file = 'enas_cgan/configs/enas_cgan.yaml'
  with open(yaml_file, 'r') as f:
    configs = EasyDict(yaml.load(f))
  config = configs.search_cgan_gen_dis_cifar10.model
  resolution = 32
  # gen_c = config.generator
  # generator = SharedGeneratorNoSkip(
  #   resolution=resolution, config=gen_c).cuda()
  dis_c = config.discriminator
  discriminator = SharedDiscriminatorNoSkip(
    resolution=resolution, config=dis_c).cuda()

  from enas_cgan.models.controller import Controller
  controller_c = configs.search_cgan_gen_dis_cifar10.model.controller
  n_classes = 10
  controller = Controller(n_classes=n_classes,
                          num_layers=discriminator.num_layers,
                          num_branches=len(discriminator.ops),
                          config=controller_c)
  controller = controller.cuda()
  class_id = 0
  with torch.no_grad():
    controller(class_id)  # perform forward pass to generate a new architecture
  # sample_arc = controller.sample_arc
  sample_arc = torch.randint(
    0, len(discriminator.ops), (128, discriminator.num_layers))
  dummy_data = torch.rand(128, 3, 32, 32).cuda()
  x = discriminator(dummy_data, sample_arc)

  # from torchsummary import summary
  # summary(discriminator, [[3, 32, 32], [discriminator.num_layers, ]])

  import torchviz
  g = torchviz.make_dot(x)
  g.view()

  pass


def test_SharedGeneratorCBN(args, myargs):
  yaml_file = 'enas_cgan/configs/enas_cgan.yaml'
  with open(yaml_file, 'r') as f:
    configs = EasyDict(yaml.load(f))
  config = configs.retrain_cgan_gen_cifar10_v6_cbn.model
  resolution = 32
  n_classes = 10
  gen_c = config.generator
  generator = SharedGeneratorCBN(
    resolution=resolution, n_classes = n_classes, config=gen_c).cuda()

  bs = len(generator.ops)
  sample_arc = torch.arange(bs).view(-1, 1).repeat((1, generator.num_layers))
  dummy_data = torch.rand(bs, gen_c.dim_z).cuda()
  y = torch.arange(bs).cuda()
  x = generator(z=dummy_data, y=y, sample_arcs=sample_arc)

  import torchviz
  g = torchviz.make_dot(x)
  g.view()

  pass


def test_SharedGeneratorNoSkip_v4(args, myargs):
  yaml_file = 'enas_cgan/configs/enas_cgan.yaml'
  with open(yaml_file, 'r') as f:
    configs = EasyDict(yaml.load(f))
  config = configs.retrain_cgan_gen_cifar10_v4.model
  resolution = 32
  gen_c = config.generator
  generator = SharedGeneratorNoSkip(
    resolution=resolution, config=gen_c).cuda()

  from enas_cgan.models.controller import Controller
  controller_c = configs.retrain_cgan_gen_cifar10_v4.model.controller
  n_classes = 10
  controller = Controller(n_classes=n_classes,
                          num_layers=generator.num_layers,
                          num_branches=len(generator.ops),
                          config=controller_c)
  controller = controller.cuda()
  class_id = 0
  with torch.no_grad():
    controller(class_id)  # perform forward pass to generate a new architecture
  # sample_arc = controller.sample_arc
  sample_arc = torch.randint(0, len(generator.ops), (128, generator.num_layers))
  dummy_data = torch.rand(128, gen_c.dim_z).cuda()
  x = generator(dummy_data, sample_arc)

  import torchviz
  g = torchviz.make_dot(x)
  g.view()

  pass


def test_SharedGeneratorCBN_v5(args, myargs):
  yaml_file = 'enas_cgan/configs/enas_cgan.yaml'
  with open(yaml_file, 'r') as f:
    configs = EasyDict(yaml.load(f))
  config = configs.search_cgan_gen_cifar10_v5.model
  resolution = 32
  n_classes = 10
  gen_c = config.generator
  generator = SharedGeneratorCBN(
    resolution=resolution, n_classes = n_classes, config=gen_c).cuda()

  from enas_cgan.models.controller import Controller
  controller_c = configs.search_cgan_gen_cifar10_v5.model.controller
  controller = Controller(n_classes=n_classes,
                          num_layers=generator.num_layers,
                          num_branches=len(generator.ops),
                          config=controller_c)
  controller = controller.cuda()
  class_id = 0
  with torch.no_grad():
    controller(class_id)  # perform forward pass to generate a new architecture
  # sample_arc = controller.sample_arc
  sample_arc = torch.randint(0, len(generator.ops), (128, generator.num_layers))
  dummy_data = torch.rand(128, gen_c.dim_z).cuda()
  y = torch.randint(0, n_classes, (128,)).cuda()
  x = generator(z=dummy_data, y=y, sample_arcs=sample_arc)

  import torchviz
  g = torchviz.make_dot(x)
  g.view()

  pass

def test_SharedGeneratorNoSkip_v6(args, myargs):
  yaml_file = 'enas_cgan/configs/enas_cgan.yaml'
  with open(yaml_file, 'r') as f:
    configs = EasyDict(yaml.load(f))
  config = configs.retrain_cgan_gen_cifar10_v6.model
  resolution = 32
  gen_c = config.generator
  generator = SharedGeneratorNoSkip(
    resolution=resolution, config=gen_c).cuda()

  bs = len(generator.ops)
  sample_arc = torch.arange(bs).view(-1, 1)
  sample_arc = sample_arc.repeat((1, generator.num_layers))
  # sample_arc = torch.randint(0, len(generator.ops), (128, generator.num_layers))
  dummy_data = torch.rand(bs, gen_c.dim_z).cuda()
  x = generator(dummy_data, sample_arc)

  import torchviz
  g = torchviz.make_dot(x)
  g.view()

  pass


def test_retrain_cgan_gen_cifar10_v6_disc(args, myargs):
  yaml_file = 'enas_cgan/configs/enas_cgan.yaml'
  with open(yaml_file, 'r') as f:
    configs = EasyDict(yaml.load(f))
  config = configs.retrain_cgan_gen_cifar10_v6.model
  resolution = 32
  bs = 8

  from ..models.autogan_cifar10_a import Discriminator
  disc_c = config.discriminator
  D = Discriminator(args=disc_c).cuda()

  x = torch.rand(bs, 3, resolution, resolution).cuda()
  x = D(x)

  import torchviz
  g = torchviz.make_dot(x)
  g.view()

  pass


def test_retrain_cgan_gen_stl10_v6_gen(args, myargs):
  yaml_file = 'enas_cgan/configs/stl10_cvpr_final.yaml'
  with open(yaml_file, 'r') as f:
    configs = EasyDict(yaml.load(f))
  config = configs.retrain_cgan_gen_stl10_v6_cbn.model
  resolution = 48
  n_classes = 1000
  gen_c = config.generator
  generator = SharedGeneratorCBN(
    resolution=resolution, n_classes=n_classes,
    no_optim=False, config=config.generator).cuda()

  bs = len(generator.ops)
  y = torch.arange(bs).cuda()
  sample_arc = torch.arange(bs).view(-1, 1)
  sample_arc = sample_arc.repeat((1, generator.num_layers))
  # sample_arc = torch.randint(0, len(generator.ops), (128, generator.num_layers))
  dummy_data = torch.rand(bs, gen_c.dim_z).cuda()
  x = generator(dummy_data, y, sample_arc)

  import torchviz
  g = torchviz.make_dot(x)
  g.view()

  pass