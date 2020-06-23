from easydict import EasyDict
import yaml
import functools
import unittest

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from . import layers
from .BigGAN import G_arch
from .ops import ENASLayer, UpSample


'''
Implementation Notes:
-Setting track_running_stats to True in BatchNorm layers seems to hurt validation
    and test performance for some reason, so here it is disabled even though it
    is used in the official implementation.
'''


class FactorizedReduction(nn.Module):
  '''
  Reduce both spatial dimensions (width and height) by a factor of 2, and
  potentially to change the number of output filters

  https://github.com/melodyguan/enas/blob/master/src/cifar10/general_child.py#L129
  '''

  def __init__(self, in_planes, out_planes, stride=2):
    super(FactorizedReduction, self).__init__()

    assert out_planes % 2 == 0, (
      "Need even number of filters when using this factorized reduction.")

    self.in_planes = in_planes
    self.out_planes = out_planes
    self.stride = stride

    if stride == 1:
      self.fr = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_planes, track_running_stats=False))
    else:
      self.path1 = nn.Sequential(
        nn.AvgPool2d(1, stride=stride),
        nn.Conv2d(in_planes, out_planes // 2, kernel_size=1, bias=False))

      self.path2 = nn.Sequential(
        nn.AvgPool2d(1, stride=stride),
        nn.Conv2d(in_planes, out_planes // 2, kernel_size=1, bias=False))
      self.bn = nn.BatchNorm2d(out_planes, track_running_stats=False)

  def forward(self, x):
    if self.stride == 1:
      return self.fr(x)
    else:
      path1 = self.path1(x)

      # pad the right and the bottom, then crop to include those pixels
      path2 = F.pad(x, pad=(0, 1, 0, 1), mode='constant', value=0.)
      path2 = path2[:, :, 1:, 1:]
      path2 = self.path2(path2)

      out = torch.cat([path1, path2], dim=1)
      out = self.bn(out)
      return out


class FixedLayer(nn.Module):
  '''
  https://github.com/melodyguan/enas/blob/master/src/cifar10/general_child.py#L245
  '''

  def __init__(self, layer_id, in_planes, out_planes, sample_arc):
    super(FixedLayer, self).__init__()

    self.layer_id = layer_id
    self.in_planes = in_planes
    self.out_planes = out_planes
    self.sample_arc = sample_arc

    self.layer_type = sample_arc[0]
    if self.layer_id > 0:
      self.skip_indices = sample_arc[1]
    else:
      self.skip_indices = torch.zeros(1)

    if self.layer_type == 0:
      self.branch = ConvBranch(in_planes, out_planes, kernel_size=3)
    elif self.layer_type == 1:
      self.branch = ConvBranch(in_planes, out_planes, kernel_size=3,
                               separable=True)
    elif self.layer_type == 2:
      self.branch = ConvBranch(in_planes, out_planes, kernel_size=5)
    elif self.layer_type == 3:
      self.branch = ConvBranch(in_planes, out_planes, kernel_size=5,
                               separable=True)
    elif self.layer_type == 4:
      self.branch = PoolBranch(in_planes, out_planes, 'avg')
    elif self.layer_type == 5:
      self.branch = PoolBranch(in_planes, out_planes, 'max')
    else:
      raise ValueError("Unknown layer_type {}".format(self.layer_type))

    # Use concatentation instead of addition in the fixed layer for some reason
    in_planes = int((torch.sum(self.skip_indices).item() + 1) * in_planes)
    self.dim_reduc = nn.Sequential(
      nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
      nn.ReLU(),
      nn.BatchNorm2d(out_planes, track_running_stats=False))

  def forward(self, x, prev_layers, sample_arc):
    out = self.branch(x)

    res_layers = []
    for i, skip in enumerate(self.skip_indices):
      if skip == 1:
        res_layers.append(prev_layers[i])
    prev = res_layers + [out]
    prev = torch.cat(prev, dim=1)

    out = self.dim_reduc(prev)
    return out


class SeparableConv(nn.Module):
  def __init__(self, in_planes, out_planes, kernel_size, bias):
    super(SeparableConv, self).__init__()
    padding = (kernel_size - 1) // 2
    self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size,
                               padding=padding, groups=in_planes, bias=bias)
    self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=bias)

  def forward(self, x):
    out = self.depthwise(x)
    out = self.pointwise(out)
    return out


class ConvBranch(nn.Module):
  '''
  https://github.com/melodyguan/enas/blob/master/src/cifar10/general_child.py#L483
  '''

  def __init__(self, in_planes, out_planes, kernel_size, separable=False,
               which_conv=None, which_bn=None):
    super(ConvBranch, self).__init__()
    assert kernel_size in [3, 5], "Kernel size must be either 3 or 5"

    self.in_planes = in_planes
    self.out_planes = out_planes
    self.kernel_size = kernel_size
    self.separable = separable
    if not which_conv:
      which_conv = nn.Conv2d
    if not which_bn:
      which_bn = nn.BatchNorm2d

    self.inp_conv1 = nn.Sequential(
      which_conv(in_planes, out_planes, kernel_size=1, bias=False),
      which_bn(out_planes, track_running_stats=False),
      nn.ReLU())

    if separable:
      self.out_conv = nn.Sequential(
        SeparableConv(in_planes, out_planes, kernel_size=kernel_size,
                      bias=False),
        nn.BatchNorm2d(out_planes, track_running_stats=False),
        nn.ReLU())
    else:
      padding = (kernel_size - 1) // 2
      self.out_conv = nn.Sequential(
        which_conv(in_planes, out_planes, kernel_size=kernel_size,
                  padding=padding, bias=False),
        which_bn(out_planes, track_running_stats=False),
        nn.ReLU())

  def forward(self, x):
    out = self.inp_conv1(x)
    out = self.out_conv(out)
    return out


class PoolBranch(nn.Module):
  '''
  https://github.com/melodyguan/enas/blob/master/src/cifar10/general_child.py#L546
  '''

  def __init__(self, in_planes, out_planes, avg_or_max):
    super(PoolBranch, self).__init__()

    self.in_planes = in_planes
    self.out_planes = out_planes
    self.avg_or_max = avg_or_max

    self.conv1 = nn.Sequential(
      nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
      nn.BatchNorm2d(out_planes, track_running_stats=False),
      nn.ReLU())

    if avg_or_max == 'avg':
      self.pool = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
    elif avg_or_max == 'max':
      self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    else:
      raise ValueError("Unknown pool {}".format(avg_or_max))

  def forward(self, x):
    out = self.conv1(x)
    out = self.pool(out)
    return out


class SharedCNN(nn.Module):
  def __init__(self,
               num_layers=12,
               num_branches=6,
               out_filters=24,
               keep_prob=1.0,
               fixed_arc=None
               ):
    super(SharedCNN, self).__init__()

    self.num_layers = num_layers
    self.num_branches = num_branches
    self.out_filters = out_filters
    self.keep_prob = keep_prob
    self.fixed_arc = fixed_arc

    pool_distance = self.num_layers // 3
    self.pool_layers = [pool_distance - 1, 2 * pool_distance - 1]

    self.stem_conv = nn.Sequential(
      nn.Conv2d(3, out_filters, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_filters, track_running_stats=False))

    self.layers = nn.ModuleList([])
    self.pooled_layers = nn.ModuleList([])

    for layer_id in range(self.num_layers):
      if self.fixed_arc is None:
        layer = ENASLayer(layer_id, self.out_filters, self.out_filters)
      else:
        layer = FixedLayer(layer_id, self.out_filters, self.out_filters,
                           self.fixed_arc[str(layer_id)])
      self.layers.append(layer)

      if layer_id in self.pool_layers:
        for i in range(len(self.layers)):
          if self.fixed_arc is None:
            self.pooled_layers.append(
              FactorizedReduction(self.out_filters, self.out_filters))
          else:
            self.pooled_layers.append(
              FactorizedReduction(self.out_filters, self.out_filters * 2))
        if self.fixed_arc is not None:
          self.out_filters *= 2

    self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    self.dropout = nn.Dropout(p=1. - self.keep_prob)
    self.classify = nn.Linear(self.out_filters, 10)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

  def forward(self, x, sample_arc):

    x = self.stem_conv(x)

    prev_layers = []
    pool_count = 0
    for layer_id in range(self.num_layers):
      x = self.layers[layer_id](x, prev_layers, sample_arc[str(layer_id)])
      prev_layers.append(x)
      if layer_id in self.pool_layers:
        for i, prev_layer in enumerate(prev_layers):
          # Go through the outputs of all previous layers and downsample them
          prev_layers[i] = self.pooled_layers[pool_count](prev_layer)
          pool_count += 1
        x = prev_layers[-1]

    x = self.global_avg_pool(x)
    x = x.view(x.shape[0], -1)
    x = self.dropout(x)
    out = self.classify(x)

    return out


class SharedGenerator(nn.Module):
  def __init__(self, resolution, no_optim=False,
               config=None):
    super(SharedGenerator, self).__init__()
    self.config = config
    self.init = self.config.init
    self.ops = self.config.ops
    self.track_running_stats = self.config.track_running_stats
    self.bottom_width = self.config.bottom_width

    self.arch = G_arch(config.ch, config.attention)[resolution]

    self.which_linear = functools.partial(
      layers.SNLinear, num_svs=1, num_itrs=1, eps=1e-6)
    self.which_conv = functools.partial(
      layers.SNConv2d, kernel_size=3, padding=1,
      num_svs=1, num_itrs=1, eps=1e-6)
    self.which_conv_1x1 = functools.partial(
      layers.SNConv2d, kernel_size=1, padding=0,
      num_svs=1, num_itrs=1, eps=1e-6)

    # First linear layer
    self.linear = self.which_linear(
      config.dim_z, self.arch['in_channels'][0] * (config.bottom_width ** 2))

    num_conv_in_block = 2
    self.num_layers = len(self.arch['in_channels']) * num_conv_in_block
    self.upsample_layer_idx = \
      [num_conv_in_block * l
        for l in range(0, self.num_layers//num_conv_in_block)]

    self.layers = nn.ModuleList([])
    self.upsample_layers = nn.ModuleList([])

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
                      layer_id % num_conv_in_block == 0)
                  else None)
      layer = ENASLayer(layer_id, in_channels, out_channels, ops=self.ops,
                        track_running_stats=self.track_running_stats,
                        upsample=upsample)
      # layer = ENASLayer(layer_id, in_channels, out_channels,
      #                   track_running_stats=True)
      # layer = FixedLayer(layer_id, self.out_filters, self.out_filters,
      #                    self.fixed_arc[str(layer_id)])
      self.layers.append(layer)
      if layer_id in self.upsample_layer_idx and layer_id != 0:
        for i in range(len(self.layers) - 1):
          upsample_conv = []
          upsample_conv.append(UpSample())
          if in_channels != out_channels:
            conv_1x1 = self.which_conv_1x1(in_channels, out_channels,
                                           kernel_size=1, padding=0)
            upsample_conv.append(conv_1x1)
          upsample_conv = nn.Sequential(*upsample_conv)
          self.upsample_layers.append(upsample_conv)

    self.output_layer = nn.Sequential(
      nn.BatchNorm2d(
        self.arch['out_channels'][-1],
        affine=True, track_running_stats=self.track_running_stats),
      nn.ReLU(),
      self.which_conv(self.arch['out_channels'][-1], 3))

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

  def forward(self, x, sample_arc):

    x = self.linear(x)
    x = x.view(x.size(0), -1, self.bottom_width, self.bottom_width)

    prev_layers = []
    upsample_layer = 0
    for layer_id in range(self.num_layers):
      if layer_id in self.upsample_layer_idx and layer_id != 0:
        for i, prev_layer in enumerate(prev_layers):
          # Go through the outputs of all previous layers and upsample them
          prev_layers[i] = self.upsample_layers[upsample_layer](prev_layer)
          upsample_layer += 1
      x = self.layers[layer_id](x, prev_layers, sample_arc[str(layer_id)])
      prev_layers.append(x)

    x = torch.tanh(self.output_layer(x))
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


class TestingCase(unittest.TestCase):

  def testSharedGenerator(self):
    yaml_file = 'configs/enas_cgan.yaml'
    with open(yaml_file, 'r') as f:
      configs = EasyDict(yaml.load(f))
    config = configs.train_cgan_cnn_macro_cifar10.model
    resolution = 'test'
    resolution = 64
    generator = SharedGenerator(
      resolution=resolution, config=config.generator).cuda()

    from enas_cgan.models.controller import Controller
    args = configs.train_cgan_cnn_macro_cifar10.model.controller
    n_classes = 10
    controller = Controller(n_classes=n_classes,
                            num_layers=generator.num_layers,
                            num_branches=len(generator.ops),
                            config=config)
    controller = controller.cuda()
    class_id = 0
    with torch.no_grad():
      controller(class_id)  # perform forward pass to generate a new architecture
    sample_arc = controller.sample_arc
    for item in sample_arc.values():
      if len(item) > 1:
        item[1][:] = 1
    dummy_data = torch.rand(128, 120).cuda()
    x = generator(dummy_data, sample_arc)
    pass
