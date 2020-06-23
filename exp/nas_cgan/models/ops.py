import functools
import unittest
import yaml
from easydict import EasyDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import layers
# import genotypes as gt


OPS = {
  'none': lambda C, stride, affine: Zero(stride),
  # 'avg_pool_3x3': lambda C, stride, affine: PoolBN('avg', C, 3, stride, 1,
  #                                                  affine=affine),
  # 'max_pool_3x3': lambda C, stride, affine: PoolBN('max', C, 3, stride, 1,
  #                                                  affine=affine),
  # 'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1,
  #                                                   affine=affine),
  # 'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2,
  #                                                   affine=affine),
  # 'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3,
  #                                                   affine=affine),
  # 'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2,
  #                                                   affine=affine),  # 5x5
  # 'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2,
  #                                                   affine=affine),  # 9x9
  # 'conv_7x1_1x7': lambda C, stride, affine: FacConv(C, C, 7, stride, 3,
  #                                                   affine=affine),
  'snconv_3x3': lambda **kwargs: \
    SNConv(kernel_size=3, stride=1, padding=1, **kwargs),
  'snconv_5x5': lambda **kwargs: \
    SNConv(kernel_size=5, stride=1, padding=2, **kwargs),
  'conv_3x3': lambda **kwargs: \
    Conv(kernel_size=3, stride=1, padding=1, **kwargs),
  'conv_5x5': lambda **kwargs: \
    Conv(kernel_size=5, stride=1, padding=2, **kwargs),
  'snconv_dil_3x3': lambda **kwargs: \
    SNConv(kernel_size=3, stride=1, padding=2, dilation=2, **kwargs),# 5x5
  'conv_dil_3x3': lambda **kwargs: \
    Conv(kernel_size=3, stride=1, padding=2, dilation=2, **kwargs),# 5x5
  'snconv_dil_5x5': lambda **kwargs: \
    SNConv(kernel_size=5, stride=1, padding=4, dilation=2, **kwargs),# 9x9
  'conv_dil_5x5': lambda **kwargs: \
    Conv(kernel_size=5, stride=1, padding=4, dilation=2, **kwargs),# 9x9
  'snconv_dil2_3x3': lambda **kwargs: \
    SNConv(kernel_size=3, stride=1, padding=2, dilation=2, **kwargs),# 5x5
  'conv_dil2_3x3': lambda **kwargs: \
    Conv(kernel_size=3, stride=1, padding=2, dilation=2, **kwargs),# 5x5
  'snconv_dil3_3x3': lambda **kwargs: \
    SNConv(kernel_size=3, stride=1, padding=3, dilation=3, **kwargs),# 7x7
  'conv_dil3_3x3': lambda **kwargs: \
    Conv(kernel_size=3, stride=1, padding=3, dilation=3, **kwargs),# 7x7
  'snconv_dil4_3x3': lambda **kwargs: \
    SNConv(kernel_size=3, stride=1, padding=4, dilation=4, **kwargs),# 9x9
  'conv_dil4_3x3': lambda **kwargs: \
    Conv(kernel_size=3, stride=1, padding=4, dilation=4, **kwargs),# 9x9
  'skip_connect': lambda C_in, C_out, **kwargs: \
    Identity(**kwargs) if C_in == C_out else \
      Conv(kernel_size=1, stride=1, padding=0, **kwargs),

  'max_pool_sn': lambda C_in, C_out, stride, affine, track_running_stats: \
    PoolBranch(C_in, C_out, 'max', use_sn=True,
               track_running_stats=track_running_stats),
  'max_pool': lambda C_in, C_out, stride, affine, track_running_stats: \
    PoolBranch(C_in, C_out, 'max', use_sn=False,
               track_running_stats=track_running_stats),
  'avg_pool_sn': lambda C_in, C_out, stride, affine, track_running_stats: \
    PoolBranch(C_in, C_out, 'avg', use_sn=True,
               track_running_stats=track_running_stats),
  'avg_pool': lambda C_in, C_out, stride, affine, track_running_stats: \
    PoolBranch(C_in, C_out, 'avg', use_sn=False,
               track_running_stats=track_running_stats),
  'skip_connect_sn': lambda C_in, C_out, stride, affine, track_running_stats: \
    Identity() if C_in == C_out else \
      SNConv(C_in, C_out, 1, stride, 0, affine=affine,
             track_running_stats=track_running_stats),
  # 'skip_connect': lambda C_in, C_out, stride, affine, track_running_stats: \
  #   Identity() if C_in == C_out else \
  #     Conv(C_in, C_out, 1, stride, 0, affine=affine,
  #            track_running_stats=track_running_stats),

}


class UpSample(nn.Module):
  def __init__(self, scale_factor=2):
    super(UpSample, self).__init__()
    self.upsample = functools.partial(F.interpolate, scale_factor=scale_factor)

  def forward(self, x):
    x = self.upsample(x)
    return x


class DownSample(nn.Module):
  def __init__(self, scale_factor=2):
    super(DownSample, self).__init__()
    self.downsample = nn.AvgPool2d(scale_factor)

  def forward(self, x):
    x = self.downsample(x)
    return x


class FactorizedReduce(nn.Module):
  """
  Reduce feature map size by factorized pointwise(stride=2).
  """

  def __init__(self, C_in, C_out, affine=True, track_running_stats=False):
    super().__init__()
    self.relu = nn.ReLU()
    self.bn = nn.BatchNorm2d(
      C_out, affine=affine, track_running_stats=track_running_stats)
    self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)

  def forward(self, x):
    x = self.relu(x)
    x = self.bn(x)
    x = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
    return x


class PoolBranch(nn.Module):
  '''
  https://github.com/melodyguan/enas/blob/master/src/cifar10/general_child.py#L546
  '''

  def __init__(self, in_planes, out_planes, avg_or_max, use_sn=False,
               track_running_stats=False):
    super(PoolBranch, self).__init__()

    self.in_planes = in_planes
    self.out_planes = out_planes
    self.avg_or_max = avg_or_max

    if use_sn:
      bn_relu_conv = SNConv
    else:
      bn_relu_conv = Conv

    self.conv1 = bn_relu_conv(in_planes, out_planes, 1, 1, 0,
                              track_running_stats=track_running_stats)
    # self.conv1 = nn.Sequential(
    #   nn.BatchNorm2d(out_planes, track_running_stats=False),
    #   nn.ReLU(),
    #   nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
    # )

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


class Identity(nn.Module):
  def __init__(self, *args, **kwargs):
    super().__init__()

  def forward(self, x):
    return x


class SNConv(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride,
               padding, dilation=1, affine=True,
               track_running_stats=False,
               which_bn=None,
               weight=None, bias=None, bn_type='none',
               which_act=nn.ReLU):
    super(SNConv, self).__init__()
    self.kernel_size = kernel_size
    self.y = False
    if which_bn is not None:
      BN = which_bn
      self.y = True
    else:
      if bn_type == 'bn':
        BN = functools.partial(
          nn.BatchNorm2d,
          affine=affine, track_running_stats=track_running_stats)
      elif bn_type == 'in':
        BN = functools.partial(
          nn.InstanceNorm2d,
          affine=affine, track_running_stats=False)
      elif bn_type == 'none':
        BN = Identity
      else:
        assert 0

    self.bn = BN(C_in)
    if which_act is not None:
      self.act = which_act()
    else:
      self.act = Identity()
    if weight is not None:
      self.conv = layers.SNConv2dFunc(
        weight=weight, bias=bias, kernel_size=kernel_size,
        num_svs=1, num_itrs=1, eps=1e-6,
        stride=stride, padding=padding, dilation=dilation)
    else:
      snconv = functools.partial(
        layers.SNConv2d, num_svs=1, num_itrs=1, eps=1e-6)
      self.conv = snconv(
        C_in, C_out, kernel_size, stride, padding, dilation=dilation)

  def forward(self, *inputs):
    if self.y:
      x, y = inputs
      x = self.bn(x, y)
    else:
      x = inputs[0]
      x = self.bn(x)
    x = self.act(x)
    x = self.conv(x)
    return x


class Conv(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride,
               padding, dilation=1, affine=True,
               track_running_stats=False, which_bn=None,
               weight=None, bias=None, bn_type='none',
               which_act=nn.ReLU):
    super(Conv, self).__init__()
    self.kernel_size = kernel_size
    self.y = False
    if which_bn is not None:
      BN = which_bn
      self.y = True
    else:
      if bn_type == 'bn':
        BN = functools.partial(
          nn.BatchNorm2d,
          affine=affine, track_running_stats=track_running_stats)
      elif bn_type == 'in':
        BN = functools.partial(
          nn.InstanceNorm2d,
          affine=affine, track_running_stats=False)
      elif bn_type == 'none':
        BN = Identity
      else:
        assert 0

    self.bn = BN(C_in)
    if which_act is not None:
      self.act = which_act()
    else:
      self.act = Identity()
    if weight is not None:
      self.weight = weight
      self.bias = bias
      self.conv = functools.partial(
        F.conv2d, stride=stride, padding=padding, dilation=dilation, groups=1)
    else:
      self.conv = nn.Conv2d(
        C_in, C_out, kernel_size, stride,padding, dilation=dilation)

  def forward(self, *inputs):
    if self.y:
      x, y = inputs
      x = self.bn(x, y)
    else:
      x = inputs[0]
      x = self.bn(x)
    x = self.act(x)
    if hasattr(self, 'weight'):
      max_ks = self.weight.size(-1)
      start = (max_ks - self.kernel_size) // 2
      end = -start + max_ks
      weight = self.weight[:, :, start:end, start:end]
      x = self.conv(x, weight=weight, bias=self.bias)
    else:
      x = self.conv(x)
    return x


def drop_path_(x, drop_prob, training):
  if training and drop_prob > 0.:
    keep_prob = 1. - drop_prob
    # per data point mask; assuming x in cuda.
    mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
    x.div_(keep_prob).mul_(mask)

  return x


class DropPath_(nn.Module):
  def __init__(self, p=0.):
    """ [!] DropPath is inplace module
    Args:
        p: probability of an path to be zeroed.
    """
    super().__init__()
    self.p = p

  def extra_repr(self):
    return 'p={}, inplace'.format(self.p)

  def forward(self, x):
    drop_path_(x, self.p, self.training)

    return x


class PoolBN(nn.Module):
  """
  AvgPool or MaxPool - BN
  """

  def __init__(self, pool_type, C, kernel_size, stride, padding, affine=True):
    """
    Args:
        pool_type: 'max' or 'avg'
    """
    super().__init__()
    if pool_type.lower() == 'max':
      self.pool = nn.MaxPool2d(kernel_size, stride, padding)
    elif pool_type.lower() == 'avg':
      self.pool = nn.AvgPool2d(kernel_size, stride, padding,
                               count_include_pad=False)
    else:
      raise ValueError()

    self.bn = nn.BatchNorm2d(C, affine=affine)

  def forward(self, x):
    out = self.pool(x)
    out = self.bn(out)
    return out


class StdConv(nn.Module):
  """ Standard conv
  ReLU - Conv - BN
  """

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super().__init__()
    self.net = nn.Sequential(
      nn.ReLU(),
      nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.net(x)


class FacConv(nn.Module):
  """ Factorized conv
  ReLU - Conv(Kx1) - Conv(1xK) - BN
  """

  def __init__(self, C_in, C_out, kernel_length, stride, padding, affine=True):
    super().__init__()
    self.net = nn.Sequential(
      nn.ReLU(),
      nn.Conv2d(C_in, C_in, (kernel_length, 1), stride, padding, bias=False),
      nn.Conv2d(C_in, C_out, (1, kernel_length), stride, padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.net(x)


class DilConv(nn.Module):
  """ (Dilated) depthwise separable conv
  ReLU - (Dilated) depthwise separable - Pointwise - BN

  If dilation == 2, 3x3 conv => 5x5 receptive field
                    5x5 conv => 9x9 receptive field
  """

  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation,
               affine=True):
    super().__init__()
    self.net = nn.Sequential(
      nn.ReLU(),
      nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation=dilation,
                groups=C_in,
                bias=False),
      nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.net(x)


class SepConv(nn.Module):
  """ Depthwise separable conv
  DilConv(dilation=1) * 2
  """

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super().__init__()
    self.net = nn.Sequential(
      DilConv(C_in, C_in, kernel_size, stride, padding, dilation=1,
              affine=affine),
      DilConv(C_in, C_out, kernel_size, 1, padding, dilation=1, affine=affine)
    )

  def forward(self, x):
    return self.net(x)


class Zero(nn.Module):
  def __init__(self, stride):
    super().__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x * 0.

    # re-sizing by stride
    return x[:, :, ::self.stride, ::self.stride] * 0.


class MixedOp(nn.Module):
  """ Mixed operation """

  def __init__(self, C, stride):
    super().__init__()
    self._ops = nn.ModuleList()
    for primitive in gt.PRIMITIVES:
      op = OPS[primitive](C, stride, affine=False)
      self._ops.append(op)

  def forward(self, x, weights):
    """
    Args:
        x: input
        weights: weight for each operation
    """
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class ENASLayer(nn.Module):
  '''
  https://github.com/melodyguan/enas/blob/master/src/cifar10/general_child.py#L245
  '''

  def __init__(self, layer_id, in_planes, out_planes, ops,
               track_running_stats=False, upsample=None):
    super(ENASLayer, self).__init__()

    self.layer_id = layer_id
    self.ops = ops
    self.upsample = upsample

    self.branches = nn.ModuleList()
    for op in ops:
      branch = OPS[op](in_planes, out_planes, stride=1, affine=True,
                  track_running_stats=track_running_stats)
      self.branches.append(branch)

    self.bn = nn.BatchNorm2d(
      out_planes, track_running_stats=track_running_stats)

  def forward(self, x, prev_layers, sample_arc):
    layer_type = sample_arc[0]
    if self.layer_id > 0:
      skip_indices = sample_arc[1]
    else:
      skip_indices = []

    x = self.branches[layer_type](x)

    if self.upsample:
      x = self.upsample(x)

    for i, skip in enumerate(skip_indices):
      if skip == 1:
        x += prev_layers[i]
    x = self.bn(x)

    return x


class MixedLayer(nn.Module):

  def __init__(self, layer_id, in_planes, out_planes, ops,
               track_running_stats=False, scalesample=None,
               **kwargs):
    super(MixedLayer, self).__init__()

    self.layer_id = layer_id
    self.out_planes = out_planes
    self.ops = ops
    self.scalesample = scalesample

    self.branches = nn.ModuleList()
    self.num_para_list = []
    for op in ops:
      branch = OPS[op](C_in=in_planes, C_out=out_planes,
                       track_running_stats=track_running_stats,
                       **kwargs)
      self.branches.append(branch)
      self.num_para_list.append(
        sum([p.data.nelement() for p in branch.parameters()]))

    # self.bn = nn.BatchNorm2d(
    #   out_planes, track_running_stats=track_running_stats)

  def forward(self, x, sample_arc):
    bs = len(sample_arc)
    sample_arc = sample_arc.type(torch.int64)
    num_branch = len(self.ops)
    sample_arc_onehot = torch.zeros(bs, num_branch).cuda()
    sample_arc_onehot[torch.arange(bs), sample_arc] = 1
    sample_arc_onehot = sample_arc_onehot.view(bs, num_branch, 1, 1, 1)
    x = [branch(x).unsqueeze(1) if idx in sample_arc else \
           (torch.zeros(bs, self.out_planes, x.size(-1), x.size(-1))
            .cuda().requires_grad_(False).unsqueeze(1)) \
         for idx, branch in enumerate(self.branches)]
    # x = [branch(x).unsqueeze(1) for branch in self.branches]
    x = torch.cat(x, 1)
    x = sample_arc_onehot * x
    x = x.sum(dim=1)

    if self.scalesample:
      x = self.scalesample(x)
    return x



class MixedLayerCond(nn.Module):

  def __init__(self, layer_id, in_planes, out_planes, ops,
               track_running_stats=False, scalesample=None,
               which_bn=None):
    super(MixedLayerCond, self).__init__()

    self.layer_id = layer_id
    self.out_planes = out_planes
    self.ops = ops
    self.scalesample = scalesample

    self.branches = nn.ModuleList()
    self.num_para_list = []
    self.bn = which_bn(in_planes)
    self.act = nn.ReLU()
    for op in ops:
      branch = OPS[op](C_in=in_planes, C_out=out_planes,
                       track_running_stats=track_running_stats,
                       which_act=None)
      self.branches.append(branch)
      self.num_para_list.append(
        sum([p.data.nelement() for p in branch.parameters()]))

    # self.bn = nn.BatchNorm2d(
    #   out_planes, track_running_stats=track_running_stats)

  def forward(self, x, y, sample_arc):
    bs = len(sample_arc)
    sample_arc = sample_arc.type(torch.int64)
    num_branch = len(self.ops)
    sample_arc_onehot = torch.zeros(bs, num_branch).cuda()
    sample_arc_onehot[torch.arange(bs), sample_arc] = 1
    sample_arc_onehot = sample_arc_onehot.view(bs, num_branch, 1, 1, 1)

    x = self.bn(x, y)
    x = self.act(x)
    x = [branch(x, y).unsqueeze(1) if idx in sample_arc else \
           (torch.zeros(bs, self.out_planes, x.size(-1), x.size(-1))
            .cuda().requires_grad_(False).unsqueeze(1)) \
         for idx, branch in enumerate(self.branches)]
    # x = [branch(x, y).unsqueeze(1) for branch in self.branches]
    x = torch.cat(x, 1)
    x = sample_arc_onehot * x
    x = x.sum(dim=1)

    if self.scalesample:
      x = self.scalesample(x)
    return x


class MixedLayerSharedWeights(nn.Module):

  def __init__(self, layer_id, in_planes, out_planes, ops,
               track_running_stats=False, scalesample=None,
               **kwargs):
    super(MixedLayerSharedWeights, self).__init__()

    self.layer_id = layer_id
    self.ops = ops
    self.out_planes = out_planes
    self.scalesample = scalesample

    self.conv_weights = nn.ParameterDict()
    self.conv_bias = nn.ParameterDict()
    self.branches = nn.ModuleList()
    for op in ops:
      if op[-1].isdigit() and op[-3:] not in self.conv_weights:
        ks = int(op[-1])
        key = op[-3:]
        self.conv_weights[key] = nn.Parameter(
          torch.randn(out_planes ,in_planes ,ks ,ks))
        self.conv_bias[key] = nn.Parameter(
          torch.randn(out_planes))
      if op[-1].isdigit():
        key = op[-3:]
        branch = OPS[op](C_in=in_planes, C_out=out_planes,
                         track_running_stats=track_running_stats,
                         weight=self.conv_weights[key],
                         bias=self.conv_bias[key],
                         **kwargs)
        self.branches.append(branch)
      else:
        branch = OPS[op](in_planes, out_planes,
                    track_running_stats=track_running_stats)
        self.branches.append(branch)

    # self.bn = nn.BatchNorm2d(
    #   out_planes, track_running_stats=track_running_stats)

  def forward(self, x, sample_arc):
    bs = len(sample_arc)
    sample_arc = sample_arc.type(torch.int64)
    num_branch = len(self.ops)
    sample_arc_onehot = torch.zeros(bs, num_branch).cuda()
    sample_arc_onehot[torch.arange(bs), sample_arc] = 1
    sample_arc_onehot = sample_arc_onehot.view(bs, num_branch, 1, 1, 1)
    x = [branch(x).unsqueeze(1) if idx in sample_arc else \
           (torch.zeros(bs, self.out_planes, x.size(-1), x.size(-1))
            .cuda().requires_grad_(False).unsqueeze(1)) \
         for idx, branch in enumerate(self.branches)]
    # x = [branch(x).unsqueeze(1) for branch in self.branches]
    x = torch.cat(x, 1)
    x = sample_arc_onehot * x
    x = x.sum(dim=1)

    if self.scalesample:
      x = self.scalesample(x)
    return x


class SinglePathLayer(nn.Module):

  def __init__(self, layer_id, in_planes, out_planes, ops,
               track_running_stats=False, scalesample=None,
               share_bias=True, **kwargs):
    super(SinglePathLayer, self).__init__()

    self.layer_id = layer_id
    self.ops = ops
    self.out_planes = out_planes
    self.scalesample = scalesample

    max_ks = self.get_max_kernel_size()
    self.conv_weights_space = nn.Parameter(
          torch.randn(out_planes ,in_planes ,max_ks ,max_ks))
    self.conv_bias_space = nn.ParameterList()
    # self.conv_bias_space = nn.Parameter(torch.randn(out_planes))
    self.branches = nn.ModuleList()
    self.num_para_list = []
    for op in ops:
      if op[-1].isdigit():
        if share_bias:
          if len(self.conv_bias_space) == 0:
            conv_bias = nn.Parameter(torch.randn(out_planes))
            self.conv_bias_space.append(conv_bias)
          else:
            conv_bias = self.conv_bias_space[0]
        else:
          conv_bias = nn.Parameter(torch.randn(out_planes))
          self.conv_bias_space.append(conv_bias)
        branch = OPS[op](C_in=in_planes, C_out=out_planes,
                         track_running_stats=track_running_stats,
                         weight=self.conv_weights_space,
                         bias=conv_bias,
                         **kwargs)
        self.branches.append(branch)
      else:
        branch = OPS[op](C_in=in_planes, C_out=out_planes,
                         track_running_stats=track_running_stats,
                         **kwargs)
        self.branches.append(branch)

      self.num_para_list.append(0)

    # self.bn = nn.BatchNorm2d(
    #   out_planes, track_running_stats=track_running_stats)

  def get_max_kernel_size(self):
    max_ks = 1
    for op in self.ops:
      if not op[-1].isdigit():
        continue
      ks = int(op[-1])
      max_ks = ks if ks > max_ks else max_ks
    return max_ks

  def forward(self, x, sample_arc):
    if self.scalesample:
      x = self.scalesample(x)

    bs = len(sample_arc)
    sample_arc = sample_arc.type(torch.int64)
    num_branch = len(self.ops)
    sample_arc_onehot = torch.zeros(bs, num_branch).cuda()
    sample_arc_onehot[torch.arange(bs), sample_arc] = 1
    sample_arc_onehot = sample_arc_onehot.view(bs, num_branch, 1, 1, 1)
    x = [branch(x).unsqueeze(1) if idx in sample_arc else \
           (torch.zeros(bs, self.out_planes, x.size(-1), x.size(-1))
            .cuda().requires_grad_(False).unsqueeze(1)) \
         for idx, branch in enumerate(self.branches)]
    # x = [branch(x).unsqueeze(1) for branch in self.branches]
    x = torch.cat(x, 1)
    x = sample_arc_onehot * x
    x = x.sum(dim=1)

    return x


class MixedLayerCondSharedWeights(nn.Module):

  def __init__(self, layer_id, in_planes, out_planes, ops,
               track_running_stats=False, scalesample=None,
               which_bn=None):
    super(MixedLayerCondSharedWeights, self).__init__()

    self.layer_id = layer_id
    self.out_planes = out_planes
    self.ops = ops
    self.scalesample = scalesample

    self.conv_weights = nn.ParameterDict()
    self.conv_bias = nn.ParameterDict()
    self.branches = nn.ModuleList()
    self.num_para_list = []

    self.bn = which_bn(in_planes)
    self.act = nn.ReLU()
    for op in ops:
      if op[-1].isdigit() and op[-3:] not in self.conv_weights:
        ks = int(op[-1])
        key = op[-3:]
        self.conv_weights[key] = nn.Parameter(
          torch.randn(out_planes, in_planes, ks, ks))
        self.conv_bias[key] = nn.Parameter(
          torch.randn(out_planes))
      if op[-1].isdigit():
        key = op[-3:]
        branch = OPS[op](C_in=in_planes, C_out=out_planes,
                         track_running_stats=track_running_stats,
                         weight=self.conv_weights[key],
                         bias=self.conv_bias[key],
                         which_bn=None, which_act=None)
        self.branches.append(branch)
      else:
        branch = OPS[op](C_in=in_planes, C_out=out_planes,
                         track_running_stats=track_running_stats,
                         which_bn=None, which_act=None)
        self.branches.append(branch)
      self.num_para_list.append(0)

    # self.bn = nn.BatchNorm2d(
    #   out_planes, track_running_stats=track_running_stats)

  def forward(self, x, y, sample_arc):
    bs = len(sample_arc)
    sample_arc = sample_arc.type(torch.int64)
    num_branch = len(self.ops)
    sample_arc_onehot = torch.zeros(bs, num_branch).cuda()
    sample_arc_onehot[torch.arange(bs), sample_arc] = 1
    sample_arc_onehot = sample_arc_onehot.view(bs, num_branch, 1, 1, 1)

    x = self.bn(x, y)
    x = self.act(x)
    x = [branch(x, y).unsqueeze(1) if idx in sample_arc else \
           (torch.zeros(bs, self.out_planes, x.size(-1), x.size(-1))
            .cuda().requires_grad_(False).unsqueeze(1)) \
         for idx, branch in enumerate(self.branches)]
    # x = [branch(x, y).unsqueeze(1) for branch in self.branches]
    x = torch.cat(x, 1)
    x = sample_arc_onehot * x
    x = x.sum(dim=1)

    if self.scalesample:
      x = self.scalesample(x)
    return x




def test_MixedLayerCond(args, myargs):
  from template_lib.utils import seed_utils
  seed_utils.set_random_seed(0)

  yaml_file = 'enas_cgan/configs/enas_cgan.yaml'
  with open(yaml_file, 'r') as f:
    configs = EasyDict(yaml.load(f))
  ops = configs.search_cgan_gen_cifar10_v3.model.generator.ops

  bs = 16
  layer_id = 0
  num_layers = 6
  n_classes = 10
  shared_dim = 128
  z_chunk_size = 120 // 5
  G_shared = True
  which_linear = functools.partial(layers.SNLinear,
                                        num_svs=1,
                                        num_itrs=1,
                                        eps=1e-6)
  which_embedding = nn.Embedding
  bn_linear = (functools.partial(which_linear, bias=False) if G_shared
               else which_embedding)
  which_bn = functools.partial(
    layers.ccbn,
    which_linear=bn_linear,
    cross_replica=False,
    mybn=False,
    input_size=(shared_dim + z_chunk_size if G_shared
                else n_classes),
    norm_style='bn',
    eps=1e-5)
  mixedlayer = MixedLayerCond(
    layer_id=layer_id, in_planes=64, out_planes=32,
    ops=ops, track_running_stats=False,
    scalesample=None,
    which_bn=which_bn).cuda()
  x = torch.rand(bs, 64, 4, 4).cuda()
  y = torch.randint(0, n_classes, (bs, )).cuda()
  shared = (which_embedding(n_classes, shared_dim) if G_shared
            else layers.identity()).cuda()

  shared_y = shared(y)
  shared_y = torch.cat((shared_y, torch.rand(bs, z_chunk_size).cuda()), dim=1)
  sample_arcs = torch.randint(0, len(ops), (bs, num_layers))
  out = mixedlayer(x, y=shared_y, sample_arc=sample_arcs[:, layer_id])
  pass


def test_MixedLayerSharedWeights(args, myargs):
  from template_lib.utils import seed_utils
  seed_utils.set_random_seed(0)

  yaml_file = 'enas_cgan/configs/enas_cgan.yaml'
  with open(yaml_file, 'r') as f:
    configs = EasyDict(yaml.load(f))
  ops = configs.search_cgan_gen_cifar10_v4.model.generator.ops

  bs = 16
  layer_id = 0
  num_layers = 6
  in_planes = 3
  out_planes = 32
  n_classes = 10

  mixedlayer = MixedLayerSharedWeights(
    layer_id=layer_id, in_planes=in_planes, out_planes=out_planes,
    ops=ops, track_running_stats=False,
    scalesample=None).cuda()
  x = torch.rand(bs, in_planes, 32, 32).cuda()
  sample_arcs = torch.randint(0, len(ops), (bs, num_layers))
  out = mixedlayer(x, sample_arc=sample_arcs[:, layer_id])
  import torchviz
  g = torchviz.make_dot(out)
  g.view()
  pass


def test_MixedLayerCondSharedWeights(args, myargs):
  from template_lib.utils import seed_utils
  seed_utils.set_random_seed(0)

  yaml_file = 'enas_cgan/configs/enas_cgan.yaml'
  with open(yaml_file, 'r') as f:
    configs = EasyDict(yaml.load(f))
  ops = configs.search_cgan_gen_cifar10_v5.model.generator.ops

  bs = 16
  in_planes = 3
  out_planes = 64
  layer_id = 0
  num_layers = 6
  n_classes = 10
  shared_dim = 128
  z_chunk_size = 120 // 4
  G_shared = True
  which_linear = functools.partial(layers.SNLinear,
                                        num_svs=1,
                                        num_itrs=1,
                                        eps=1e-6)
  which_embedding = nn.Embedding
  bn_linear = (functools.partial(which_linear, bias=False) if G_shared
               else which_embedding)
  which_bn = functools.partial(
    layers.ccbn,
    which_linear=bn_linear,
    cross_replica=False,
    mybn=False,
    input_size=(shared_dim + z_chunk_size if G_shared
                else n_classes),
    norm_style='bn',
    eps=1e-5)
  mixedlayer = MixedLayerCondSharedWeights(
    layer_id=layer_id, in_planes=in_planes, out_planes=out_planes,
    ops=ops, track_running_stats=False,
    scalesample=None,
    which_bn=which_bn).cuda()
  x = torch.rand(bs, in_planes, 32, 32).cuda()
  y = torch.randint(0, n_classes, (bs, )).cuda()
  shared = (which_embedding(n_classes, shared_dim) if G_shared
            else layers.identity()).cuda()

  shared_y = shared(y)
  shared_y = torch.cat((shared_y, torch.rand(bs, z_chunk_size).cuda()), dim=1)
  sample_arcs = torch.randint(0, len(ops), (bs, num_layers))
  out = mixedlayer(x, y=shared_y, sample_arc=sample_arcs[:, layer_id])
  import torchviz
  g = torchviz.make_dot(out)
  g.view()
  pass


def test_SinglePathLayer(args, myargs):
  from template_lib.utils import seed_utils
  seed_utils.set_random_seed(0)

  yaml_file = 'enas_cgan/configs/enas_cgan.yaml'
  with open(yaml_file, 'r') as f:
    configs = EasyDict(yaml.load(f))
  ops = configs.search_cgan_gen_cifar10_v6.model.generator.ops

  bs = 16
  layer_id = 0
  num_layers = 6
  in_planes = 3
  out_planes = 32
  n_classes = 10

  mixedlayer = SinglePathLayer(
    layer_id=layer_id, in_planes=in_planes, out_planes=out_planes,
    ops=ops, track_running_stats=False,
    scalesample=None, bn_type='none').cuda()
  bs = len(ops)
  x = torch.rand(bs, in_planes, 32, 32).cuda()
  # sample_arcs = torch.randint(0, len(ops), (bs, num_layers))
  # sample_arc = sample_arcs[:, layer_id]
  sample_arc = torch.arange(0, bs)
  out = mixedlayer(x, sample_arc=sample_arc)
  import torchviz
  g = torchviz.make_dot(out)
  g.view()
  pass