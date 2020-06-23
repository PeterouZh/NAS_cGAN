import os
import numpy as np
import tqdm
import torch
from torch import nn

from detectron2.utils import comm
from detectron2.checkpoint import Checkpointer
from template_lib.utils import get_ddp_attr, print_number_params, get_attr_kwargs
from template_lib.gans.models import build_GAN_model

from .build import TRAINER_REGISTRY
from . import PAGANSearch


@TRAINER_REGISTRY.register()
class PAGANRetrain(PAGANSearch):

  def __init__(self, cfg, **kwargs):
    super(PAGANRetrain, self).__init__(cfg=cfg, **kwargs)

    self.fixed_arc_file               = get_attr_kwargs(cfg.trainer, 'fixed_arc_file', **kwargs)
    self.fixed_epoch                  = get_attr_kwargs(cfg.trainer, 'fixed_epoch', **kwargs)

    self.classes_arcs = self.get_fixed_arc(self.fixed_arc_file, self.fixed_epoch)
    pass

  def build_controller(self, cfg, num_layers, num_branches):

    controller = None
    controller_optim = None
    return controller, controller_optim

  def train_func(self, data, iteration, pbar):
    if comm.is_main_process() and iteration % self.iter_every_epoch == 0:
      pbar.set_postfix_str(s="Retrain ")
    images, labels = self.preprocess_image(data)
    images = images.tensor

    # FairNAS architecture sampling
    batched_arcs = self.classes_arcs[labels]

    self.gan_model(images=images, labels=labels, z=self.z_train, iteration=iteration, batched_arcs=batched_arcs)

    self.evaluate_model(classes_arcs=self.classes_arcs, iteration=iteration)
    comm.synchronize()

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

  def _strip_module_if_present(self, state_dict, prefix='.module.'):

    keys = sorted(state_dict.keys())
    if not all(len(key) == 0 or prefix in key for key in keys):
        return

    for key in keys:
        newkey = key.replace(prefix, '.')
        state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata, if any..
    try:
        metadata = state_dict._metadata
    except AttributeError:
        pass
    else:
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key.replace(prefix, '.')
            metadata[newkey] = metadata.pop(key)

  def _load_G(self, eval_ckpt):
    checkpointer = Checkpointer(self.get_saved_model(), save_to_disk=False)

    # strip .module.
    checkpoint = checkpointer._load_file(eval_ckpt)
    checkpoint_state_dict = checkpoint['model']
    checkpointer._convert_ndarray_to_tensor(checkpoint_state_dict)
    self._strip_module_if_present(checkpoint_state_dict)

    self.logger.info(f"Load model from {eval_ckpt}")
    checkpointer._load_model(checkpoint)
    pass

  def eval_func(self, eval_ckpt):
    self._load_G(eval_ckpt)
    self.evaluate_model(classes_arcs=self.classes_arcs, iteration=0)
    comm.synchronize()

