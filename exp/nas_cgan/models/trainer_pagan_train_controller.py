import json

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from detectron2.utils import comm
from detectron2.checkpoint import DetectionCheckpointer, Checkpointer

from template_lib.utils import get_ddp_attr, get_ddp_module
from template_lib.gans.models import build_GAN_model
from template_lib.d2.models import build_d2model
from template_lib.d2.optimizer import build_optimizer


from .build import TRAINER_REGISTRY
from . import PAGANSearch


@TRAINER_REGISTRY.register()
class PAGANTrainController(PAGANSearch):

  def __init__(self, cfg, **kwargs):
    super(PAGANTrainController, self).__init__(cfg=cfg, **kwargs)

    self.trained_supernet_ckpt_dir           = cfg.trainer.trained_supernet_ckpt_dir
    self.ckpt_epoch                          = cfg.trainer.ckpt_epoch
    self.supernet_iter_every_epoch           = cfg.trainer.supernet_iter_every_epoch

    self.load_G()
    pass

  def build_controller(self, cfg, num_layers, num_branches):

    controller = build_d2model(cfg.model.controller, n_classes=self.n_classes, num_layers=num_layers,
                               num_branches=num_branches, FID_IS=self.FID_IS_pytorch,
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

  def load_G(self):
    ckpt_path = self._get_ckpt_path(ckpt_dir=self.trained_supernet_ckpt_dir, ckpt_epoch=self.ckpt_epoch,
                                    iter_every_epoch=self.supernet_iter_every_epoch)
    saved_model = self.get_saved_model()
    checkpointer = Checkpointer(saved_model, save_to_disk=False)
    loaded_checkpoint = checkpointer.resume_or_load(ckpt_path, resume=False)
    pass

  def train_func(self, data, iteration, pbar):
    # images, labels = self.preprocess_image(data)
    # images = images.tensor

    get_ddp_attr(self.controller, 'train_controller')(G=self.G, z=self.z_train, y=self.y_train,
                    controller=self.controller, controller_optim=self.controller_optim,
                    iteration=iteration, pbar=pbar)

    # Just for monitoring the training processing
    # sync arcs
    sampled_arc = get_ddp_attr(self.controller, 'get_sampled_arc')()
    sampled_arc = self.get_tensor_of_main_processing(sampled_arc)

    classes_arcs = sampled_arc[[0,],].repeat(self.n_classes, 1)
    self.evaluate_model(classes_arcs=classes_arcs, iteration=iteration)
    comm.synchronize()

  def _get_iter_per_worker(self, iteration):
    iter_per_worker = iteration
    return iter_per_worker

  def _get_eval_iter(self, iteration):
    # if iteration % self.eval_torch_every_itr == 0:
    #   get_ddp_attr(self.controller, 'print_distribution')(iteration=iteration)
    return iteration


@TRAINER_REGISTRY.register()
class EvalArcInSuperNet(PAGANTrainController):

  def __init__(self, cfg, **kwargs):
    super(EvalArcInSuperNet, self).__init__(cfg=cfg, **kwargs)

    self.layer_op_idx                 = cfg.trainer.layer_op_idx
    self.layer_op_idx                 = json.loads(self.layer_op_idx.replace(' ', ', ').replace('][', '], [').strip())

    pass

  def train_func(self, data, iteration, pbar):
    # images, labels = self.preprocess_image(data)
    # images = images.tensor

    # Just for monitoring the training processing
    # sync arcs

    sampled_arc = torch.tensor(self.layer_op_idx).view(1, -1).to(self.device)

    classes_arcs = sampled_arc[[0, ],].repeat(self.n_classes, 1)
    self.evaluate_model(classes_arcs=classes_arcs, iteration=iteration)
    comm.synchronize()

  def _get_eval_iter(self, iteration):
    return iteration
