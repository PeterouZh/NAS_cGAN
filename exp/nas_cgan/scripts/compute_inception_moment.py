import functools
import logging
import os
import tqdm
import numpy as np
import torch
import torch.utils.data

import detectron2.utils.comm as comm
from detectron2.data import (
  get_detection_dataset_dicts,
  DatasetFromList, DatasetMapper, MapDataset, samplers,
)
from detectron2.checkpoint import DetectionCheckpointer

from template_lib.utils import modelarts_utils, get_attr_kwargs
from template_lib.d2.data import build_dataset_mapper
from template_lib.gans.evaluation import build_GAN_metric_dict
from template_lib.trainer.base_trainer import Trainer
from template_lib.d2template.trainer import build_trainer
from template_lib.d2template.scripts import START_REGISTRY

logger = logging.getLogger("detectron2")


def trivial_batch_collator(batch):
  """
  A batch collator that does nothing.
  """
  return batch


def build_detection_test_loader(cfg, dataset_name, batch_size, mapper=None):
  """
  Similar to `build_detection_train_loader`.
  But this function uses the given `dataset_name` argument (instead of the names in cfg),
  and uses batch size 1.

  Args:
      cfg: a detectron2 CfgNode
      dataset_name (str): a name of the dataset that's available in the DatasetCatalog
      mapper (callable): a callable which takes a sample (dict) from dataset
         and returns the format to be consumed by the model.
         By default it will be `DatasetMapper(cfg, False)`.

  Returns:
      DataLoader: a torch DataLoader, that loads the given detection
      dataset, with test-time transformation and batching.
  """
  dataset_dicts = get_detection_dataset_dicts(
    [dataset_name],
    filter_empty=False,
    proposal_files=[
      cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]
    ]
    if cfg.MODEL.LOAD_PROPOSALS
    else None,
  )

  dataset = DatasetFromList(dataset_dicts)
  if mapper is None:
    mapper = DatasetMapper(cfg, False)
  dataset = MapDataset(dataset, mapper)

  sampler = samplers.InferenceSampler(len(dataset))
  # Always use 1 image per worker during inference since this is the
  # standard when reporting inference time in papers.
  batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last=False)

  data_loader = torch.utils.data.DataLoader(
    dataset,
    num_workers=cfg.DATALOADER.NUM_WORKERS,
    batch_sampler=batch_sampler,
    collate_fn=trivial_batch_collator,
  )
  return data_loader


@START_REGISTRY.register()
def compute_fid_stats(cfg, args, myargs):

  dataset_name                 = cfg.start.dataset_name
  IMS_PER_BATCH                = cfg.start.IMS_PER_BATCH
  img_size                     = cfg.start.img_size
  NUM_WORKERS                  = cfg.start.NUM_WORKERS

  cfg.defrost()
  cfg.DATALOADER.NUM_WORKERS = NUM_WORKERS
  cfg.GAN_metric.torch_fid_stat = cfg.GAN_metric.torch_fid_stat.format(
    dataset_name=dataset_name, img_size=img_size)
  cfg.GAN_metric.tf_fid_stat = cfg.GAN_metric.tf_fid_stat.format(
    dataset_name=dataset_name, img_size=img_size)
  cfg.freeze()

  num_workers = comm.get_world_size()
  batch_size = IMS_PER_BATCH // num_workers

  dataset_mapper = build_dataset_mapper(cfg.dataset_mapper, img_size=img_size)
  data_loader = build_detection_test_loader(
    cfg, dataset_name=dataset_name, batch_size=batch_size, mapper=dataset_mapper)

  metric_dict = build_GAN_metric_dict(cfg)
  if "PyTorchFIDISScore" in metric_dict:
    FID_IS_torch = metric_dict['PyTorchFIDISScore']
    FID_IS_torch.calculate_fid_stat_of_dataloader(data_loader=data_loader, stdout=myargs.stdout)
  if "TFFIDISScore" in metric_dict:
    FID_IS_tf = metric_dict['TFFIDISScore']
    FID_IS_tf.calculate_fid_stat_of_dataloader(data_loader=data_loader, stdout=myargs.stdout)

  comm.synchronize()
  return


@START_REGISTRY.register()
def compute_fid_stats_per_class(cfg, args, myargs):

  imagenet_root_dir            = cfg.start.imagenet_root_dir
  dataset_name                 = cfg.start.dataset_name
  IMS_PER_BATCH                = cfg.start.IMS_PER_BATCH
  img_size                     = cfg.start.img_size
  NUM_WORKERS                  = cfg.start.NUM_WORKERS
  torch_fid_stat               = cfg.GAN_metric.torch_fid_stat
  tf_fid_stat                  = cfg.GAN_metric.tf_fid_stat

  if dataset_name.startswith('cifar10_train_per_class'):
    from template_lib.d2.data.build_cifar10_per_class import find_classes
  elif dataset_name.startswith('cifar100_train_per_class'):
    from template_lib.d2.data.build_cifar100_per_class import find_classes

  torch_fid_stat = torch_fid_stat.format(dataset_name=dataset_name, img_size=img_size)
  tf_fid_stat = tf_fid_stat.format(dataset_name=dataset_name, img_size=img_size)
  cfg.defrost()
  cfg.DATALOADER.NUM_WORKERS = NUM_WORKERS
  cfg.GAN_metric.torch_fid_stat = torch_fid_stat
  cfg.GAN_metric.tf_fid_stat = tf_fid_stat
  cfg.freeze()

  num_workers = comm.get_world_size()
  batch_size = IMS_PER_BATCH // num_workers

  metric_dict = build_GAN_metric_dict(cfg)
  if "PyTorchFIDISScore" in metric_dict:
    os.makedirs(torch_fid_stat, exist_ok=True)
    FID_IS_torch = metric_dict['PyTorchFIDISScore']
  if "TFFIDISScore" in metric_dict:
    os.makedirs(tf_fid_stat, exist_ok=True)
    FID_IS_tf = metric_dict['TFFIDISScore']
    # FID_IS_tf.calculate_fid_stat_of_dataloader(data_loader=data_loader, stdout=myargs.stdout)

  classes, class_to_idx = find_classes(imagenet_root_dir)
  dataset_mapper = build_dataset_mapper(cfg.dataset_mapper, img_size=img_size)
  comm.synchronize()

  for class_path, idx in tqdm.tqdm(class_to_idx.items(), desc=f"compute_fid_stats_per_class"):
    registed_name = f'{dataset_name}_{class_path}'
    data_loader = build_detection_test_loader(
      cfg, dataset_name=registed_name, batch_size=batch_size, mapper=dataset_mapper)

    if "PyTorchFIDISScore" in metric_dict:
      mu, sigma = FID_IS_torch.calculate_fid_stat_of_dataloader(
        data_loader=data_loader, return_fid_stat=True, stdout=myargs.stdout)
      if comm.is_main_process():
        np.savez(os.path.join(torch_fid_stat, f'{idx}.npz'), **{'mu': mu, 'sigma': sigma})

    if "TFFIDISScore" in metric_dict:
      mu, sigma = FID_IS_tf.calculate_fid_stat_of_dataloader(
        data_loader=data_loader, return_fid_stat=True, stdout=myargs.stdout)
      if comm.is_main_process():
        np.savez(os.path.join(tf_fid_stat, f'{idx}.npz'), **{'mu': mu, 'sigma': sigma})

  comm.synchronize()
  return


def _get_last_checkpoint_file(model_dir):
  """
  Returns:
      str: The latest checkpoint file in target directory.
  """
  save_file = os.path.join(model_dir, "last_checkpoint")
  try:
    with open(save_file, "r") as f:
      last_saved = f.read().strip()
  except IOError:
    # if file doesn't exist, maybe because it has just been
    # deleted by a separate process
    return ""
  return os.path.join(model_dir, last_saved)

def _sample_func_with_arcs(G, z, y, arcs):
  G.eval()
  with torch.no_grad():
    z.sample_()
    y = z.new_full((z.size(0),), fill_value=y, dtype=torch.int64)
    # y.sample_()
    sample_arcs = arcs[y]
    G_z = G(z, sample_arcs=sample_arcs, y=y)
  return G_z

@START_REGISTRY.register()
def compute_intra_FID(cfg, args, myargs):
  # register all class of ImageNet for dataloader
  # from template_lib.d2.data.build_ImageNet_per_class import ImageNetDatasetPerClassMapper

  imagenet_root_dir            = cfg.start.imagenet_root_dir
  model_path                   = cfg.start.model_path
  model_file                   = cfg.start.model_file
  use_last_checkpoint          = cfg.start.use_last_checkpoint
  eval_tf                      = cfg.start.eval_tf
  tf_fid_stats_dir             = cfg.start.tf_fid_stats_dir
  num_inception_images         = cfg.start.num_inception_images // comm.get_world_size()
  intra_FID_tfs_file           = cfg.start.intra_FID_tfs_file
  eval_torch                   = cfg.start.eval_torch
  torch_fid_stats_dir          = cfg.start.torch_fid_stats_dir
  intra_FID_torchs_file        = cfg.start.intra_FID_torchs_file

  intra_FID_tfs_file = os.path.join(myargs.args.outdir, intra_FID_tfs_file)
  intra_FID_torchs_file = os.path.join(myargs.args.outdir, intra_FID_torchs_file)

  model = build_trainer(cfg, myargs=myargs)

  # optims_dict = model.build_optimizer()
  checkpointer = DetectionCheckpointer(
    model, cfg.OUTPUT_DIR)
  if use_last_checkpoint:
    model_path = _get_last_checkpoint_file(model_dir=model_path)
  else:
    model_path = os.path.join(model_path, model_file)

  start_iter = (
        checkpointer.resume_or_load(model_path, resume=args.resume).get("iteration", -1) + 1
  )

  model.evaluate_model(iteration=0, fixed_arc=model.fixed_arc)

  classes, class_to_idx = find_classes(imagenet_root_dir)
  intra_FID_tfs = []
  intra_FID_torchs = []
  for class_dir, idx in tqdm.tqdm(class_to_idx.items(),
                                  desc=f"compute intra FID {myargs.args.time_str_suffix}", file=myargs.stdout):
    if eval_tf:
      mu_sigma = np.load(os.path.join(tf_fid_stats_dir, f'{idx}.npz'))
      class_mu, class_sigma = mu_sigma['mu'], mu_sigma['sigma']
      sample_func = functools.partial(
        _sample_func_with_arcs, G=model.G, z=model.z_test, y=idx, arcs=model.fixed_arc)

      mu, sigma = model.FID_IS_tf(sample_func, return_fid_stat=True, num_inception_images=num_inception_images,
                                  stdout=myargs.stdout)
      if comm.is_main_process():
        intra_FID_tf = model.FID_IS_tf._calculate_frechet_distance(mu, sigma, class_mu, class_sigma)
        intra_FID_tfs.append(intra_FID_tf)
        Trainer.summary_dict2txtfig(dict_data=dict(intra_FID_tf=intra_FID_tf), prefix='intraFIDtf',
                                    step=idx, textlogger=myargs.textlogger)
        modelarts_utils.modelarts_sync_results(args=myargs.args, myargs=myargs, join=False)

    if eval_torch:
      mu_sigma = np.load(os.path.join(torch_fid_stats_dir, f'{idx}.npz'))
      class_mu, class_sigma = mu_sigma['mu'], mu_sigma['sigma']
      sample_func = functools.partial(
        _sample_func_with_arcs, G=model.G, z=model.z_test, y=idx, arcs=model.fixed_arc)

      mu, sigma = model.FID_IS_pytorch(sample_func, return_fid_stat=True, num_inception_images=num_inception_images,
                                       stdout=myargs.stdout)
      if comm.is_main_process():
        intra_FID_torch = model.FID_IS_pytorch._calculate_frechet_distance(mu, sigma, class_mu, class_sigma)
        intra_FID_torchs.append(intra_FID_torch)
        Trainer.summary_dict2txtfig(dict_data=dict(intra_FID_torch=intra_FID_torch), prefix='intraFIDtorch',
                                    step=idx, textlogger=myargs.textlogger)
        modelarts_utils.modelarts_sync_results(args=myargs.args, myargs=myargs, join=False)

  if len(intra_FID_tfs) > 0:
    intra_FID_tfs = np.array(intra_FID_tfs)
    np.savez(intra_FID_tfs_file, intra_FID_tfs=intra_FID_tfs)
  if len(intra_FID_torchs) > 0:
    intra_FID_torchs = np.array(intra_FID_torchs)
    np.savez(intra_FID_torchs_file, intra_FID_torchs=intra_FID_torchs)
  comm.synchronize()
  return
