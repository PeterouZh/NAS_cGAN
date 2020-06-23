import collections
import logging
import os
import weakref
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import tqdm

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data.catalog import DatasetCatalog
from detectron2.data import (
  MetadataCatalog,
  build_detection_test_loader,
  build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import (
  COCOEvaluator,
  COCOPanopticEvaluator,
  DatasetEvaluators,
  LVISEvaluator,
  PascalVOCDetectionEvaluator,
  SemSegEvaluator,
  inference_on_dataset,
  print_csv_format,
)
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
  CommonMetricPrinter,
  EventStorage,
  JSONWriter,
  TensorboardXWriter,
)

from template_lib.utils import detection2_utils, get_attr_eval, get_attr_kwargs
from template_lib.utils.detection2_utils import D2Utils
from template_lib.utils import modelarts_utils
from template_lib.utils import seed_utils
from template_lib.utils.modelarts_utils import prepare_dataset
from template_lib.d2.data import build_dataset_mapper
from template_lib.d2template.trainer import build_trainer
from template_lib.d2template.scripts import build_start, START_REGISTRY
from template_lib.d2.data import build_cifar10

import nas_cgan.scripts

logger = logging.getLogger("detectron2")


@START_REGISTRY.register()
def plot_flops_cifar10(cfg, args, myargs):
  import matplotlib.pyplot as plt
  import pickle

  flops_pickle_files                   = cfg.flops_pickle_files
  n_classes                            = cfg.n_classes

  # plt.style.use('seaborn')
  fig, ax = plt.subplots()
  fig.show()
  n_colors = len(flops_pickle_files)
  colors = [plt.cm.cool(i / float(n_colors - 1)) for i in range(n_colors)]

  ax.set_xticks(range(n_classes))
  ax.set_xlabel('Class', fontsize=15)
  ax.set_ylabel(r'FLOPs $(10^9)$', fontsize=15)
  ax.tick_params(labelsize=14)

  for idx, (label, v_dict) in enumerate(flops_pickle_files.items()):
    with open(v_dict["pickle_file"], 'rb') as f:
      flops = pickle.load(f)
      ax.scatter(flops[:, 0], flops[:, 1], color=colors[idx], **v_dict['properties'])

  # ax.legend()
  ax.legend(loc='lower right', prop={'size': 10})
  saved_file = os.path.join(myargs.args.outdir, 'flops_cifar10.pdf')
  # fig.tight_layout()
  fig.savefig(saved_file, bbox_inches = 'tight', pad_inches=0.01)
  print(f'Saved to {saved_file}')
  pass

@START_REGISTRY.register()
def compute_op_proportion_in_arcs(cfg, args, myargs):
  import matplotlib.pyplot as plt
  import matplotlib.ticker as mtick
  import numpy as np
  from nas_cgan.models.trainer_nasgan import TrainerNASGAN

  fixed_arc_file                    = cfg.fixed_arc_file
  fixed_epoch                       = cfg.fixed_epoch
  nrows                             = cfg.nrows
  ops                               = cfg.ops
  saved_file                        = cfg.saved_file
  yticks                            = cfg.yticks

  class_arcs = TrainerNASGAN._get_arc_from_file(fixed_arc_file, fixed_epoch, nrows=nrows)
  n_layers = class_arcs.shape[1]

  fig, ax = plt.subplots()
  colors = [plt.cm.cool(i / float(len(ops) - 1)) for i in range(len(ops))]
  # colors = [plt.cm.spring(i / float(len(ops) - 1)) for i in range(len(ops))]

  ax.tick_params(labelsize=14)
  ax.set_xlim([0, n_layers-1])
  ax.set_ylim([0, 100])
  ax.set_yticks(yticks)
  ax.yaxis.set_major_formatter(mtick.PercentFormatter())
  ax.set_xlabel('Layer', fontsize=20)
  # ax.set_ylabel(None, fontsize=10)

  proportion = collections.defaultdict(list)
  for layer_idx in range(class_arcs.shape[1]):
    unique_elements, counts_elements = np.unique(class_arcs[:, layer_idx], return_counts=True)
    unique_elements = unique_elements.tolist()
    counts_elements = list(counts_elements / counts_elements.sum() * 100)
    for op_idx in range(len(ops)):
      if op_idx in unique_elements:
        proportion[op_idx].append(counts_elements[unique_elements.index(op_idx)])
      else:
        proportion[op_idx].append(0)

  layer = np.arange(0, class_arcs.shape[1])
  proportion_list = []
  for p in proportion:
    proportion_list.append(proportion[p])

  proportion_list = reversed(proportion_list)
  ops = reversed(ops)
  ax.stackplot(layer, *proportion_list, labels=ops, colors=colors)
  ax.legend(loc='upper right', prop={'size': 20})
  # ax.legend()
  saved_file = os.path.join(myargs.args.outdir, saved_file)
  fig.savefig(saved_file, bbox_inches='tight', pad_inches=0.01)
  print(f'Save to {saved_file}')
  pass

@START_REGISTRY.register()
def do_test(cfg, args, myargs):

  eval_ckpt_dir            = cfg.start.eval_ckpt_dir
  eval_epoch               = cfg.start.eval_epoch
  dataset_name             = cfg.start.dataset_name
  IMS_PER_BATCH            = cfg.start.IMS_PER_BATCH

  cfg.defrost()

  cfg.freeze()

  # build dataset
  DatasetCatalog.get(dataset_name)
  metadata = MetadataCatalog.get(dataset_name)
  num_images = metadata.get('num_images')
  iter_every_epoch = num_images // IMS_PER_BATCH

  model = build_trainer(cfg, myargs=myargs, iter_every_epoch=1, img_size=cfg.dataset.img_size, train_bs=32)

  logger.info("Model:\n{}".format(model))

  eval_iter = (eval_epoch) * iter_every_epoch - 1
  eval_ckpt = os.path.join(eval_ckpt_dir, f'model_{eval_iter:07}.pth')
  model.eval_func(eval_ckpt=eval_ckpt)


@START_REGISTRY.register()
def do_train(cfg, args, myargs):
  run_func                                     = cfg.start.get('run_func', 'train_func')
  dataset_name                                 = cfg.start.dataset_name
  IMS_PER_BATCH                                = cfg.start.IMS_PER_BATCH
  max_epoch                                    = cfg.start.max_epoch
  ASPECT_RATIO_GROUPING                        = cfg.start.ASPECT_RATIO_GROUPING
  NUM_WORKERS                                  = cfg.start.NUM_WORKERS
  checkpoint_period                            = cfg.start.checkpoint_period
  dataset_mapper                               = cfg.start.dataset_mapper
  resume_ckpt_dir                              = get_attr_kwargs(cfg.start, 'resume_ckpt_dir', default=None)
  resume_ckpt_epoch                            = get_attr_kwargs(cfg.start, 'resume_ckpt_epoch', default=0)
  resume_ckpt_iter_every_epoch                 = get_attr_kwargs(cfg.start, 'resume_ckpt_iter_every_epoch', default=0)

  cfg.defrost()
  cfg.DATASETS.TRAIN                           = (dataset_name, )
  cfg.SOLVER.IMS_PER_BATCH                     = IMS_PER_BATCH
  cfg.DATALOADER.ASPECT_RATIO_GROUPING         = ASPECT_RATIO_GROUPING
  cfg.DATALOADER.NUM_WORKERS                   = NUM_WORKERS
  cfg.freeze()

  # build dataset
  mapper = build_dataset_mapper(dataset_mapper)
  data_loader = build_detection_train_loader(cfg, mapper=mapper)
  metadata = MetadataCatalog.get(dataset_name)
  num_images = metadata.get('num_images')
  iter_every_epoch = num_images // IMS_PER_BATCH
  max_iter = iter_every_epoch * max_epoch

  model = build_trainer(cfg, myargs=myargs, iter_every_epoch=iter_every_epoch, img_size=dataset_mapper.img_size,
                        dataset_name=dataset_name, train_bs=IMS_PER_BATCH, max_iter=max_iter)
  model.train()

  # optimizer = build_optimizer(cfg, model)
  optims_dict = model.build_optimizer()
  # scheduler = build_lr_scheduler(cfg, optimizer)

  checkpointer = DetectionCheckpointer(model.get_saved_model(), cfg.OUTPUT_DIR, **optims_dict)
  if args.resume:
    resume_ckpt_dir = model._get_ckpt_path(ckpt_dir=resume_ckpt_dir, ckpt_epoch=resume_ckpt_epoch,
                                           iter_every_epoch=resume_ckpt_iter_every_epoch)
    start_iter = (checkpointer.resume_or_load(resume_ckpt_dir).get("iteration", -1) + 1)
    if get_attr_kwargs(args, 'finetune', default=False):
      start_iter = 0
  else:
    start_iter = 0

  model.after_resume()

  if run_func != 'train_func':
    eval(f'model.{run_func}()')
    exit(0)

  checkpoint_period = eval(checkpoint_period, dict(iter_every_epoch=iter_every_epoch))
  periodic_checkpointer = PeriodicCheckpointer(checkpointer, checkpoint_period, max_iter=max_iter)
  logger.info("Starting training from iteration {}".format(start_iter))
  modelarts_utils.modelarts_sync_results(args=myargs.args, myargs=myargs, join=True, end=False)
  with EventStorage(start_iter) as storage:
    pbar = zip(data_loader, range(start_iter, max_iter))
    if comm.is_main_process():
      pbar = tqdm.tqdm(pbar,
                       desc=f'do_train, {myargs.args.time_str_suffix}, '
                            f'iters {iter_every_epoch} * bs {IMS_PER_BATCH} = imgs {iter_every_epoch*IMS_PER_BATCH}',
                       file=myargs.stdout,
                       initial=start_iter, total=max_iter)

    for data, iteration in pbar:
      comm.synchronize()
      iteration = iteration + 1
      storage.step()

      model.train_func(data, iteration - 1, pbar=pbar)

      periodic_checkpointer.step(iteration)
      pass
  modelarts_utils.modelarts_sync_results(args=myargs.args, myargs=myargs, join=True, end=True)
  comm.synchronize()


def setup(args, config):
  """
  Create configs and perform basic setups.
  """
  from detectron2.config import CfgNode
  # detectron2 default cfg
  # cfg = get_cfg()
  cfg = CfgNode()
  cfg.OUTPUT_DIR = "./output"
  cfg.SEED = -1
  cfg.CUDNN_BENCHMARK = False
  cfg.DATASETS = CfgNode()
  cfg.SOLVER = CfgNode()

  cfg.DATALOADER = CfgNode()
  cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
  cfg.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"

  cfg.MODEL = CfgNode()
  cfg.MODEL.KEYPOINT_ON = False
  cfg.MODEL.LOAD_PROPOSALS = False
  cfg.MODEL.WEIGHTS = ""

  # cfg.merge_from_file(args.config_file)
  cfg.merge_from_list(args.opts)

  cfg = detection2_utils.D2Utils.cfg_merge_from_easydict(cfg, config)

  cfg.freeze()
  default_setup(
    cfg, args
  )  # if you don't like any of the default setup, write your own setup code
  return cfg


def main(args, myargs):
  cfg = setup(args, myargs.config)
  myargs = D2Utils.setup_myargs_for_multiple_processing(myargs)
  # seed_utils.set_random_seed(cfg.seed)

  build_start(cfg=cfg, args=args, myargs=myargs)

  modelarts_utils.modelarts_sync_results(args=myargs.args, myargs=myargs, join=True, end=True)
  return


def run(argv_str=None):
  from template_lib.utils.config import parse_args_and_setup_myargs, config2args
  run_script = os.path.relpath(__file__, os.getcwd())
  args1, myargs, _ = parse_args_and_setup_myargs(argv_str, run_script=run_script, start_tb=False)
  myargs.args = args1
  myargs.config = getattr(myargs.config, args1.command)

  if hasattr(myargs.config, 'datasets'):
    prepare_dataset(myargs.config.datasets, cfg=myargs.config)

  args = default_argument_parser().parse_args(args=[])
  args = config2args(myargs.config.args, args)

  args.opts += ['OUTPUT_DIR', args1.outdir + '/detectron2']
  print("Command Line Args:", args)

  myargs = D2Utils.unset_myargs_for_multiple_processing(myargs, num_gpus=args.num_gpus)

  launch(
    main,
    args.num_gpus,
    num_machines=args.num_machines,
    machine_rank=args.machine_rank,
    dist_url=args.dist_url,
    args=(args, myargs),
  )


if __name__ == "__main__":
  run()
  # from template_lib.examples import test_bash
  # test_bash.TestingUnit().test_resnet(gpu=os.environ['CUDA_VISIBLE_DEVICES'])
