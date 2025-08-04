import torch
import logging
import pdb
import os
import datetime
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import cfg
from data import make_data_loader
from solver import build_optimizer, build_scheduler

from utils.check_point import DetectronCheckpointer
from engine import (
    default_argument_parser,
    default_setup,
    launch,
)
from utils import comm
from utils.backup_files import sync_root

from engine.trainer import do_train
from engine.test_net import run_test

from model.detector import KeypointDetector
from data import build_test_loader
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

torch.backends.cudnn.enabled = True # enable cudnn and uncertainty imported
# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True # enable cudnn to search the best algorithm

def freeze_backbone(model):
    """
    Freeze the DLA backbone parameters to prevent them from being updated during training
    """
    print("Freezing DLA backbone parameters...")
    
    # Freeze the backbone (DLA)
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # Also freeze the DLA up and IDA up modules
    if hasattr(model.backbone, 'dla_up'):
        for param in model.backbone.dla_up.parameters():
            param.requires_grad = False
    
    if hasattr(model.backbone, 'ida_up'):
        for param in model.backbone.ida_up.parameters():
            param.requires_grad = False
    
    # Print statistics
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    
    return model

def finetune(cfg, model, device, distributed):
    """
    Finetuning function that freezes the backbone and only trains the detection heads
    """
    data_loader = make_data_loader(cfg, is_train=True)
    data_loaders_val = build_test_loader(cfg, is_train=False)

    total_iters_each_epoch = len(data_loader.dataset) // cfg.SOLVER.IMS_PER_BATCH
    # use epoch rather than iterations for saving checkpoint and validation
    if cfg.SOLVER.EVAL_AND_SAVE_EPOCH:
        cfg.SOLVER.MAX_ITERATION = cfg.SOLVER.MAX_EPOCHS * total_iters_each_epoch
        cfg.SOLVER.SAVE_CHECKPOINT_INTERVAL = total_iters_each_epoch * cfg.SOLVER.SAVE_CHECKPOINT_EPOCH_INTERVAL
        cfg.SOLVER.EVAL_INTERVAL = total_iters_each_epoch * cfg.SOLVER.EVAL_EPOCH_INTERVAL
        cfg.SOLVER.STEPS = [total_iters_each_epoch * x for x in cfg.SOLVER.DECAY_EPOCH_STEPS]
        cfg.SOLVER.WARMUP_STEPS = cfg.SOLVER.WARMUP_EPOCH * total_iters_each_epoch
    
    cfg.freeze()

    # Build optimizer FIRST (like original script)
    optimizer = build_optimizer(model, cfg)
    scheduler, warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=total_iters_each_epoch, 
        optim_cfg=cfg.SOLVER,
    )

    arguments = {}
    arguments["iteration"] = 0
    arguments["iter_per_epoch"] = total_iters_each_epoch

    output_dir = cfg.OUTPUT_DIR
    save_to_disk = comm.get_rank() == 0
    if not cfg.SOLVER.LOAD_OPTIMIZER_SCHEDULER:
        checkpointer = DetectronCheckpointer(cfg, model, None, None, output_dir, save_to_disk)
    else:
        checkpointer = DetectronCheckpointer(
            cfg, model, None, None, output_dir, save_to_disk
        )

    # Load the pretrained model for finetuning
    if len(cfg.MODEL.WEIGHT) > 0:
        extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, use_latest=False)
        if not cfg.SOLVER.LOAD_OPTIMIZER_SCHEDULER:
            extra_checkpoint_data.pop("optimizer", None)
            extra_checkpoint_data.pop("scheduler", None)
            extra_checkpoint_data["iteration"] = 0  # Reset iteration
        arguments.update(extra_checkpoint_data)
        print(f"Loaded pretrained model from {cfg.MODEL.WEIGHT}")
    else:
        print("Warning: No pretrained model specified. Please set MODEL.WEIGHT in config.")
        if "iteration" not in arguments:
            arguments["iteration"] = 0

    # Freeze the backbone AFTER loading the pretrained weights
    model = freeze_backbone(model)
    
    # Adjust learning rates for finetuning (backbone should be 0, heads can be higher)
    for param_group in optimizer.param_groups:
        if 'backbone' in param_group.get('name', ''):
            param_group['lr'] = 0.0  # Ensure backbone learning rate is 0
        else:
            # Optionally increase learning rate for heads during finetuning
            param_group['lr'] = param_group['lr'] * 2.0  # 2x learning rate for heads

    do_train(
        cfg,
        distributed,
        model,
        data_loader,
        data_loaders_val,
        optimizer,
        scheduler,
        warmup_scheduler,
        checkpointer,
        device,
        arguments,
    )

def setup(args):
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.DATALOADER.NUM_WORKERS = args.num_work
    cfg.TEST.EVAL_DIS_IOUS = args.eval_iou
    cfg.TEST.EVAL_DEPTH = args.eval_depth
    cfg.TEST.VIS = args.vis
    cfg.TEST.VIS_ALL = args.vis_all
    cfg.TEST.VIS_HORIZON = args.vis_horizon
    
    if args.vis_thre > 0:
        cfg.TEST.VISUALIZE_THRESHOLD = args.vis_thre 
    
    if args.output is not None:
        cfg.OUTPUT_DIR = args.output

    if args.test:
        cfg.DATASETS.TEST_SPLIT = 'test'
        cfg.DATASETS.TEST = ("kitti_test",)

    cfg.START_TIME = datetime.datetime.strftime(datetime.datetime.now(), '%m-%d %H:%M:%S')
    default_setup(cfg, args)

    return cfg

def main(args):
    cfg = setup(args)

    distributed = comm.get_world_size() > 1
    if not distributed: 
        cfg.MODEL.USE_SYNC_BN = False

    model = KeypointDetector(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    if args.eval_only:
        checkpointer = DetectronCheckpointer(
            cfg, model, save_dir=cfg.OUTPUT_DIR
        )
        ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
        _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)
        return run_test(cfg, checkpointer.model, vis=args.vis, eval_score_iou=args.eval_score_iou, eval_all_depths=args.eval_all_depths,
                        vis_all=args.vis_all)

    if distributed:
        # convert BN to SyncBN
        if cfg.MODEL.USE_SYNC_BN:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
            find_unused_parameters=True,
        )

    finetune(cfg, model, device, distributed)

if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    
    print("Command Line Args:", args)
    print("=" * 50)
    print("FINETUNING MODE: Backbone will be frozen")
    print("=" * 50)

    # backup all python files when training
    if not args.eval_only and args.output is not None:
        sync_root('.', os.path.join(args.output, 'backup'))
        import shutil
        shutil.copy2(args.config_file, os.path.join(args.output, 'backup', os.path.basename(args.config_file)))

        print("Finish backup all files")

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    ) 