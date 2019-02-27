# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import numpy as np
import os
import random
import torch
from utils import utils
from os.path import expanduser, join


from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir


def train(cfg, local_rank, distributed, seed):
    """Start training.

    Parameters
    ----------
        cfg : CfgNode
            Configuration object for the experiment.
        local_rank : int
            CUDA parameter.
        distributed : bool
            Boolean flag for distributed training.
        seed : int
            The seed initialization for random methods.

    Returns
    -------
    None

    """
    model = build_detection_model(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = join(cfg.OUTPUT_DIR, str(seed))

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"]
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
    )

    return model


def run(seed):
    """Setup parameters for training.

    Parameters
    ----------
        seed : int
            The seed initialization for random methods.

    Returns
    -------
    None

    """

    home = expanduser("~")
    parser = argparse.ArgumentParser(
        description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default=os.path.join(
            home, "inequity-release/maskrcnn-benchmark/configs/bdd100k/e2e_faster_rcnn_R_50_FPN_1x_cocostyle.yaml"),
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
        default=True
    )
    parser.add_argument(
        "--augmented_loss_weights",
        default=[1, 1, 1, 1],
        help="Weighted training",
    )
    parser.add_argument(
        "--augmented_rpn_loss_weights",
        default=[1, 1, 1, 1],
        help="Weighted rpn training",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    if type(args.augmented_loss_weights) != list:
        args.augmented_loss_weights = args.augmented_loss_weights \
            .replace("[", "") \
            .replace("]", "") \
            .split(",")
        args.augmented_loss_weights = [int(x) for x
                                       in args.augmented_loss_weights]

    if type(args.augmented_rpn_loss_weights) != list:
        args.augmented_rpn_loss_weights = args.augmented_rpn_loss_weights \
            .replace("[", "") \
            .replace("]", "") \
            .split(",")
        args.augmented_rpn_loss_weights = [int(x) for x
                                           in args.augmented_rpn_loss_weights]

    args.opts += ["MODEL.ROI_BOX_HEAD.AUGMENTED_LOSS_WEIGHTS",
                  args.augmented_loss_weights]
    args.opts += ["MODEL.RPN.AUGMENTED_LOSS_WEIGHTS",
                  args.augmented_rpn_loss_weights]

    num_gpus = int(os.environ["WORLD_SIZE"]
                   ) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = join(cfg.OUTPUT_DIR, str(seed))
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using seed: {}".format(seed))
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = train(cfg, args.local_rank, args.distributed, seed)


def set_seeds(seed):
    """Set the seeds in an attempt
    to improve reproducibility.
    Due to some inherent nondeterministic behavior:
    https://github.com/facebookresearch/maskrcnn-benchmark/issues/376#issuecomment-456803015
    this will only mitigate some of the randomness, and
    results can and will likely differ between trials

    Parameters
    ----------
        seed : int
            The seed initialization for random methods.

    Returns
    -------
    None

    """

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)


def main():
    for seed in np.arange(0, 10):
        set_seeds(seed)
        run(seed)


if __name__ == "__main__":
    main()
