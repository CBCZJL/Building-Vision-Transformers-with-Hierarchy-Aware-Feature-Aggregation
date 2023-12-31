# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import atexit
import builtins
import decimal
import functools
import logging
import os
import sys

import functools
import logging
import pickle
from torch import nn


import torch
import torch.distributed as dist

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_my_model(model):
    if isinstance(model, nn.DataParallel):
        return model.module
    return model

def is_master_process():
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def get_parallel_model(model, device):

    if get_world_size() >= 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], find_unused_parameters=True
        )
    else:
        raise NotImplementedError
    return model


def _suppress_print():
    """
    Suppresses printing from the current process.
    """
    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        pass

    builtins.print = print_pass


def setup_logging(save_path, mode='a'):
    """
    Sets up the logging for multiple processes. Only enable the logging for the
    master process, and suppress logging for the non-master processes.
    """
    if is_master_process():
        # Enable logging for the master process.
        logging.root.handlers = []
    else:
        # Suppress logging for non-master processes.
        _suppress_print()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    print_plain_formatter = logging.Formatter(
        "[%(asctime)s]: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )
    fh_plain_formatter = logging.Formatter("%(message)s")

    if is_master_process():
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(print_plain_formatter)
        logger.addHandler(ch)

    if save_path is not None and is_master_process():
        fh = logging.FileHandler(save_path, mode=mode)
        fh.setLevel(logging.INFO)
        fh.setFormatter(fh_plain_formatter)
        logger.addHandler(fh)


def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)



