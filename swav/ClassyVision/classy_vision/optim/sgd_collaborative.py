#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Callable

import torch.optim

from . import ClassyOptimizer, register_optimizer

import hivemind
import time, socket

import logging


import math
import warnings
import torch.nn as nn
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import _LRScheduler

class LinearWarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"] + (base_lr - self.eta_min) *
                (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))) /
            (
                1 +
                math.cos(math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs))
            ) * (group["lr"] - self.eta_min) + self.eta_min for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]

@register_optimizer("sgd_collaborative")
class SGDCollaborative(ClassyOptimizer):
    def __init__(
        self,
        larc_config: Dict[str, Any] = None,
        lr: float = 0,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        use_larc: bool = False,
        dht_initial_peers: List[str] = [],
        batch_size_for_tracking: int = 200,
        dht_listen_on_port: str = "1337",
        averager_listen_on_port: str = "1338",
        exp_prefix: str = "test_exp",
        target_group_size: int = 5,
        max_allowed_epoch_difference: int = 1,
        total_steps_in_epoch: int = 6405,
        scheduler_cls = None,
        averaging_steps_period: int = 5,
        averaging_time_period: float = 0,
        report_progress_expiration: int = 60,
        timeout: float = None
    ):
        super().__init__()

        self._lr = lr
        self._momentum = momentum
        self._weight_decay = weight_decay
        self._nesterov = nesterov
        self._use_larc = use_larc
        self._larc_config = larc_config
        self._exp_prefix = exp_prefix
        self._listen_on_dht = f"[::]:{dht_listen_on_port}"
        self._listen_on_averager = f"[::]:{averager_listen_on_port}"
        self._dht_initial_peers = dht_initial_peers
        self._target_group_size = target_group_size
        self._batch_size_for_tracking = batch_size_for_tracking
        self._max_allowed_epoch_difference = max_allowed_epoch_difference
        self._total_steps_in_epoch = total_steps_in_epoch
        self._scheduler_cls = scheduler_cls
        self._averaging_steps_period = averaging_steps_period
        self._averaging_time_period = averaging_time_period
        self._report_progress_expiration = report_progress_expiration
        self._timeout = timeout

    def prepare(self, param_groups):
        for group in param_groups:
            group['lr'] = self._lr
        self.optimizer = torch.optim.SGD(param_groups, lr=self._lr, nesterov=self._nesterov, momentum=self._momentum,
                                         weight_decay=self._weight_decay,)
        assert self._use_larc, "we can't use collab sgd without larc"
        try:
            from apex.parallel.LARC import LARC
        except ImportError:
            raise RuntimeError("Apex needed for LARC")

        class BLYARC(LARC, torch.optim.Optimizer): pass
        self.optimizer = BLYARC(optimizer=self.optimizer, **self._larc_config)
        self.optimizer = hivemind.CollaborativeOptimizer(
            opt=self.optimizer, scheduler=self._scheduler_cls(self.optimizer),
            dht=hivemind.DHT(listen_on=self._listen_on_dht,
                             initial_peers=self._dht_initial_peers,
                             start=True, listen=True),
            verbose=True, compression=hivemind.utils.CompressionType.Value('FLOAT16'),
            prefix=self._exp_prefix,
            target_group_size=self._target_group_size,
            target_batch_size=32768, batch_size_per_step=self._batch_size_for_tracking,
            listen_on=self._listen_on_averager,
            averaging_expiration=5.0, metadata_expiration=30, averaging_timeout=30,
            start=True
        )

    def get_classy_state(self) -> Dict[str, Any]:
        return

    def set_classy_state(self, state: Dict[str, Any]) -> None:
        return

    def on_epoch(self, where: float) -> None:
        return

    def step(
        self, *args, closure: Optional[Callable] = None, where: float = None
    ) -> None:
        self.optimizer.step()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SGD":
        """Instantiates a SGD from a configuration.

        Args:
            config: A configuration for a SGD.
                See :func:`__init__` for parameters expected in the config.

        Returns:
          A SGD instance.
        """
        # Default params
        config.setdefault("lr", 0)
        config.setdefault("momentum", 0.0)
        config.setdefault("weight_decay", 0.0)
        config.setdefault("nesterov", False)
        config.setdefault("use_larc", False)
        config.setdefault(
            "larc_config", {"clip": True, "eps": 1e-08, "trust_coefficient": 0.02}
        )

        assert (
            config["momentum"] >= 0.0
            and config["momentum"] < 1.0
            and type(config["momentum"]) == float
        ), "Config must contain a 'momentum' in [0, 1) for SGD optimizer"
        assert isinstance(
            config["nesterov"], bool
        ), "Config must contain a boolean 'nesterov' param for SGD optimizer"
        assert isinstance(
            config["use_larc"], bool
        ), "Config must contain a boolean 'use_larc' param for SGD optimizer"

        config.setdefault("dht_initial_peers", [])
        config.setdefault("batch_size_for_tracking", 200)
        config.setdefault("exp_prefix", "test_exp")
        config.setdefault("dht_listen_on_port", "1337")
        config.setdefault("averager_listen_on_port", "1338")
        config.setdefault("target_group_size", 5)
        config.setdefault("max_allowed_epoch_difference", 1)
        config.setdefault("total_steps_in_epoch", 6405)
        config.setdefault("averaging_steps_period", 1)
        config.setdefault("averaging_time_period", 0)
        config.setdefault("report_progress_expiration", 60)
        config.setdefault("timeout", None)
        config.setdefault("warmup_epochs", 0)
        config.setdefault("max_epochs", 200)
        config.setdefault("warmup_start_lr", 0)
        config.setdefault("eta_min", 0.006 * 200/256)

        logging.info(f"creating SGDCollaborative with config: {config}")

        def scheduler_cls(optim):
            return LinearWarmupCosineAnnealingLR(optim, warmup_epochs=config["warmup_epochs"], max_epochs=config["max_epochs"], warmup_start_lr=config["warmup_start_lr"], eta_min=config["eta_min"])
        config.setdefault("scheduler_cls", scheduler_cls)

        return cls(
            larc_config=config["larc_config"],
            lr=config["lr"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
            nesterov=config["nesterov"],
            use_larc=config["use_larc"],
            dht_initial_peers=config["dht_initial_peers"],
            dht_listen_on_port=config["dht_listen_on_port"],
            averager_listen_on_port=config["averager_listen_on_port"],
            exp_prefix=config["exp_prefix"],
            target_group_size=config["target_group_size"],
            batch_size_for_tracking=config["batch_size_for_tracking"],
            max_allowed_epoch_difference=config["max_allowed_epoch_difference"],
            total_steps_in_epoch=config["total_steps_in_epoch"],
            scheduler_cls=config["scheduler_cls"],
            averaging_steps_period=config["averaging_steps_period"],
            averaging_time_period=config["averaging_time_period"],
            report_progress_expiration=config["report_progress_expiration"],
            timeout=config["timeout"]
        )
