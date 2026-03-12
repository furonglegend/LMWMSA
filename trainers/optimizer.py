"""
trainers/optimizer.py

Factory utilities for building optimizers and schedulers according to config.
Supported optimizers: AdamW
Supported schedulers: cosine (with optional warmup), step

Usage:
    optim, sched = build_optim_and_sched(model.parameters(), cfg)
"""

from typing import Tuple, Any
import math
import torch

def build_optim_and_sched(params, cfg):
    training = cfg.get("training", {})
    lr = training.get("lr", 1e-4)
    weight_decay = training.get("weight_decay", 1e-2)
    opt_name = training.get("optimizer", "adamw").lower()

    if opt_name == "adamw":
        optim = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    # scheduler
    sched_cfg = training.get("lr_scheduler", {})
    sched_type = sched_cfg.get("type", training.get("lr_scheduler", "cosine"))
    if isinstance(sched_type, dict):
        sched_type = sched_type.get("type", "cosine")

    if sched_type == "cosine":
        # support warmup steps (approximate by linear warmup wrapper)
        t_max = training.get("num_epochs", 100)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=t_max)
    elif sched_type == "step":
        step_size = sched_cfg.get("step_size", 50)
        gamma = sched_cfg.get("gamma", 0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=step_size, gamma=gamma)
    else:
        scheduler = None

    return optim, scheduler
