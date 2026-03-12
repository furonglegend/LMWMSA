"""
trainers/checkpoint.py

Small utilities to save/load model checkpoints robustly, and to resume training.
"""

import os
import torch
from typing import Dict, Any, Optional

def save_checkpoint(state: Dict[str, Any], path: str, keep_latest: bool = True):
    """
    Atomically save checkpoint to disk.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)
    if keep_latest:
        latest = os.path.join(os.path.dirname(path), "latest.pth")
        try:
            torch.save(state, latest)
        except Exception:
            pass

def load_checkpoint(path: str, device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Load checkpoint and return its dict.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    map_loc = device if device is not None else "cpu"
    ck = torch.load(path, map_location=map_loc)
    return ck

def resume_from_checkpoint(model, optimizer, ckpt_path: str, device: Optional[torch.device] = None):
    """
    Load model and optimizer state dicts if present in checkpoint.
    Returns loaded_epoch (or 0) and loaded_step (or 0).
    """
    ck = load_checkpoint(ckpt_path, device=device)
    if "model_state" in ck:
        model.load_state_dict(ck["model_state"])
    elif "state_dict" in ck:
        model.load_state_dict(ck["state_dict"])

    if optimizer is not None and "optim_state" in ck:
        try:
            optimizer.load_state_dict(ck["optim_state"])
        except Exception as e:
            print(f"[CKPT] Warning: failed to load optimizer state: {e}")

    epoch = ck.get("epoch", 0)
    step = ck.get("step", 0)
    return epoch, step
