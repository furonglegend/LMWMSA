"""
utils/logger.py

Simple logging wrapper that supports TensorBoard and optional Weights & Biases (wandb).
Provides:
 - Logger class with add_scalar, add_image, add_text
 - automatic local logging to console and tensorboard directory

If wandb is installed and use_wandb=True in config, logger will try to log runs there.
"""

import os
import time
import socket
from typing import Optional, Any
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except Exception:
    _HAS_TB = False

try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False

class Logger:
    def __init__(self, log_dir: str = "./logs", use_wandb: bool = False, project: str = "ctrlvsr", name: Optional[str] = None):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.use_wandb = use_wandb and _HAS_WANDB
        self.name = name or f"{project}_{int(time.time())}_{socket.gethostname()}"
        if _HAS_TB:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None
            print("[LOGGER] tensorboard not available; install torch.utils.tensorboard for better logging.")

        if self.use_wandb:
            try:
                wandb.init(project=project, name=self.name, dir=log_dir)
            except Exception as e:
                self.use_wandb = False
                print(f"[LOGGER] wandb init failed: {e}")

    def add_scalar(self, tag: str, value: float, step: int):
        if self.writer:
            self.writer.add_scalar(tag, value, step)
        if self.use_wandb:
            wandb.log({tag: value}, step=step)
        # also print minimal console log
        if step % 100 == 0:
            print(f"[LOG][{step}] {tag}: {value:.6f}")

    def add_image(self, tag: str, img: Any, step: int):
        """
        img: either numpy array HWC or torch tensor CHW in [0,1]
        """
        if self.writer:
            try:
                self.writer.add_image(tag, img, step)
            except Exception:
                # attempt conversion
                import torchvision
                if isinstance(img, np.ndarray):
                    img_t = (torch.from_numpy(img).permute(2,0,1).float())
                else:
                    img_t = img
                self.writer.add_image(tag, img_t, step)
        if self.use_wandb:
            wandb.log({tag: wandb.Image(img)}, step=step)

    def add_text(self, tag: str, text: str, step: int):
        if self.writer:
            self.writer.add_text(tag, text, step)
        if self.use_wandb:
            wandb.log({tag: text}, step=step)

    def close(self):
        if self.writer:
            self.writer.close()
        if self.use_wandb:
            wandb.finish()
