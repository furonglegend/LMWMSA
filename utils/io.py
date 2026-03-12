"""
utils/io.py

I/O helpers for images, frames and simple checkpoint listing utilities:
 - read_image / save_image (PIL wrapper)
 - save_video (imageio-backed if available, else saves frames)
 - list_checkpoints: helper to list saved checkpoints in a directory

This module avoids hard dependency on imageio; if available it will use it.
"""

import os
from typing import Union, List
from PIL import Image
import numpy as np

try:
    import imageio
    _HAS_IMAGEIO = True
except Exception:
    _HAS_IMAGEIO = False

def read_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_image(img_tensor, path: str):
    """
    Save a torch tensor (C,H,W) or numpy array (H,W,C) to path.
    Values expected in [0,1] for tensors.
    """
    if hasattr(img_tensor, "cpu"):
        import torch
        if img_tensor.dim() == 3:
            img_np = img_tensor.detach().cpu().permute(1,2,0).numpy()
        elif img_tensor.dim() == 4:
            # save first in batch
            img_np = img_tensor[0].detach().cpu().permute(1,2,0).numpy()
        else:
            raise ValueError("Unsupported tensor shape for save_image")
    else:
        img_np = img_tensor
    img_np = (np.clip(img_np, 0.0, 1.0) * 255.0).astype(np.uint8)
    pil = Image.fromarray(img_np)
    pil.save(path)


def save_frames_as_video(frame_paths: List[str], out_path: str, fps: int = 24):
    """
    Save a sequence of image paths into a video. If imageio not available, fallback to saving PNGs only.
    """
    if _HAS_IMAGEIO:
        writer = imageio.get_writer(out_path, fps=fps)
        for p in frame_paths:
            img = imageio.imread(p)
            writer.append_data(img)
        writer.close()
    else:
        # fallback: copy frames to out_path folder if .mp4 not supported
        if out_path.lower().endswith((".mp4", ".mov", ".avi")):
            folder = os.path.splitext(out_path)[0] + "_frames"
            os.makedirs(folder, exist_ok=True)
            for p in frame_paths:
                import shutil
                shutil.copy(p, os.path.join(folder, os.path.basename(p)))
            print(f"[IO] imageio not installed; copied frames to {folder} instead of writing video.")
        else:
            # if out_path is directory, ensure exists
            os.makedirs(out_path, exist_ok=True)
            for p in frame_paths:
                import shutil
                shutil.copy(p, os.path.join(out_path, os.path.basename(p)))


def list_checkpoints(ckpt_dir: str, pattern: str = ".pth") -> List[str]:
    """
    Return list of checkpoint files sorted by modification time (newest first).
    """
    import glob
    files = glob.glob(os.path.join(ckpt_dir, f"*{pattern}"))
    files_sorted = sorted(files, key=lambda p: os.path.getmtime(p), reverse=True)
    return files_sorted
