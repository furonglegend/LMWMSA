"""
data/datasets.py

PyTorch Dataset for CtrlVSR. This module provides a flexible VideoFrameDataset
that yields low-resolution frames, optional high-resolution frames and external flow
inputs for a sliding temporal window around a target index.

All comments are in English. Implementation focuses on clarity and extension points.
"""

import os
from typing import List, Optional, Tuple, Dict, Callable
from PIL import Image
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

# helper: safe image loader
def pil_load(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


class VideoFrameDataset(Dataset):
    """
    VideoFrameDataset yields tuples for training CtrlVSR.
    Each sample returns:
      - lr_frames: Tensor of shape (W, C, H, W) or (T, C, H, W) depending on stacking convention
      - hr_frame: Optional Tensor for supervised training
      - flows: Optional dict of external flows for the window (user supplies FlowCache)
      - meta: dict with video id and frame index

    Args:
        videos_root: root folder containing per-video folders (each folder contains frames named 000000.png ...)
        video_list: list of video folder names to include
        window_radius: tau in the paper (number of frames to each side)
        lr_transform: torchvision-like transform to apply to LR images
        hr_transform: transform for HR images (if present)
        flow_cache: optional FlowCache instance providing precomputed external flows
        return_hr: whether to return HR ground truth frames (may be False for unpaired data)
    """

    def __init__(
        self,
        videos_root: str,
        video_list: List[str],
        window_radius: int = 2,
        lr_scale: int = 4,
        lr_transform: Optional[Callable] = None,
        hr_transform: Optional[Callable] = None,
        flow_cache: Optional[object] = None,
        return_hr: bool = True,
    ):
        self.videos_root = videos_root
        self.video_list = video_list
        self.window_radius = window_radius
        self.window_length = 2 * window_radius + 1
        self.lr_scale = lr_scale
        self.flow_cache = flow_cache
        self.return_hr = return_hr

        # default transforms: convert to tensor and normalize to [0,1]
        self.lr_transform = lr_transform or T.Compose([
            T.ToTensor()
        ])
        self.hr_transform = hr_transform or T.Compose([
            T.ToTensor()
        ])

        # build index: list of (video_id, frame_idx)
        self.index = []
        for vid in video_list:
            folder = os.path.join(videos_root, vid)
            if not os.path.isdir(folder):
                continue
            frames = sorted([f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
            # ensure we have at least window_length frames
            if len(frames) < self.window_length:
                continue
            for i in range(len(frames)):
                # we allow samples near edges but will clamp indices when loading
                self.index.append((vid, i, frames))

    def __len__(self):
        return len(self.index)

    def _get_frame_path(self, vid: str, frame_name: str, hr: bool = False) -> str:
        """
        Return path to frame. If HR frames are stored in 'HR' subfolder, adapt here.
        This method can be customized for different dataset layouts.
        """
        folder = os.path.join(self.videos_root, vid)
        if hr:
            # common layout: videos_root/vid/HR/000000.png
            hr_folder = os.path.join(folder, "HR")
            if os.path.isdir(hr_folder):
                return os.path.join(hr_folder, frame_name)
        return os.path.join(folder, frame_name)

    def _clamp_idx(self, idx: int, max_len: int) -> int:
        return max(0, min(idx, max_len - 1))

    def __getitem__(self, idx: int) -> Dict:
        vid, center_idx, frames_list = self.index[idx]
        num_frames = len(frames_list)

        # compute window indices and clamp at edges
        indices = [self._clamp_idx(center_idx + offset, num_frames) for offset in range(-self.window_radius, self.window_radius + 1)]
        lr_imgs = []
        hr_img = None

        for i in indices:
            frame_name = frames_list[i]
            lr_path = self._get_frame_path(vid, frame_name, hr=False)
            lr_img = pil_load(lr_path)
            lr_imgs.append(self.lr_transform(lr_img))

        if self.return_hr:
            # center HR
            hr_path = self._get_frame_path(vid, frames_list[center_idx], hr=True)
            if os.path.exists(hr_path):
                hr_img = pil_load(hr_path)
                hr_img = self.hr_transform(hr_img)
            else:
                hr_img = None

        # optionally load external flows from flow_cache
        flows = None
        if self.flow_cache is not None:
            # flow_cache is expected to provide a get_flow(video_id, frame_index) method
            try:
                flows = self.flow_cache.get_flows_for_window(vid, center_idx, self.window_radius)
            except Exception:
                flows = None

        sample = {
            "lr_frames": torch.stack(lr_imgs, dim=0),  # (T, C, H, W)
            "hr_frame": hr_img,                         # Tensor or None
            "flows": flows,
            "meta": {"video_id": vid, "center_idx": center_idx}
        }

        return sample


# Small utility to parse a list file (text file with one video folder per line)
def load_video_list(list_file: str) -> List[str]:
    with open(list_file, "r") as fh:
        lines = [ln.strip() for ln in fh.readlines() if ln.strip()]
    return lines


# Example augmentation factory (users can override)
def make_transforms(lr_crop: int = 64, hr_crop: int = 256, random_flip: bool = True):
    """
    Create simple LR/HR transforms. This is intentionally minimal; plug in
    stronger augmentations as needed (kornia, albumentations, etc).
    """
    lr_transform = T.Compose([
        T.RandomCrop((lr_crop, lr_crop)),
        T.RandomHorizontalFlip(p=0.5 if random_flip else 0.0),
        T.ToTensor()
    ])
    hr_transform = T.Compose([
        T.RandomCrop((hr_crop, hr_crop)),
        T.RandomHorizontalFlip(p=0.5 if random_flip else 0.0),
        T.ToTensor()
    ])
    return lr_transform, hr_transform


if __name__ == "__main__":
    # quick sanity check (not exhaustive)
    dataset_root = "/tmp/dataset_example"
    video_list = ["video_000"]  # replace with real
    ds = VideoFrameDataset(dataset_root, video_list, window_radius=1, flow_cache=None, return_hr=False)
    print("Dataset length:", len(ds))
    if len(ds) > 0:
        sample = ds[0]
        print("Sample keys:", sample.keys())
        print("LR frames shape:", sample["lr_frames"].shape)
