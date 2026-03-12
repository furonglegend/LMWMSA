"""
datasets/loader_video.py

A more feature-complete Video dataset loader for CtrlVSR.
Returns a dict with LR frame stack, optional HR center frame, and external/internal motion placeholders.
Designed to plug into the training pipeline in place of the simpler dataset earlier.

Key features:
 - supports variable-length videos (clamps edges)
 - supports loading precomputed flows via a FlowCache
 - supports on-the-fly LR downscaling (if HR exists but LR not stored)
 - returns tensors in shape (T, C, H, W) for frames and (T-1, H, W, 2) for flows (forward flows)
"""

import os
from typing import List, Optional, Callable, Dict
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

# Utility loader for images
def load_image_as_tensor(path: str, transform: Optional[Callable] = None) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if transform is not None:
        return transform(img)
    # default: to tensor (C,H,W) in [0,1]
    return T.ToTensor()(img)


class VideoLoaderDataset(Dataset):
    """
    VideoLoaderDataset

    Args:
        root: root directory containing videos (each video is a folder of frames)
        videos: list of video folder names to include
        window_radius: number of frames to each side of center (tau)
        lr_scale: downscale factor from HR to LR if needed
        lr_transform, hr_transform: optional torchvision transforms
        flow_cache: optional FlowCache providing precomputed flows via get_flow(video_id, idx)
        return_hr: whether to attempt to load HR frame (default True)
        pad_mode: 'edge' or 'reflect' for index clamping near boundaries
    """

    def __init__(
        self,
        root: str,
        videos: List[str],
        window_radius: int = 2,
        lr_scale: int = 4,
        lr_transform: Optional[Callable] = None,
        hr_transform: Optional[Callable] = None,
        flow_cache: Optional[object] = None,
        return_hr: bool = True,
        pad_mode: str = "edge",
    ):
        self.root = root
        self.videos = videos
        self.window_radius = window_radius
        self.window_length = 2 * window_radius + 1
        self.lr_scale = lr_scale
        self.flow_cache = flow_cache
        self.return_hr = return_hr
        self.pad_mode = pad_mode

        self.lr_transform = lr_transform or T.Compose([T.ToTensor()])
        self.hr_transform = hr_transform or T.Compose([T.ToTensor()])

        # build per-video frame lists
        self.video_frames = {}
        for vid in videos:
            folder = os.path.join(root, vid)
            if not os.path.isdir(folder):
                continue
            frames = sorted([f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
            if len(frames) == 0:
                continue
            self.video_frames[vid] = frames

        # build global index: (video_id, center_index)
        self.index = []
        for vid, frames in self.video_frames.items():
            for i in range(len(frames)):
                self.index.append((vid, i))

    def __len__(self):
        return len(self.index)

    def _frame_path(self, vid: str, frame_name: str, hr: bool = False) -> str:
        """
        Resolve frame path. If HR frames are stored under 'HR' subfolder within video folder,
        this method checks that first.
        """
        base = os.path.join(self.root, vid)
        if hr:
            hr_dir = os.path.join(base, "HR")
            if os.path.isdir(hr_dir):
                cand = os.path.join(hr_dir, frame_name)
                if os.path.exists(cand):
                    return cand
        # fallback to base folder
        return os.path.join(base, frame_name)

    def _clamp_index(self, idx: int, n: int) -> int:
        if self.pad_mode == "edge":
            return max(0, min(idx, n - 1))
        elif self.pad_mode == "reflect":
            # simple reflect padding
            if idx < 0:
                return -idx
            if idx >= n:
                return n - (idx - n + 2)
            return idx
        else:
            return max(0, min(idx, n - 1))

    def __getitem__(self, idx: int) -> Dict:
        vid, center = self.index[idx]
        frames = self.video_frames[vid]
        n = len(frames)

        # collect indices for window
        indices = [self._clamp_index(center + offset, n) for offset in range(-self.window_radius, self.window_radius + 1)]

        lr_tensors = []
        hr_tensor = None

        for i in indices:
            frame_name = frames[i]
            # load HR if available and downscale to LR if necessary
            hr_path = self._frame_path(vid, frame_name, hr=True)
            if os.path.exists(hr_path):
                hr_img = load_image_as_tensor(hr_path, transform=self.hr_transform)
                # optionally create LR by downsampling
                lr_img = T.functional.resize(hr_img, [hr_img.shape[1] // self.lr_scale, hr_img.shape[2] // self.lr_scale])
            else:
                # fall back to base path (maybe only LR exists)
                base_path = self._frame_path(vid, frame_name, hr=False)
                lr_img = load_image_as_tensor(base_path, transform=self.lr_transform)
                hr_img = None

            lr_tensors.append(lr_img)
            if i == center:
                hr_tensor = hr_img

        lr_stack = torch.stack(lr_tensors, dim=0)  # (T, C, H, W)

        # gather flows for the forward directions in the window (i -> i+1)
        flows = None
        if self.flow_cache is not None:
            flows = []
            for i in range(indices[0], indices[-1]):
                f = self.flow_cache.get_flow(vid, i)
                # if missing, append None to keep alignment length T-1
                flows.append(f)
            # flows is list length T-1, elements are arrays or None
        sample = {
            "lr_frames": lr_stack,
            "hr_frame": hr_tensor,
            "flows": flows,
            "meta": {"video_id": vid, "center_idx": center, "window_indices": indices}
        }
        return sample


if __name__ == "__main__":
    # simple dry-run (replace paths with real ones to test)
    ds = VideoLoaderDataset("/path/to/videos", videos=["vid1"], window_radius=1, return_hr=False)
    print("Length:", len(ds))
    if len(ds) > 0:
        s = ds[0]
        print("LR frames shape:", s["lr_frames"].shape)
        print("Flows:", s["flows"])
