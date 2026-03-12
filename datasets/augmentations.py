"""
datasets/augmentations.py

Lightweight augmentation utilities for CtrlVSR training pipeline.
All functions return torchvision-style callables that accept PIL images and return transformed tensors.

Provided transforms:
 - random_degradation: apply mild camera/video degradations (blur, noise, JPEG)
 - paired_crop: random crop that crops HR and produces LR by downscaling
 - color_jitter_wrapper: wrapper around torchvision ColorJitter
"""

import random
from typing import Tuple, Callable, Optional

from PIL import Image, ImageFilter
import numpy as np
import torchvision.transforms.functional as TF
from torchvision import transforms as T


def random_blur(img: Image.Image, max_radius: float = 2.0) -> Image.Image:
    """Apply a random Gaussian blur to a PIL Image."""
    radius = random.random() * max_radius
    if radius <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def random_jpeg_compress(img: Image.Image, min_quality: int = 60, max_quality: int = 95) -> Image.Image:
    """
    Simulate JPEG compression by re-encoding the PIL image to JPEG buffer
    and reloading it at a random quality level.
    """
    from io import BytesIO
    q = random.randint(min_quality, max_quality)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def add_gaussian_noise_tensor(tensor, std=0.01):
    """Add Gaussian noise to a torch tensor in [0,1]."""
    import torch
    noise = torch.randn_like(tensor) * std
    return torch.clamp(tensor + noise, 0.0, 1.0)


def paired_random_crop(hr: Image.Image, hr_crop: int, lr_scale: int) -> Tuple[Image.Image, Image.Image]:
    """
    Randomly crop an HR image and produce a corresponding LR crop by downsampling.
    Returns (lr_pil, hr_pil).
    """
    w, h = hr.size
    if w < hr_crop or h < hr_crop:
        # fallback to center crop when image smaller than requested crop
        hr_cropped = TF.center_crop(hr, (hr_crop, hr_crop))
    else:
        left = random.randint(0, w - hr_crop)
        top = random.randint(0, h - hr_crop)
        hr_cropped = hr.crop((left, top, left + hr_crop, top + hr_crop))

    lr_size = hr_crop // lr_scale
    lr_resized = hr_cropped.resize((lr_size, lr_size), resample=Image.BICUBIC)
    return lr_resized, hr_cropped


def random_degradation_pipeline(hr_crop: int = 256, lr_scale: int = 4):
    """
    Return a callable that accepts a PIL image (HR) and returns (lr_tensor, hr_tensor).
    This pipeline performs:
      - paired random crop
      - random blur
      - random JPEG compression
      - color jitter (small)
      - conversion to tensor
      - optional gaussian noise addition
    """
    color_jitter = T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02)

    def _pipeline(hr_img: Image.Image):
        lr_pil, hr_pil = paired_random_crop(hr_img, hr_crop, lr_scale)
        # stochastic degradations on HR -> used to create degraded LR
        if random.random() < 0.5:
            lr_pil = random_blur(lr_pil, max_radius=1.5)
        if random.random() < 0.3:
            lr_pil = random_jpeg_compress(lr_pil, 60, 95)
        # color jitter on both to keep distribution similar (mild)
        hr_pil = color_jitter(hr_pil)
        lr_pil = color_jitter(lr_pil)
        # convert to tensors
        hr_tensor = TF.to_tensor(hr_pil)
        lr_tensor = TF.to_tensor(lr_pil)
        # add small gaussian noise sometimes
        if random.random() < 0.2:
            lr_tensor = add_gaussian_noise_tensor(lr_tensor, std=0.01)
        return lr_tensor, hr_tensor

    return _pipeline


def simple_lr_transform(lr_size: int = 64):
    """Return a transform for LR-only datasets (no HR available)."""
    return T.Compose([
        T.RandomCrop((lr_size, lr_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor()
    ])


if __name__ == "__main__":
    # simple demo of pipeline
    from PIL import Image
    img = Image.new("RGB", (512, 512), color=(128, 128, 128))
    pipeline = random_degradation_pipeline(hr_crop=256, lr_scale=4)
    lr_t, hr_t = pipeline(img)
    print("LR shape:", lr_t.shape, "HR shape:", hr_t.shape)
