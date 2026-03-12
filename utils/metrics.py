"""
utils/metrics.py

Common image & video quality metrics used for evaluation:
 - PSNR
 - simple SSIM (fallback implementation if skimage unavailable)
 - LPIPS wrapper (if lpips installed)

Note: For best-quality SSIM/LPIPS, install scikit-image and lpips packages.
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional

# Attempt optional imports
try:
    import lpips
    _HAS_LPIPS = True
except Exception:
    _HAS_LPIPS = False

try:
    from skimage.metrics import structural_similarity as sk_ssim
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

def psnr(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0) -> float:
    """
    Compute PSNR between two images (torch tensors in [0,1]): returns PSNR in dB.
    img1/img2 expected shape (C,H,W) or (B,C,H,W)
    """
    if img1.dim() == 3:
        mse = F.mse_loss(img1, img2)
        if mse == 0:
            return float('inf')
        return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse.item())
    else:
        mse = F.mse_loss(img1, img2, reduction='mean')
        if mse == 0:
            return float('inf')
        return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse.item())

def ssim(img1: torch.Tensor, img2: torch.Tensor, win_size: int = 11) -> float:
    """
    Compute SSIM. If skimage available, use its implementation (per-channel averaged).
    Otherwise, use a simple PyTorch-based approximation (luminance-only).
    Inputs are torch tensors in [0,1] shape (C,H,W) or (B,C,H,W).
    """
    if _HAS_SKIMAGE:
        import numpy as np
        if img1.dim() == 4:
            img1_np = img1.permute(0,2,3,1).cpu().numpy()
            img2_np = img2.permute(0,2,3,1).cpu().numpy()
            vals = []
            for a,b in zip(img1_np, img2_np):
                vals.append(sk_ssim(a, b, multichannel=True, data_range=1.0))
            return float(sum(vals) / len(vals))
        else:
            a = img1.permute(1,2,0).cpu().numpy()
            b = img2.permute(1,2,0).cpu().numpy()
            return float(sk_ssim(a, b, multichannel=True, data_range=1.0))
    else:
        # fallback: compute SSIM on grayscale via simple stats (not a perfect substitute)
        def to_gray(x):
            if x.shape[0] == 3:
                r, g, b = x[0], x[1], x[2]
                return 0.2989 * r + 0.5870 * g + 0.1140 * b
            return x.mean(dim=0)
        if img1.dim() == 4:
            vals = []
            for i in range(img1.shape[0]):
                g1 = to_gray(img1[i])
                g2 = to_gray(img2[i])
                mu1 = g1.mean()
                mu2 = g2.mean()
                sigma1 = ((g1 - mu1) ** 2).mean()
                sigma2 = ((g2 - mu2) ** 2).mean()
                cov = ((g1 - mu1) * (g2 - mu2)).mean()
                C1 = (0.01 ** 2)
                C2 = (0.03 ** 2)
                ssim_val = ((2*mu1*mu2 + C1) * (2*cov + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2))
                vals.append(float(ssim_val))
            return sum(vals)/len(vals)
        else:
            g1 = to_gray(img1)
            g2 = to_gray(img2)
            mu1 = g1.mean()
            mu2 = g2.mean()
            sigma1 = ((g1 - mu1) ** 2).mean()
            sigma2 = ((g2 - mu2) ** 2).mean()
            cov = ((g1 - mu1) * (g2 - mu2)).mean()
            C1 = (0.01 ** 2)
            C2 = (0.03 ** 2)
            ssim_val = ((2*mu1*mu2 + C1) * (2*cov + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2))
            return float(ssim_val)

class LPIPSWrapper:
    """
    Thin wrapper around lpips.PatchL2 if available.
    """
    def __init__(self, net='vgg'):
        if not _HAS_LPIPS:
            raise RuntimeError("lpips not installed. Install via `pip install lpips` for perceptual metrics.")
        self.loss_fn = lpips.LPIPS(net=net)

    def __call__(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Accepts images in [-1,1] range or [0,1]; lpips expects [-1,1].
        """
        def to_lpips(x):
            if x.min() >= 0 and x.max() <= 1:
                return x * 2.0 - 1.0
            return x
        a = to_lpips(img1)
        b = to_lpips(img2)
        with torch.no_grad():
            val = self.loss_fn(a, b)
        return float(val.mean().item())


if __name__ == "__main__":
    import torch
    a = torch.rand(1,3,64,64)
    b = a * 0.9 + 0.05
    print("PSNR:", psnr(a,b))
    print("SSIM (approx):", ssim(a,b))
