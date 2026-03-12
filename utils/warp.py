"""
utils/warp.py

Differentiable warping utilities used by CtrlVSR.
Provides:
 - warp_image: warp image / tensor with optical flow using bilinear sampling
 - warp_latent: warp latent tokens if they are spatially arranged (B, N, C)

The flow convention:
 - flow: tensor of shape (B, 2, H, W) with flow vectors (dx, dy) in pixels indicating
         source coordinate offsets when sampling target -> source (standard).
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def meshgrid(height: int, width: int, device: torch.device):
    ys = torch.linspace(-1.0, 1.0, steps=height, device=device)
    xs = torch.linspace(-1.0, 1.0, steps=width, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return grid_x, grid_y


def flow_to_sampling_grid(flow: torch.Tensor) -> torch.Tensor:
    """
    Convert flow in pixel units (B, 2, H, W) to a sampling grid in normalized coords (B, H, W, 2)
    suitable for grid_sample. Here we assume flow is dx,dy in pixels relative to image coordinates.
    The caller must provide flow in same resolution as target image.
    """
    B, _, H, W = flow.shape
    device = flow.device
    # convert pixel offsets dx,dy to normalized [-1,1] offsets
    dx = flow[:, 0:1, :, :]  # (B,1,H,W)
    dy = flow[:, 1:2, :, :]
    # normalized scale factors
    nx = dx / ((W - 1) / 2.0)
    ny = dy / ((H - 1) / 2.0)
    # base grid
    xs = torch.linspace(-1.0, 1.0, W, device=device)
    ys = torch.linspace(-1.0, 1.0, H, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_x = grid_x.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, 1)
    grid_y = grid_y.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, 1)
    # apply offsets
    grid = torch.cat([grid_x + nx.permute(0,2,3,1), grid_y + ny.permute(0,2,3,1)], dim=-1)  # (B,H,W,2) (x,y)
    return grid


def warp_image(img: torch.Tensor, flow: torch.Tensor, mode: str = "bilinear", padding_mode: str = "border") -> torch.Tensor:
    """
    Warp an image tensor by flow. Uses torch.grid_sample (differentiable).

    Args:
      img: (B, C, H, W)
      flow: (B, 2, H, W) with (dx, dy) in pixels representing target->source offsets
      mode: interpolation mode
      padding_mode: padding for out-of-bound sampling

    Returns:
      warped image tensor (B, C, H, W)
    """
    B, C, H, W = img.shape
    grid = flow_to_sampling_grid(flow)  # (B,H,W,2) normalized coords
    return F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)


def warp_latent(z: torch.Tensor, flow: torch.Tensor, side: Optional[int] = None) -> torch.Tensor:
    """
    Warp latent tokens arranged in a spatial grid.

    Args:
      z: (B, N, C) where N = H*W tokens (square)
      flow: (B, 2, H, W) in pixels
      side: optional side length (sqrt(N)). If None will be inferred.

    Returns:
      warped tokens of shape (B, N, C)
    """
    B, N, C = z.shape
    if side is None:
        side = int(N ** 0.5)
    assert side * side == N, "Tokens must arrange into square grid"
    z_map = z.transpose(1, 2).view(B, C, side, side)  # (B,C,H,W)
    warped = warp_image(z_map, flow)
    return warped.view(B, C, -1).transpose(1, 2)  # (B,N,C)


if __name__ == "__main__":
    import torch
    img = torch.rand(1, 3, 64, 64)
    flow = torch.zeros(1,2,64,64)
    flow[:,0,:,:] = 1.0  # shift right by 1 pixel
    w = warp_image(img, flow)
    print("Warped shape:", w.shape)
