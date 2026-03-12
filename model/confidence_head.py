"""
models/motion/confidence_head.py

Computes a compact confidence map c_t in [0,1] from photometric inconsistency maps and lightweight features.

Design notes:
 - The photometric inconsistency e_t is expected to be a per-pixel map (B, 1, H, W)
 - We perform coarse pooling (spatial downsample) then a small conv head and sigmoid to produce map
 - The head returns a tensor (B, 1, h', w') where h',w' are pooled spatial dims (coarse confidence),
   plus an upsampled version optionally.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceHead(nn.Module):
    def __init__(self, in_feat_channels: int = 16, pool_stride: int = 8, out_channels: int = 1):
        """
        Args:
            in_feat_channels: number of channels in auxiliary feature map (feat_t)
            pool_stride: coarse pooling stride (reduces resolution to save compute)
            out_channels: output channels (1 -> scalar confidence map)
        """
        super().__init__()
        self.pool_stride = pool_stride
        # small conv head that consumes pooled photometric error and auxiliary features
        # We'll concatenate pooled photometric error (1 ch) with avg-pooled aux features
        self.conv = nn.Sequential(
            nn.Conv2d(1 + in_feat_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1)
        )

    def forward(self, phot_err: torch.Tensor, aux_feat: Optional[torch.Tensor] = None):
        """
        phot_err: (B, 1, H, W)
        aux_feat: (B, C, H, W) or None
        returns: confidence map c_t in [0,1] at pooled resolution (B, 1, H//pool, W//pool)
        """
        # coarse pooling: use adaptive avg pooling to reduce to target size
        B, _, H, W = phot_err.shape
        pooled_H = max(1, H // self.pool_stride)
        pooled_W = max(1, W // self.pool_stride)
        pe = F.adaptive_avg_pool2d(phot_err, (pooled_H, pooled_W))  # (B,1,h',w')

        if aux_feat is None:
            # create zeros for aux features
            aux = torch.zeros(B, 0, pooled_H, pooled_W, device=pe.device)
        else:
            aux = F.adaptive_avg_pool2d(aux_feat, (pooled_H, pooled_W))

        # if aux has zero channels, expand to zeros with 0 channels handled by concat below
        if aux.shape[1] == 0:
            x = pe
        else:
            x = torch.cat([pe, aux], dim=1)

        out = self.conv(x)
        c = torch.sigmoid(out)
        return c


if __name__ == "__main__":
    import torch
    head = ConfidenceHead(in_feat_channels=8, pool_stride=8)
    pe = torch.rand(2, 1, 128, 128)
    aux = torch.rand(2, 8, 128, 128)
    c = head(pe, aux)
    print("Confidence shape:", c.shape)
