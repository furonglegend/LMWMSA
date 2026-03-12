"""
models/motion/residual_net.py

Residual correction network R_rho that predicts a residual correction r_t to be added to
external motion embeddings. The input is a short sequence of LR frames and the external
motion embedding; the output is a residual vector matching the embedding dimension.

Design:
 - Small CNN to extract visual features from the image window (stacked along channel dim)
 - MLP to combine image features with external motion embedding and produce residual vector
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualCorrector(nn.Module):
    def __init__(self, img_channels: int = 3, window_length: int = 5, ext_emb_dim: int = 128, hidden: int = 64):
        super().__init__()
        in_ch = img_channels * window_length
        # compact conv backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # global pooling
        )
        # MLP to combine image feature + external motion embedding
        self.mlp = nn.Sequential(
            nn.Linear(64 + ext_emb_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, ext_emb_dim)
        )

    def forward(self, img_window: torch.Tensor, ext_emb: Optional[torch.Tensor]):
        """
        img_window: (B, T*C, H, W) or (B, in_ch, H, W)
        ext_emb: (B, ext_emb_dim) or None
        returns: residual vector r_t of shape (B, ext_emb_dim)
        """
        B = img_window.shape[0]
        feat = self.cnn(img_window).view(B, -1)  # (B, 64)

        if ext_emb is None:
            device = img_window.device
            ext_emb = torch.zeros(B, self.mlp[0].in_features - feat.shape[1], device=device) \
                if (self.mlp[0].in_features - feat.shape[1]) > 0 else torch.zeros(B, 0, device=device)

        # if shapes mismatch, pad or slice ext_emb to ext_emb_dim
        if ext_emb.dim() == 1:
            ext_emb = ext_emb.unsqueeze(0).repeat(B, 1)

        inp = torch.cat([feat, ext_emb], dim=1)
        residual = self.mlp(inp)
        return residual


if __name__ == "__main__":
    import torch
    net = ResidualCorrector(img_channels=3, window_length=5, ext_emb_dim=128, hidden=64)
    dummy_img = torch.randn(2, 3 * 5, 64, 64)
    dummy_ext = torch.randn(2, 128)
    r = net(dummy_img, dummy_ext)
    print("Residual shape:", r.shape)
