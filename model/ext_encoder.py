"""
models/motion/ext_encoder.py

Lightweight external motion encoder F_ext.
This module encodes a short sequence of external motion cues (e.g. flow maps or keypoint arrays)
into a compact embedding vector for each center frame.

Design:
 - Accepts a list/array of flows for frames t-tau ... t+tau (forward flows are typical)
 - If flows are None or missing, returns zeros
 - Implementation uses small convnet + global pooling to produce embedding of size ext_emb_dim
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFlowEncoder(nn.Module):
    """
    SimpleFlowEncoder converts stacked flow fields into a compact embedding.
    Input: tensor of shape (B, T, 2, H, W) or (B, T-1, H, W, 2) rearranged to (B, T, 2, H, W)
    Output: (B, ext_emb_dim)
    """

    def __init__(self, in_channels: int = 2, timesteps: int = 3, ext_emb_dim: int = 128):
        super().__init__()
        self.timesteps = timesteps
        self.in_ch = in_channels
        ch = 32
        # a tiny convnet that processes concatenated flows along channel dim
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels * timesteps, ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # global pooling
        )
        self.project = nn.Sequential(
            nn.Linear(ch * 2, ext_emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ext_emb_dim, ext_emb_dim)
        )

    def forward(self, flows: Optional[torch.Tensor]):
        """
        flows: None or tensor of shape (B, T, 2, H, W)
        returns: (B, ext_emb_dim)
        """
        batch = None
        if flows is None:
            # return zeros (caller should be robust to missing flows)
            # Determine batch size from device heuristics (if possible); otherwise return scalar zero
            device = next(self.parameters()).device
            return torch.zeros(1, self.project[-1].out_features, device=device)

        # ensure shape (B, T, 2, H, W) -> merge T and 2 into channel dim
        if flows.dim() == 5:
            # flows is (B, T, 2, H, W)
            B, T, C, H, W = flows.shape
            x = flows.reshape(B, T * C, H, W)
        elif flows.dim() == 4:
            # flows is (B, C, H, W) single timestep
            B, C, H, W = flows.shape
            x = flows.unsqueeze(1).reshape(B, 1 * C, H, W)
        else:
            raise ValueError("Unexpected flows tensor shape: {}".format(flows.shape))

        feat = self.conv_net(x).view(x.shape[0], -1)  # (B, ch*2)
        emb = self.project(feat)
        return emb


if __name__ == "__main__":
    # quick sanity check
    m = SimpleFlowEncoder(in_channels=2, timesteps=3, ext_emb_dim=128)
    dummy = torch.randn(2, 3, 2, 64, 64)
    e = m(dummy)
    print("Embedding shape:", e.shape)
