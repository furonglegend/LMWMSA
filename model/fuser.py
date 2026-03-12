"""
models/motion/fuser.py

Motion fusion module M_phi which combines:
  - confidence-weighted external embedding
  - (1 - confidence)-weighted internal embedding
  - residual correction
Then projects fused vector into the transformer's motion embedding space.

Also supports modulation by user controls (global scalar gamma and spatial gate g_t).
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionFuser(nn.Module):
    def __init__(self, emb_dim: int = 128, proj_dim: int = 256, spatial_gate_size: Optional[tuple] = None):
        """
        Args:
            emb_dim: input embedding dimensionality (external/internal/residual)
            proj_dim: output fused embedding dimension expected by LWT
            spatial_gate_size: if provided, expected size (H', W') for learned spatial gate g_t.
                               If None, spatial gating is disabled and only global scalar is supported.
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        )
        if spatial_gate_size is not None:
            # learnable small conv to produce spatial gate; initialize near zero
            H, W = spatial_gate_size
            self.spatial_gate = nn.Parameter(torch.zeros(1, 1, H, W))
        else:
            self.spatial_gate = None

    def forward(
        self,
        ext_emb: Optional[torch.Tensor],
        int_emb: Optional[torch.Tensor],
        residual: Optional[torch.Tensor],
        confidence: Optional[torch.Tensor],
        gamma: float = 1.0,
        g_t: Optional[torch.Tensor] = None
    ):
        """
        ext_emb: (B, emb_dim) or None
        int_emb: (B, emb_dim) or None
        residual: (B, emb_dim) or None
        confidence: (B, 1, h', w') or (B, emb_dim) broadcastable or None
        gamma: global scalar control
        g_t: user-supplied spatial gate (same shape as confidence) or None

        Returns:
          fused_emb: (B, proj_dim)
        """
        B = None
        # ensure embeddings exist
        if ext_emb is None and int_emb is None:
            raise ValueError("At least one of ext_emb or int_emb must be provided.")

        # unify shapes and set defaults
        if ext_emb is None:
            ext_emb = torch.zeros_like(int_emb)
        if int_emb is None:
            int_emb = torch.zeros_like(ext_emb)
        B = ext_emb.shape[0]

        if residual is None:
            residual = torch.zeros_like(ext_emb)

        # Default confidence: scalar per-batch (if missing), set to 1.0 for trusting external
        if confidence is None:
            c = torch.ones(B, 1, device=ext_emb.device)
        else:
            # reduce spatial confidence to a scalar per-sample by global averaging if needed
            if confidence.dim() == 4:
                # (B,1,h,w) -> (B,1)
                c = confidence.mean(dim=[2, 3], keepdim=True)
            elif confidence.dim() == 2 or confidence.dim() == 1:
                c = confidence.view(B, -1)
            else:
                c = confidence

        # allow user spatial gate to adjust confidence if provided
        if g_t is not None:
            # reduce g_t similarly
            if g_t.dim() == 4:
                g = g_t.mean(dim=[2, 3], keepdim=True)
            else:
                g = g_t.view(B, -1)
            # combine: small-norm regularization for g is expected elsewhere; here just add
            c = c + g

        # apply global scalar gamma by scaling external contribution
        c = torch.clamp(c * gamma, 0.0, 1.0)

        # fuse embeddings element-wise using broadcasting
        # ext contribution: c * ext_emb
        # int contribution: (1 - c) * int_emb
        # residual added afterwards
        # ensure c has shape (B,1) to broadcast over emb dim
        if c.dim() == 1:
            c = c.unsqueeze(1)
        if c.dim() == 3:  # unexpected 3 dims -> squeeze
            c = c.squeeze(-1)

        fused = c * ext_emb + (1.0 - c) * int_emb + residual
        out = self.proj(fused)
        return out


if __name__ == "__main__":
    import torch
    fuser = MotionFuser(emb_dim=128, proj_dim=256)
    ext = torch.randn(2, 128)
    inta = torch.randn(2, 128)
    res = torch.randn(2, 128)
    conf = torch.rand(2, 1, 8, 8)
    fused = fuser(ext, inta, res, conf, gamma=0.9)
    print("Fused shape:", fused.shape)
