"""
utils/controls.py

Utilities for handling user controls (global scalar gamma and spatial gate g_t)
and regularization used in training to keep gates small.

Provides:
 - apply_control: apply (gamma, g_t) modulation to motion embedding a_t (Eq.12)
 - spatial_gate_regularizer: returns L2 penalty term encouraging small norm for g_t
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple

def apply_control(a_t: torch.Tensor, gamma: float = 1.0, g_t: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Apply control modulation: (gamma * 1 + g_t) * a_t elementwise.
    a_t: (B, D) or (B, D, H, W) if spatial
    g_t: if provided, must broadcast to a_t shape or be (B, 1, H, W) for spatial gating
    """
    if g_t is None:
        return a_t * gamma
    # ensure shapes align
    if g_t.dim() == 2 and a_t.dim() == 2:
        # both (B, D) or (B,1)
        return (gamma + g_t) * a_t
    else:
        # broadcast
        return (gamma + g_t) * a_t

def spatial_gate_regularizer(g_t: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    """
    L2 regularization encouraging small spatial gate norm.
    Returns scalar penalty tensor.
    """
    return weight * (g_t.pow(2).mean())

def clamp_gamma(gamma: float, min_val: float = 0.0, max_val: float = 2.0) -> float:
    return float(max(min(gamma, max_val), min_val))


if __name__ == "__main__":
    import torch
    a = torch.randn(2, 128)
    g = torch.randn(2,1) * 0.01
    out = apply_control(a, gamma=0.9, g_t=g)
    print("Controlled shape:", out.shape)
    print("Reg:", spatial_gate_regularizer(g))
