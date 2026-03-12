"""
models/lwt/blocks.py

Transformer building blocks used by the Latent World Transformer (LWT).
Includes:
 - PreNorm wrapper
 - FeedForward (MLP) block
 - MultiHeadSelfAttention wrapper that supports optional attention masks
 - LocalSelfAttention helper (windowed attention) -- simple implementation

All code is PyTorch-based and intentionally lightweight so you can
replace or extend pieces (e.g. swap attention for FlashAttention).
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PreNorm(nn.Module):
    """LayerNorm followed by the wrapped module (common transformer pattern)."""
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """Simple MLP/FFN used inside transformer blocks."""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention wrapper compatible with (B, N, C) inputs.
    Accepts optional attention mask of shape (B, N, N) or (N, N) for causal / sparse masking.
    """
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        assert self.head_dim * heads == dim, "dim must be divisible by heads"

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        """
        x: (B, N, C)
        attn_mask: optional mask broadcastable to (B, heads, N, N) or (N, N)
                   mask should be additive (0 for keep, -inf or large negative for mask)
        """
        B, N, C = x.shape
        qkv = self.to_qkv(x)  # (B, N, 3C)
        q, k, v = qkv.chunk(3, dim=-1)
        # reshape to (B, heads, N, head_dim)
        q = q.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.heads, self.head_dim).transpose(1, 2)

        # scaled dot-product
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, heads, N, N)

        if attn_mask is not None:
            # attn_mask additive: expected shape (B, 1, N, N) or (1, 1, N, N) or (N, N)
            attn = attn + attn_mask

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.out(out)


class TransformerBlock(nn.Module):
    """
    Single Transformer block consisting of:
      x = x + Attn(LN(x))
      x = x + FFN(LN(x))
    Attn module can be a MultiHeadSelfAttention or a custom attention that
    supports 'attn_mask' argument.
    """
    def __init__(self, dim: int, heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.attn = PreNorm(dim, MultiHeadSelfAttention(dim, heads=heads, dropout=dropout))
        hidden = int(dim * mlp_ratio)
        self.ff = PreNorm(dim, FeedForward(dim, hidden, dropout=dropout))

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.attn(x, attn_mask=attn_mask)
        x = x + self.ff(x)
        return x
