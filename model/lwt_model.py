"""
models/lwt/lwt_model.py

Top-level Latent World Transformer (LWT) skeleton.
It:
 - Accepts a low-resolution latent z^L_t (B, N, C)
 - Accepts fused motion embedding a_t (B, motion_dim)
 - Accepts a KeyValueCache summarizing past frames
 - Applies positional encodings, injects motion conditioning, runs transformer blocks
 - Optionally uses adaptive sparse attention to retrieve remote blocks

This is intentionally a minimal, well-documented skeleton — replace building blocks
(FeedForward, attention) with your optimized implementations as needed.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import TransformerBlock
from .adaptive_sparse_attention import block_pooling, AdaptiveBlockSelector
from .key_value_cache import KeyValueCache


class PositionalEncoding2D(nn.Module):
    """
    Lightweight 2D Fourier positional encoding; produces per-token vectors added to inputs.
    Input tokens shape: (B, N, C) with N = H*W
    """
    def __init__(self, dim: int, grid_size: int):
        super().__init__()
        self.dim = dim
        self.grid_size = grid_size
        # small linear projection after sin/cos features
        self.proj = nn.Linear(4, dim)

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        side = int(N ** 0.5)
        device = x.device
        assert side * side == N, "LWT expects square spatial tokens"
        coords = torch.meshgrid(torch.linspace(-1, 1, side, device=device),
                                torch.linspace(-1, 1, side, device=device), indexing="ij")
        y, xg = coords
        # stack (y,x, y^2, x^2) as simple Fourier input
        feat = torch.stack([y, xg, y**2, xg**2], dim=-1).view(-1, 4)  # (N, 4)
        pos = self.proj(feat).unsqueeze(0).expand(B, -1, -1)  # (B, N, dim)
        return x + pos


class LatentWorldTransformer(nn.Module):
    """
    LWT skeleton.

    Args:
        token_dim: latent token dimensionality (C)
        n_blocks: number of transformer layers
        n_heads: attention heads
        motion_dim: dimension of fused motion embedding (a_t)
        block_size: block pooling size for adaptive retrieval
        topk_blocks: default topk
    """
    def __init__(
        self,
        token_dim: int = 256,
        n_blocks: int = 6,
        n_heads: int = 8,
        motion_dim: int = 256,
        block_size: int = 16,
        topk_blocks: int = 8,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.n_blocks = n_blocks
        self.blocks = nn.ModuleList([TransformerBlock(token_dim, heads=n_heads) for _ in range(n_blocks)])
        self.pos_enc = PositionalEncoding2D(token_dim, grid_size=0)
        # motion projection to token_dim and optional gating
        self.motion_proj = nn.Linear(motion_dim, token_dim)
        # adaptive block selector
        self.block_size = block_size
        self.selector = AdaptiveBlockSelector(dim=token_dim, k_top=topk_blocks)
        # small head to produce final high-resolution latent (could be identity)
        self.head = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, token_dim))

    def forward(self, zL: torch.Tensor, a_t: torch.Tensor, cache: Optional[KeyValueCache] = None, return_kv: bool = False):
        """
        zL: (B, N, C) low-res latent tokens
        a_t: (B, motion_dim) fused motion embedding
        cache: KeyValueCache with past keys/values (optional)
        return_kv: if True, also return (keys, values) that can be appended to cache

        Returns:
            zH_pred: (B, N, C) predicted high-res latent tokens (still latent; to be decoded)
            optionally keys/values for caching
        """
        B, N, C = zL.shape
        x = zL

        # add positional encodings
        x = self.pos_enc(x)

        # inject motion conditioning (broadcast to tokens)
        motion_cond = self.motion_proj(a_t).unsqueeze(1)  # (B,1,C)
        x = x + motion_cond

        # optionally incorporate cached keys/values via coarse block retrieval
        if cache is not None and len(cache) > 0:
            # retrieve concatenated keys/values
            past_k, past_v = cache.get_all()  # (B, M, C)
            if past_k is not None:
                # pool query blocks for current tokens
                q_blocks, (Hb_q, Wb_q) = block_pooling(x, self.block_size)
                # pool candidate blocks from past (note: we assume past_v arranged as spatial grids)
                # For simplicity we pool past_k as if they were a single large grid
                # This is a simplification: you may want to store per-frame block pooling in cache
                k_blocks, _ = block_pooling(past_k, self.block_size)
                # compute selection weights (soft during training)
                selection = self.selector(q_blocks, k_blocks)  # (B, Qb, Bb) soft or hard
                # Expand selection to per-token mask is left as a TODO for efficiency;
                # Here we simply ignore selection when computing attention inside blocks.
                # An advanced implementation would create a token-level mask and pass to attention.
                # For now, continue with regular attention and allow blocks to attend locally.
                pass

        # run transformer blocks (currently without sparse token-level masking)
        for block in self.blocks:
            x = block(x, attn_mask=None)

        zH = self.head(x)  # (B, N, C)

        # produce keys/values for caching: use simple linear projections (could be dedicated modules)
        keys = torch.tanh(zH)  # placeholder key
        values = zH  # placeholder value

        if return_kv:
            return zH, keys, values
        return zH


if __name__ == "__main__":
    import torch
    from .key_value_cache import KeyValueCache
    B = 2
    side = 8
    N = side * side
    C = 256
    zL = torch.randn(B, N, C)
    a = torch.randn(B, 256)
    cache = KeyValueCache(max_len=4)
    model = LatentWorldTransformer(token_dim=C, n_blocks=4, n_heads=8, motion_dim=256, block_size=2, topk_blocks=4)
    zH, k, v = model(zL, a, cache, return_kv=True)
    print("zH shape:", zH.shape, "k shape:", k.shape)
