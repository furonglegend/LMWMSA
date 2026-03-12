"""
models/lwt/adaptive_sparse_attention.py

Block-level adaptive sparse attention:
 - Pool tokens into spatial blocks
 - Compute coarse block-level queries and keys
 - Score candidate blocks with scaled dot-product
 - Use a differentiable soft top-k during training (soft relaxation)
 - Use hard top-k selection during inference

This file provides two main helpers:
 - block_pooling: pool per-token q/k into per-block summaries
 - AdaptiveBlockSelector: nn.Module that computes selection weights and a final mask

The module returns a boolean mask (N_query_blocks x N_candidate_blocks) or soft weights
which can be expanded into a full token-level attention mask in the LWT.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def block_pooling(tokens: torch.Tensor, block_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Pool a 2D flattened token map into block summaries.

    Args:
        tokens: (B, N, C) where N = H*W tokens; tokens are assumed to be in raster order.
        block_size: block size in pixels (assumes square blocks).
    Returns:
        pooled: (B, Bn, C) pooled summary per block
        (H_blocks, W_blocks): number of blocks in each spatial dim
    """
    B, N, C = tokens.shape
    # attempt to infer H and W by assuming tokens are square grid
    side = int(N ** 0.5)
    assert side * side == N, "Tokens must form a square grid for this pooling helper"
    H = W = side
    # reshape to (B, C, H, W)
    x = tokens.transpose(1, 2).view(B, C, H, W)
    # compute number of blocks
    H_b = (H + block_size - 1) // block_size
    W_b = (W + block_size - 1) // block_size
    # adaptive avg pool to (H_b, W_b)
    pooled = F.adaptive_avg_pool2d(x, (H_b, W_b))  # (B, C, H_b, W_b)
    pooled = pooled.view(B, C, H_b * W_b).transpose(1, 2)  # (B, H_b*W_b, C)
    return pooled, (H_b, W_b)


class AdaptiveBlockSelector(nn.Module):
    """
    Compute relevance scores between query blocks and candidate blocks and select top-k.

    Args:
        dim: dimensionality of pooled q/k vectors
        k_top: default top-k to select
        soft_topk_temp: temperature for soft top-k relaxation during training
        training_soft_topk: if True, returns soft selection weights during training
    """
    def __init__(self, dim: int, k_top: int = 8, soft_topk_temp: float = 0.05, training_soft_topk: bool = True):
        super().__init__()
        self.k_top = k_top
        self.temp = soft_topk_temp
        self.training_soft_topk = training_soft_topk

    def forward(self, q_blocks: torch.Tensor, k_blocks: torch.Tensor, mask: Optional[torch.Tensor] = None, topk_override: Optional[int] = None):
        """
        q_blocks: (B, Qb, C)
        k_blocks: (B, Bb, C)
        mask: optional additive mask for candidate blocks (B, Bb) or (Bb,) where masked positions get -inf
        topk_override: if provided, use this top-k value instead of self.k_top

        Returns:
            selection_weights: (B, Qb, Bb) soft weights (training) or hard {0,1} mask (inference)
        """
        B, Qb, C = q_blocks.shape
        _, Bb, _ = k_blocks.shape
        topk = topk_override or self.k_top

        # normalize for stable dot-product
        qn = F.normalize(q_blocks, dim=-1)
        kn = F.normalize(k_blocks, dim=-1)

        # compute scaled dot product scores: (B, Qb, Bb)
        scores = torch.matmul(qn, kn.transpose(-2, -1))  # cosine similarity
        scores = scores / (self.temp + 1e-8)

        if mask is not None:
            # mask expected to be additive: shape (B, Bb) or (Bb,)
            if mask.dim() == 2:
                scores = scores + mask.unsqueeze(1)  # broadcast to (B, Qb, Bb)
            else:
                scores = scores + mask.view(1, 1, -1)

        if self.training and self.training_soft_topk:
            # soft-topk: compute softmax over candidates and encourage sparsity via temperature
            weights = torch.softmax(scores, dim=-1)  # (B, Qb, Bb)
            # optionally, we can sharpen and then zero out small weights (not necessary)
            return weights
        else:
            # inference: hard top-k selection per query block
            topk = min(topk, Bb)
            topk_vals, topk_idx = torch.topk(scores, k=topk, dim=-1)  # (B, Qb, topk)
            # construct binary mask
            mask_tensor = torch.zeros_like(scores, dtype=torch.bool)
            # scatter True at topk indices
            mask_tensor = mask_tensor.scatter(-1, topk_idx, True)
            return mask_tensor.float()


if __name__ == "__main__":
    import torch
    B = 2
    N = 64  # 8x8 tokens
    C = 32
    block_size = 2
    tokens = torch.randn(B, N, C)
    pooled, (Hb, Wb) = block_pooling(tokens, block_size)
    print("Pooled shape:", pooled.shape, "blocks:", Hb, Wb)
    selector = AdaptiveBlockSelector(dim=C, k_top=4)
    weights = selector(pooled, pooled)
    print("Selector output shape:", weights.shape)
