"""
utils/attention_utils.py

Helper utilities for attention-related computations:
 - soft_topk: differentiable approximation to top-k using softmax temperature and sharpening
 - expand_block_mask: helper to convert block-level boolean mask into token-level additive mask

These helpers are small but useful for the adaptive sparse attention implementation.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def soft_topk(scores: torch.Tensor, k: int, temp: float = 0.1) -> torch.Tensor:
    """
    Differentiable soft-topk: given scores (..., N), returns soft selection weights of same shape
    where mass concentrates on top-k entries. This uses softmax with temperature and a straight-through
    hard top-k during forward for stability (optional). Here we return soft weights always.

    Args:
      scores: (..., N)
      k: number of top elements to concentrate mass on
      temp: softmax temperature (lower -> sharper)

    Returns:
      weights: same shape as scores, values sum to 1 along last dim
    """
    # sharpen via scaled softmax
    w = F.softmax(scores / (temp + 1e-12), dim=-1)
    # optionally renormalize so that only top-k receive significant mass:
    # We zero out small weights below threshold (approx top-k)
    # Compute topk threshold per sample
    topk_vals, _ = torch.topk(w, k=k, dim=-1)
    kth_val = topk_vals[..., -1].unsqueeze(-1)  # (...,1)
    mask = (w >= kth_val).float()
    # renormalize masked weights
    w_masked = w * mask
    denom = w_masked.sum(dim=-1, keepdim=True) + 1e-12
    return w_masked / denom


def expand_block_mask(token_count: int, block_size: int, block_mask: torch.Tensor) -> torch.Tensor:
    """
    Expand a block-level boolean mask (B, Qb, Bb) to a token-level additive mask (B, Qn, N)
    suitable for attention masking. This is a simple helper and assumes square grids.

    Args:
      token_count: N (e.g., H*W)
      block_size: size of block in tokens along each spatial dim
      block_mask: (B, Qb, Bb) float/bool mask where 1 indicates selected block

    Returns:
      token_mask: (B, Qn, N) additive mask where selected positions are 0 and unselected are -inf
    """
    B, Qb, Bb = block_mask.shape
    side = int(token_count ** 0.5)
    H = W = side
    # compute block grid sizes
    Hb = (H + block_size - 1) // block_size
    Wb = (W + block_size - 1) // block_size
    assert Hb * Wb == Bb, "Block count mismatch"

    # build expansion mapping from block idx -> token indices
    # for simplicity construct mask per sample identically
    # create token-level mask values (0 for keep, -1e9 for mask)
    # first create a (Qb, N) matrix mapping block->tokens
    block_to_token = torch.zeros(Qb, token_count, dtype=torch.float32)
    # compute block coords for query blocks: assuming Qb == Hb*Wb
    for b in range(Qb):
        row = b // Wb
        col = b % Wb
        # token ranges
        h0 = row * block_size
        h1 = min(H, (row + 1) * block_size)
        w0 = col * block_size
        w1 = min(W, (col + 1) * block_size)
        # set corresponding tokens
        for i in range(h0, h1):
            for j in range(w0, w1):
                idx = i * W + j
                block_to_token[b, idx] = 1.0

    # now expand: for each query block q, the selected candidate blocks form a mask over tokens
    # we tile block_to_token to (B, Qb, N) and multiply by block_mask to obtain token-level mask
    # result token_mask_raw = block_mask (B, Qb, Bb) @ block_to_token (Bb, N) -> (B, Qb, N)
    device = block_mask.device
    block_to_token = block_to_token.to(device)
    token_mask_raw = torch.matmul(block_mask, block_to_token)  # (B, Qb, N)
    # token_mask_raw >0 indicates that token is covered by some selected block -> keep (0)
    keep = (token_mask_raw > 0.0).float()
    additive = (1.0 - keep) * (-1e9)  # masked tokens receive large negative additive
    return additive
