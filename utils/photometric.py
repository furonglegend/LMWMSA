"""
utils/photometric.py

Photometric error computations used to estimate motion reliability.

Provides:
 - robust_photometric_error: Charbonnier-like robust pixel error
 - bidirectional_photometric_error: combine forward and backward warp errors as in paper

Note: requires utils.warp.warp_image for warping; import those helpers.
"""

import torch
import torch.nn.functional as F
from typing import Callable, Optional

from utils.warp import warp_image  # relative import assumes package layout


def charbonnier_loss(x: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    """Charbonnier robust error: sqrt(x^2 + eps^2)"""
    return torch.sqrt(x * x + (epsilon ** 2))


def robust_photometric_error(img1: torch.Tensor, img2_warped: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute a robust per-pixel photometric error between img1 and warped img2.

    Args:
      img1: (B,C,H,W)
      img2_warped: (B,C,H,W)
      mask: optional (B,1,H,W) with 1 for valid pixels

    Returns:
      per-pixel error map (B,1,H,W)
    """
    diff = (img1 - img2_warped).abs().mean(dim=1, keepdim=True)  # L1 across channels
    err = charbonnier_loss(diff)
    if mask is not None:
        err = err * mask
    return err


def bidirectional_photometric_error(img_prev: torch.Tensor, img_curr: torch.Tensor, img_next: torch.Tensor,
                                    flow_prev_to_curr: torch.Tensor, flow_curr_to_next: torch.Tensor,
                                    phot_err_fn: Callable = robust_photometric_error) -> torch.Tensor:
    """
    Compute bidirectional photometric inconsistency map as Eq.(3) in the paper.
    For center frame t we compute:
       PhotErr(I_t, Warp(I_{t+1}, f_{t->t+1})) + PhotErr(I_{t-1}, Warp(I_t, f_{t-1->t}))

    Note: caller must provide flows in correct directions.
    """
    # warp next to current using f_{t->t+1} but note: we need to warp I_{t+1} into I_t space
    warped_next = warp_image(img_next, flow_curr_to_next)  # warp I_{t+1} -> coords of I_t (assuming flow convention)
    err1 = phot_err_fn(img_curr, warped_next)  # PhotErr(I_t, Warp(I_{t+1}, f_{t->t+1}))

    warped_curr = warp_image(img_curr, flow_prev_to_curr)  # warp I_t into I_{t-1} space using f_{t-1->t}
    err2 = phot_err_fn(img_prev, warped_curr)  # PhotErr(I_{t-1}, Warp(I_t, f_{t-1->t}))

    return err1 + err2


if __name__ == "__main__":
    import torch
    # quick sanity check with identical images -> near zero error
    img = torch.rand(1,3,64,64)
    z = robust_photometric_error(img, img)
    print("PhotErr mean:", z.mean().item())
