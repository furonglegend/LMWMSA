"""
infer/visualize_attention.py

Utilities to visualize block-level relevance scores or attention maps.

Given an image and block-level scores (H_blocks x W_blocks), overlay a heatmap
showing which blocks were selected / retrieved by the adaptive selector.

This script includes:
 - `visualize_block_scores(image_pil, block_scores, block_size, save_path)`
 - CLI to run a basic demo with a random score map if none provided.

Requirements: matplotlib
"""

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os

def visualize_block_scores(image_pil: Image.Image, block_scores: np.ndarray, block_size: int, save_path: str, cmap: str = "Reds"):
    """
    Overlay block_scores on image and save.

    Args:
      image_pil: PIL.Image (RGB)
      block_scores: 2D array (H_blocks, W_blocks) values in [0,1] or arbitrary (will be normalized)
      block_size: integer pixel size of blocks (approximate; for tokens you can compute from image size)
      save_path: path to save overlay image
    """
    img = image_pil.convert("RGBA")
    W, H = img.size
    Hb, Wb = block_scores.shape

    # normalize scores
    scores = np.array(block_scores, dtype=float)
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        scores = np.clip(scores, 0.0, 1.0)

    # create overlay figure with matplotlib colormap
    cmap = plt.get_cmap(cmap)
    overlay = Image.new("RGBA", (W, H), (0,0,0,0))
    draw = ImageDraw.Draw(overlay)

    # compute block pixel size (rounded)
    ph = H / Hb
    pw = W / Wb

    for i in range(Hb):
        for j in range(Wb):
            val = float(scores[i, j])
            if val <= 0.0:
                continue
            # get color from cmap (RGBA in 0-1)
            rgba = cmap(val)
            color = tuple(int(255 * c) for c in rgba[:3]) + (int(180 * val),)  # alpha scaled by val
            left = int(j * pw)
            top = int(i * ph)
            right = int(min(W, (j + 1) * pw))
            bottom = int(min(H, (i + 1) * ph))
            draw.rectangle([left, top, right, bottom], fill=color)

    composed = Image.alpha_composite(img, overlay)
    composed.convert("RGB").save(save_path)
    print(f"[VIS] Saved attention overlay to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=False, help="Image file to visualize (optional).")
    parser.add_argument("--out", default="./att_overlay.png", help="Where to save overlay.")
    parser.add_argument("--hb", type=int, default=8, help="Number of blocks vertically.")
    parser.add_argument("--wb", type=int, default=8, help="Number of blocks horizontally.")
    args = parser.parse_args()

    if args.image and os.path.exists(args.image):
        img = Image.open(args.image).convert("RGB")
    else:
        # create demo image
        img = Image.new("RGB", (512, 512), color=(120, 120, 120))

    # random demo block scores
    scores = np.random.rand(args.hb, args.wb)
    visualize_block_scores(img, scores, block_size=64, save_path=args.out)
