"""
plot_control_sweep.py

Sweep the global control scalar `gamma` (and optionally spatial gate strength)
and plot metrics such as PSNR / LPIPS / temporal consistency as a function of gamma.

This script assumes you have:
 - a saved checkpoint for the LWT decoder pipeline (or you can use dummy model)
 - an evaluation function `evaluate_gamma(gamma)` that returns a dict of metrics
   for the current gamma (this function is provided as a stub and must be replaced).

Usage:
    python plot_control_sweep.py --ckpt ./checkpoints/lwt.pth --out_dir ./figs --gammas 0.5 0.75 1.0 1.25 1.5

The script writes a PNG showing metrics vs gamma and also saves a CSV of results.
"""

import argparse
import os
import csv
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt

# stubbed evaluate function - replace with your real evaluation loop
def evaluate_gamma_stub(gamma: float, ckpt: str = None) -> Dict[str, float]:
    """
    Placeholder evaluation. In practice:
      - load model & decoder
      - run inference over validation videos with control gamma
      - compute PSNR/LPIPS/temporal_consistency metrics
    Here we synthesize metrics with a smooth curve for demo purposes.
    """
    # sanity bounds for demonstration (peak near gamma=1.0)
    psnr = 28.0 + 2.0 * np.exp(-((gamma - 1.0) ** 2) / 0.02) - 0.5 * (gamma - 1.0)
    lpips = max(0.02, 0.1 + 0.5 * abs(gamma - 1.0))
    temporal = 0.9 - 0.2 * abs(gamma - 1.0)
    return {"psnr": float(psnr), "lpips": float(lpips), "temporal_consistency": float(temporal)}

def sweep_gammas(gammas: List[float], ckpt: str = None) -> List[Dict]:
    results = []
    for g in gammas:
        metrics = evaluate_gamma_stub(g, ckpt=ckpt)
        metrics["gamma"] = float(g)
        print(f"[SWEEP] gamma={g:.3f} -> PSNR={metrics['psnr']:.3f}, LPIPS={metrics['lpips']:.3f}, TC={metrics['temporal_consistency']:.3f}")
        results.append(metrics)
    return results

def save_results_csv(results: List[Dict], out_csv: str):
    keys = list(results[0].keys())
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"[SAVE] Results CSV: {out_csv}")

def plot_results(results: List[Dict], out_png: str):
    gammas = [r["gamma"] for r in results]
    psnrs = [r["psnr"] for r in results]
    lpips = [r["lpips"] for r in results]
    temporal = [r["temporal_consistency"] for r in results]

    fig, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(gammas, psnrs, marker="o", label="PSNR (dB)")
    ax1.set_xlabel("Gamma (global control scalar)")
    ax1.set_ylabel("PSNR (dB)")
    ax1.grid(True)
    ax1.set_title("Control sweep: metrics vs gamma")

    ax2 = ax1.twinx()
    ax2.plot(gammas, lpips, marker="s", label="LPIPS", linestyle="--")
    ax2.plot(gammas, temporal, marker="^", label="Temporal Consistency", linestyle=":")
    ax2.set_ylabel("LPIPS / Temporal")

    # combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"[SAVE] Plot saved to {out_png}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None, help="Path to LWT checkpoint (unused in stub).")
    parser.add_argument("--out_dir", default="./figs", help="Output directory for plots and CSV.")
    parser.add_argument("--gammas", nargs="+", type=float, default=[0.5,0.75,1.0,1.25,1.5], help="List of gamma values to sweep.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results = sweep_gammas(args.gammas, ckpt=args.ckpt)
    out_csv = os.path.join(args.out_dir, "control_sweep_results.csv")
    save_results_csv(results, out_csv)
    out_png = os.path.join(args.out_dir, "control_sweep_plot.png")
    plot_results(results, out_png)

if __name__ == "__main__":
    main()
