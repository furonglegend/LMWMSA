"""
ablation_runner.py

Automates ablation experiments across different configurations and records results.

Design:
 - Accepts a base YAML config and a set of ablation overrides (via CLI or inline)
 - For each ablation variant:
     - modify config (in-memory)
     - run training or evaluation command (placeholder)
     - collect evaluation metrics (stubbed)
     - save results to CSV for inclusion in tables

This script is a driver: replace `run_experiment_variant` with actual training/eval invocation
(e.g., call trainer scripts or submit jobs to a scheduler).

Usage example:
    python ablation_runner.py --config configs/default.yaml --out_dir ./ablation_results \
        --variants "use_confidence=True;topk_blocks=4" "use_confidence=False;topk_blocks=8"
"""

import argparse
import os
import csv
import yaml
import subprocess
from typing import List, Dict, Any

def parse_variant_str(variant: str) -> Dict[str, Any]:
    """
    Parse variant string of the form "key1=val1;key2=val2" into a dict.
    Values are interpreted as int/float/bool if possible, otherwise left as strings.
    """
    out = {}
    if not variant:
        return out
    parts = variant.split(";")
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        v = v.strip()
        # interpret boolean and numeric types
        if v.lower() in ("true", "false"):
            vv = v.lower() == "true"
        else:
            try:
                if "." in v:
                    vv = float(v)
                else:
                    vv = int(v)
            except Exception:
                vv = v
        out[k.strip()] = vv
    return out

def apply_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple override applier: supports only top-level keys or dotted keys like 'model.topk_blocks'.
    """
    cfg_new = dict(cfg)
    for k, v in overrides.items():
        if "." in k:
            parts = k.split(".")
            d = cfg_new
            for p in parts[:-1]:
                if p not in d:
                    d[p] = {}
                d = d[p]
            d[parts[-1]] = v
        else:
            cfg_new[k] = v
    return cfg_new

def run_experiment_variant(cfg: Dict[str, Any], work_dir: str) -> Dict[str, float]:
    """
    Replace this function with real training/evaluation invocation.
    Here we simulate running and return synthetic metrics for demonstration.
    """
    # In real usage: write cfg to temp file and call subprocess to run trainer/eval:
    # cfg_path = os.path.join(work_dir, "config_variant.yaml")
    # with open(cfg_path, "w") as fh:
    #     yaml.safe_dump(cfg, fh)
    # subprocess.run(["python", "trainers/train_sparse.py", "--config", cfg_path], check=True)
    # then call evaluator and parse metrics
    # For now return stub metrics
    import random
    base = 28.0
    noise = random.uniform(-0.5, 0.5)
    return {"psnr": base + noise, "lpips": 0.1 + random.uniform(-0.02, 0.02)}

def save_results_csv(rows: List[Dict[str, Any]], out_csv: str):
    if len(rows) == 0:
        return
    keys = sorted(rows[0].keys())
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[ABlation] Saved results to {out_csv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Base YAML config")
    parser.add_argument("--variants", nargs="+", required=True, help="List of variant strings e.g. 'model.topk_blocks=4;training.lr=1e-4'")
    parser.add_argument("--out_dir", default="./ablation_results", help="Where to write results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.config, "r") as fh:
        base_cfg = yaml.safe_load(fh)

    results = []
    for i, vstr in enumerate(args.variants):
        overrides = parse_variant_str(vstr)
        cfg_v = apply_overrides(base_cfg, overrides)
        print(f"[AB] Running variant {i}: overrides={overrides}")
        work_dir = os.path.join(args.out_dir, f"variant_{i}")
        os.makedirs(work_dir, exist_ok=True)
        metrics = run_experiment_variant(cfg_v, work_dir)
        row = {"variant_idx": i, "overrides": str(overrides)}
        row.update(metrics)
        results.append(row)

    out_csv = os.path.join(args.out_dir, "ablation_summary.csv")
    save_results_csv(results, out_csv)

if __name__ == "__main__":
    main()
