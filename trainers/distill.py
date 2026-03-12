"""
trainers/distill.py

Distillation loop: distill a (sparse-causal) teacher into the single-step LWT student.

Design notes:
 - Loads teacher checkpoint (or EMA teacher) and student LWT
 - Freezes teacher, runs student forward, computes distillation objectives:
     - distribution matching distillation (latent MSE, L_DMD)
     - optional decoder distillation (perceptual + pixel) if teacher provides outputs
 - Supports saving checkpoints and logging simple metrics to stdout.
 - This is a skeleton: plug in your real loss functions and dataset.

CLI:
    python trainers/distill.py --config configs/full_curriculum.yaml --teacher_ckpt path --out_dir ./checkpoints
"""

import argparse
import os
import time
import yaml
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# try to import project modules; fall back to simple placeholders if absent
try:
    from data.datasets import VideoFrameDataset
except Exception:
    VideoFrameDataset = None

try:
    from models.lwt.lwt_model import LatentWorldTransformer
except Exception:
    LatentWorldTransformer = None

# -- basic helpers -----------------------------------------------------------
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)

def build_student(cfg):
    """Instantiate LWT student (or a minimal placeholder)."""
    if LatentWorldTransformer is None:
        # minimal placeholder model
        class DummyStudent(nn.Module):
            def __init__(self, token_dim=cfg['model']['latent_dim']):
                super().__init__()
                self.proj = nn.Linear(token_dim, token_dim)
            def forward(self, zL, a_t, cache=None, return_kv=False):
                out = self.proj(zL)
                if return_kv:
                    return out, out, out
                return out
        return DummyStudent()
    else:
        return LatentWorldTransformer(
            token_dim=cfg['model']['latent_dim'],
            n_blocks=cfg['model'].get('lwt_depth', 6),
            n_heads=cfg['model'].get('lwt_heads', 8),
            motion_dim=cfg['motion'].get('ext_emb_dim', 128),
            block_size=cfg['model'].get('block_size', 16),
            topk_blocks=cfg['model'].get('topk_blocks', 8)
        )

def build_teacher_stub():
    """Return a teacher stub that produces 'teacher latents' for distillation."""
    class TeacherStub(nn.Module):
        def __init__(self, token_dim=256):
            super().__init__()
            self.proj = nn.Linear(token_dim, token_dim)
        def forward(self, zL, a_t):
            # pretend teacher refines zL slightly
            return self.proj(zL) * 1.1
    return TeacherStub()

def save_ckpt(state: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

# -- training loop -----------------------------------------------------------
def distill_loop(cfg: Dict[str, Any], teacher_ckpt: str, out_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DISTILL] Using device: {device}")

    # instantiate student and teacher
    student = build_student(cfg).to(device)
    # load teacher if available; otherwise teacher stub
    teacher = build_teacher_stub().to(device)

    # optionally load teacher checkpoint (best-effort)
    if teacher_ckpt and os.path.exists(teacher_ckpt):
        try:
            ck = torch.load(teacher_ckpt, map_location=device)
            if hasattr(teacher, "load_state_dict") and "model_state" in ck:
                teacher.load_state_dict(ck["model_state"])
                print(f"[DISTILL] Loaded teacher weights from {teacher_ckpt}")
            else:
                print(f"[DISTILL] Teacher checkpoint found but no recognized keys; skipping weight load.")
        except Exception as e:
            print(f"[DISTILL] Failed to load teacher checkpoint: {e}")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # optimizer for student
    optim = torch.optim.AdamW(student.parameters(), lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'])

    # dataset / loader (best-effort): if VideoFrameDataset available, use it; else dummy
    if VideoFrameDataset is not None:
        ds = VideoFrameDataset(cfg['dataset']['root'], ["video_000"], window_radius=cfg['dataset'].get('window_radius', 2), flow_cache=None)
        loader = DataLoader(ds, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=4)
    else:
        # dummy loader yielding random latents
        class DummyDS(torch.utils.data.Dataset):
            def __init__(self, n=500):
                self.n = n
                self.N = 8 * 8
                self.C = cfg['model']['latent_dim']
            def __len__(self): return self.n
            def __getitem__(self, idx):
                return {
                    "zL": torch.randn(self.N, self.C),
                    "a_t": torch.randn(self.C),
                    "meta": {}
                }
        loader = DataLoader(DummyDS(1000), batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=2)

    # loss: distribution-matching MSE between teacher latent and student latent
    mse = nn.MSELoss()

    max_steps = cfg.get("distillation", {}).get("distill_steps", 10000)
    step = 0
    epoch = 0
    while step < max_steps:
        epoch += 1
        for sample in loader:
            student.train()
            zL = sample["zL"]
            a_t = sample["a_t"]
            if zL.dim() == 2:
                zL = zL.unsqueeze(0)
            if a_t.dim() == 1:
                a_t = a_t.unsqueeze(0)
            zL = zL.to(device)
            a_t = a_t.to(device)

            with torch.no_grad():
                teacher_z = teacher(zL, a_t)  # (B, N, C)

            student_z = student(zL, a_t)  # (B, N, C)
            loss_dmd = mse(student_z, teacher_z)
            loss = cfg['loss_weights'].get('lambda_DMD', 1.0) * loss_dmd

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), cfg['training'].get('grad_clip_norm', 1.0))
            optim.step()

            if step % 50 == 0:
                print(f"[DISTILL][epoch {epoch}] step {step} loss {loss.item():.6f} dmd {loss_dmd.item():.6f}")

            step += 1
            if step >= max_steps:
                break

        # periodic checkpoint
        if epoch % cfg['checkpoint'].get('save_every_epochs', 5) == 0 or step >= max_steps:
            ckpt = {
                "epoch": epoch,
                "step": step,
                "model_state": student.state_dict(),
                "optim_state": optim.state_dict()
            }
            out_path = os.path.join(out_dir, f"lwt_distill_epoch{epoch}_step{step}.pth")
            save_ckpt(ckpt, out_path)
            print(f"[DISTILL] Saved distillation checkpoint: {out_path}")

    print("[DISTILL] Completed distillation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config (e.g., full_curriculum.yaml)")
    parser.add_argument("--teacher_ckpt", default=None, help="Path to teacher checkpoint (optional)")
    parser.add_argument("--out_dir", default="./checkpoints", help="Where to save distilled student ckpts")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.out_dir, exist_ok=True)
    distill_loop(cfg, args.teacher_ckpt, args.out_dir)
