"""
trainers/train_sparse.py

Concrete training loop skeleton for the "sparse route" (train the sparse-causal LWT directly).
This script includes:
 - config loading
 - dataset/dataloader setup (placeholders)
 - model instantiation (LWT)
 - optimizer & scheduler
 - simple training loop with logging and checkpointing

The loss functions (DMD, flow, pixel, LPIPS, JEPA, adversarial, conf) should be
implemented in your losses/ module and plugged into the `compute_losses` function.
"""

import argparse
import os
import time
import yaml
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# placeholder imports
# from data.datasets import VideoFrameDataset
from models.lwt.lwt_model import LatentWorldTransformer

def load_config(path):
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg

def build_dataset_and_loader(cfg):
    """
    Build dataset and dataloader. Replace with your dataset class.
    """
    # TODO: replace with real dataset & transforms
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, length=100):
            self.length = length
        def __len__(self):
            return self.length
        def __getitem__(self, idx):
            # return dummy sample: lr frames stack and hr (possibly None)
            B = 1
            side = 8
            N = side * side
            C = cfg['model']['latent_dim']
            # LR latent tokens are random here; in real pipeline encoder produces them
            return {
                "zL": torch.randn(N, C),         # shape (N, C) per-sample
                "a_t": torch.randn(cfg['motion']['ext_emb_dim']),  # fused motion emb
                "hr": None
            }

    ds = DummyDataset(500)
    loader = DataLoader(ds, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=2)
    return ds, loader

def compute_losses(pred_zH, sample, model, cfg) -> Dict[str, torch.Tensor]:
    """
    Placeholder loss computation. Replace with the real loss modules.
    Return dict of loss_name -> tensor
    """
    # simplest L2 between predicted latent and a pseudo target (here zeros)
    target = torch.zeros_like(pred_zH)
    l2 = ((pred_zH - target) ** 2).mean()
    return {"l2": l2}

def save_checkpoint(state: Dict, path: str):
    torch.save(state, path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to sparse_route.yaml or default.yaml")
    parser.add_argument("--out", default=None, help="Output checkpoint prefix")
    args = parser.parse_args()
    cfg = load_config(args.config)

    save_dir = cfg['checkpoint']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    # dataset
    ds, loader = build_dataset_and_loader(cfg)

    # model
    token_dim = cfg['model']['latent_dim']
    lwt = LatentWorldTransformer(token_dim=token_dim,
                                 n_blocks=cfg['model'].get('lwt_depth', 6),
                                 n_heads=cfg['model'].get('lwt_heads', 8),
                                 motion_dim=cfg['motion'].get('ext_emb_dim', 128),
                                 block_size=cfg['model'].get('block_size', 16),
                                 topk_blocks=cfg['model'].get('topk_blocks', 6))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lwt.to(device)

    # optimizer & scheduler
    optim = torch.optim.AdamW(lwt.parameters(), lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg['training']['num_epochs'])

    # training loop
    global_step = 0
    for epoch in range(cfg['training']['num_epochs']):
        lwt.train()
        epoch_loss = 0.0
        t0 = time.time()
        for batch_idx, sample in enumerate(loader):
            # unpack sample and move to device
            # dummy dataset returns per-sample dict without batch dim; adapt for real pipeline
            # for our skeleton, we will expand dims to form a batch
            zL = sample['zL']  # shape (batch, N, C) or (N,C) depending on dataset
            a_t = sample['a_t']

            # ensure batched tensors
            if zL.dim() == 2:
                zL = zL.unsqueeze(0)
            if a_t.dim() == 1:
                a_t = a_t.unsqueeze(0)

            zL = zL.to(device)
            a_t = a_t.to(device)

            optim.zero_grad()
            pred_zH = lwt(zL, a_t, cache=None)  # (B, N, C)

            losses = compute_losses(pred_zH, sample, lwt, cfg)
            loss = sum([v for v in losses.values()])
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(lwt.parameters(), cfg['training'].get('grad_clip_norm', 1.0))
            optim.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % 50 == 0:
                print(f"[Epoch {epoch}] step {global_step} loss {loss.item():.6f}")

        scheduler.step()
        epoch_time = time.time() - t0
        print(f"Epoch {epoch} finished. avg loss {epoch_loss / (batch_idx + 1):.6f}. time {epoch_time:.1f}s")

        # checkpointing
        if (epoch + 1) % cfg['checkpoint'].get('save_every_epochs', 5) == 0:
            ckpt_path = os.path.join(save_dir, f"lwt_epoch{epoch+1}.pth")
            save_checkpoint({
                "epoch": epoch + 1,
                "model_state": lwt.state_dict(),
                "optim_state": optim.state_dict(),
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    print("Training complete.")

if __name__ == "__main__":
    main()
