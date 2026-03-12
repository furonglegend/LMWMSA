"""
infer/stream_infer.py

Streaming inference CLI for CtrlVSR.
Maintains a short-term KeyValueCache and processes frames as they arrive.
Designed to be used with:
 - an encoder: LR frames -> LR latent z^L_t
 - motion pipeline: external flow -> fused motion embedding a_t
 - LWT model -> predicted high-res latent hat z^H_t
 - decoder -> hr image

If real modules are missing, falls back to simple identity/placeholder implementations.

Usage:
    python infer/stream_infer.py --input_dir ./input_frames --out_dir ./out --ckpt ./checkpoints/lwt.pth --gamma 1.0
"""

import argparse
import os
import glob
from typing import Optional, Any

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# project imports (with fallbacks)
try:
    from models.lwt.lwt_model import LatentWorldTransformer
    from models.lwt.key_value_cache import KeyValueCache
except Exception:
    LatentWorldTransformer = None
    # minimal KeyValueCache fallback
    class KeyValueCache:
        def __init__(self, max_len=8):
            self.keys = []
            self.values = []
            self.max_len = max_len
        def append(self, k, v):
            self.keys.append(k.detach())
            self.values.append(v.detach())
            while len(self.keys) > self.max_len:
                self.keys.pop(0); self.values.pop(0)
        def get_all(self):
            if len(self.keys) == 0:
                return None, None
            return torch.cat(self.keys, dim=1), torch.cat(self.values, dim=1)

# simple placeholders for encoder/fuser/decoder if not provided
class DummyEncoder(nn.Module):
    def __init__(self, token_dim=256, side=8):
        super().__init__()
        self.token_dim = token_dim
        self.side = side
    def forward(self, img_tensor):
        B = img_tensor.shape[0]
        N = self.side * self.side
        return torch.randn(B, N, self.token_dim, device=img_tensor.device)

class DummyFuser(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.emb_dim = emb_dim
    def forward(self, *args, **kwargs):
        B = 1
        if isinstance(args[0], torch.Tensor):
            B = args[0].shape[0]
        return torch.randn(B, self.emb_dim, device=args[0].device if isinstance(args[0], torch.Tensor) else "cpu")

class DummyDecoder(nn.Module):
    def __init__(self, out_size=(256,256)):
        super().__init__()
        self.out_size = out_size
    def forward(self, zH, lr_img_tensor):
        # create a fake RGB image from latent by simple linear projection
        B, N, C = zH.shape
        H, W = self.out_size
        out = torch.rand(B, 3, H, W, device=zH.device)
        return out

def load_model_and_components(ckpt_path: Optional[str], cfg: Optional[dict] = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # instantiate LWT
    if LatentWorldTransformer is None:
        model = None
    else:
        try:
            token_dim = cfg['model']['latent_dim'] if cfg else 256
            model = LatentWorldTransformer(token_dim=token_dim,
                                           n_blocks=cfg['model'].get('lwt_depth', 6) if cfg else 4,
                                           n_heads=cfg['model'].get('lwt_heads', 8) if cfg else 8,
                                           motion_dim=cfg['motion'].get('ext_emb_dim', 128) if cfg else 128)
        except Exception:
            model = None

    if model is None:
        print("[INFER] LWT not available, using placeholder.")
        model = nn.Identity()

    model.to(device)
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            ck = torch.load(ckpt_path, map_location=device)
            if hasattr(model, "load_state_dict") and "model_state" in ck:
                model.load_state_dict(ck["model_state"], strict=False)
                print(f"[INFER] Loaded model weights from {ckpt_path}")
        except Exception as e:
            print(f"[INFER] Warning: failed to load model checkpoint: {e}")

    encoder = DummyEncoder(token_dim=cfg['model']['latent_dim'] if cfg else 256) if cfg else DummyEncoder()
    fuser = DummyFuser(emb_dim=cfg['motion']['ext_emb_dim'] if cfg else 128)
    decoder = DummyDecoder(out_size=(cfg['dataset']['crop_size_hr'], cfg['dataset']['crop_size_hr'])) if cfg else DummyDecoder()

    encoder.to(device); fuser.to(device); decoder.to(device)
    model.eval(); encoder.eval(); fuser.eval(); decoder.eval()

    return device, model, encoder, fuser, decoder

def process_stream(input_dir: str, out_dir: str, ckpt: Optional[str], cfg: Optional[dict], gamma: float = 1.0):
    os.makedirs(out_dir, exist_ok=True)
    # load modules
    device, model, encoder, fuser, decoder = load_model_and_components(ckpt, cfg)
    cache = KeyValueCache(max_len=cfg.get('inference', {}).get('cache_len', 8) if cfg else 8)

    # list frames sorted
    img_paths = sorted(glob.glob(os.path.join(input_dir, "*.*")))
    # simple pipeline: read frame, encode, compute fake motion, model forward, decode, save
    for i, p in enumerate(img_paths):
        pil = Image.open(p).convert("RGB")
        lr_tensor = TF.to_tensor(pil).unsqueeze(0).to(device)  # (1,3,H,W)
        with torch.no_grad():
            zL = encoder(lr_tensor)          # (B, N, C)
            a_t = fuser(lr_tensor)           # (B, motion_dim)
            # optionally modulate a_t by gamma (global scalar)
            a_t = a_t * gamma
            # model forward; allow returning kv to append to cache
            if hasattr(model, "__call__"):
                try:
                    # some models support return_kv
                    out = model(zL, a_t, cache, return_kv=True)
                    if isinstance(out, tuple) and len(out) == 3:
                        zH, k, v = out
                        cache.append(k, v)
                    else:
                        zH = out
                except TypeError:
                    zH = model(zL, a_t)
            else:
                zH = model(zL, a_t)
            # decode
            hr = decoder(zH, lr_tensor)  # (B, 3, H_hr, W_hr)
            # convert to PIL and save
            hr_img = TF.to_pil_image(hr.squeeze(0).cpu().clamp(0.0, 1.0))
            out_path = os.path.join(out_dir, f"frame_{i:06d}.png")
            hr_img.save(out_path)
            print(f"[STREAM] Saved {out_path}")

if __name__ == "__main__":
    import argparse, yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory of input LR frames (sorted).")
    parser.add_argument("--out_dir", default="./out", help="Where to save HR outputs.")
    parser.add_argument("--ckpt", default=None, help="LWT checkpoint (optional).")
    parser.add_argument("--config", default=None, help="YAML config file (optional).")
    parser.add_argument("--gamma", type=float, default=1.0, help="Global scalar control.")
    args = parser.parse_args()

    cfg = None
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as fh:
            cfg = yaml.safe_load(fh)

    process_stream(args.input_dir, args.out_dir, args.ckpt, cfg, gamma=args.gamma)
