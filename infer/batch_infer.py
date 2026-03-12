"""
infer/batch_infer.py

Batch-mode inference: runs the model over datasets or folders in batch and writes outputs.
Useful for evaluation on test sets (non-streaming).

CLI:
    python infer/batch_infer.py --input_dir ./test/vids --out_dir ./results --ckpt ./checkpoints/lwt.pth --batch_size 4
"""

import argparse
import os
import glob
from typing import Optional

from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset

# re-use components from stream_infer when possible
from infer.stream_infer import load_model_and_components

class FramesFolderDataset(Dataset):
    """Simple dataset that reads frames from folders; returns image tensors and meta info."""
    def __init__(self, folder: str, transform=None):
        self.paths = sorted(glob.glob(os.path.join(folder, "*.*")))
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        t = TF.to_tensor(img) if self.transform is None else self.transform(img)
        return {"img": t, "path": p, "idx": idx}

def run_batch_infer(input_dir: str, out_dir: str, ckpt: Optional[str], cfg: Optional[dict], batch_size: int = 2):
    device, model, encoder, fuser, decoder = load_model_and_components(ckpt, cfg)
    # assume input_dir contains subfolders per-video or a single folder of frames
    # if subfolders exist, process each separately
    subfolders = [d for d in sorted(os.listdir(input_dir)) if os.path.isdir(os.path.join(input_dir, d))]
    if len(subfolders) == 0:
        subfolders = [""]  # treat input_dir itself as frames folder

    for sub in subfolders:
        frames_folder = os.path.join(input_dir, sub) if sub != "" else input_dir
        ds = FramesFolderDataset(frames_folder)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=lambda x: x)
        out_sub = os.path.join(out_dir, sub) if sub != "" else out_dir
        os.makedirs(out_sub, exist_ok=True)
        print(f"[BATCH] Processing folder {frames_folder} -> {out_sub}")

        for batch in loader:
            # batch is a list of dicts due to collate_fn
            imgs = torch.stack([item["img"] for item in batch], dim=0).to(device)
            metas = [item["path"] for item in batch]
            with torch.no_grad():
                zL = encoder(imgs)
                a_t = fuser(imgs)
                zH = model(zL, a_t)
                hr = decoder(zH, imgs)
            # save outputs
            for i in range(hr.shape[0]):
                out_path = os.path.join(out_sub, os.path.basename(metas[i]))
                TF.to_pil_image(hr[i].cpu().clamp(0,1)).save(out_path)
                print(f"[BATCH] Saved {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Root input dir (video subfolders or frames).")
    parser.add_argument("--out_dir", default="./results", help="Where to save outputs.")
    parser.add_argument("--ckpt", default=None, help="LWT checkpoint (optional).")
    parser.add_argument("--config", default=None, help="Optional YAML config.")
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    cfg = None
    if args.config:
        import yaml
        with open(args.config, "r") as fh:
            cfg = yaml.safe_load(fh)

    run_batch_infer(args.input_dir, args.out_dir, args.ckpt, cfg, args.batch_size)
