"""
data/flow_cache.py

FlowCache is a small utility to manage external flow fields used by CtrlVSR.
It attempts to load precomputed flow arrays from disk (saved as .npz or .npy).
If a precomputation is missing and compute_if_missing is True, an externally
provided flow_extractor function will be called to produce and store the flow.

The interface is intentionally minimal:
  - FlowCache.get_flow(video_id, frame_idx) -> returns flow between frame_idx -> frame_idx+1
  - FlowCache.get_flows_for_window(video_id, center_idx, radius) -> returns dict of flows for window

This module does not implement any particular external flow network; instead,
it accepts a callable `flow_extractor(video_path, out_path=None)` that computes
flows and returns a dictionary or saves to disk.
"""

import os
import threading
from typing import Optional, Dict, Callable

import numpy as np


class FlowCache:
    """
    FlowCache manages loading and optional on-the-fly computation of optical flows.

    Args:
        cache_root: folder where per-video flow arrays are stored.
                    Expect layout: cache_root/{video_id}/flow_{i:06d}.npz
        compute_if_missing: if True and a requested flow is missing, call flow_extractor
        flow_extractor: optional callable to compute flows for a video. Signature:
                        flow_extractor(video_folder: str, out_dir: str) -> Dict[int, np.ndarray]
                        It should compute and write flows to out_dir or return a dict mapping frame_idx->flow.
        lock: internal lock for thread-safety
    """

    def __init__(
        self,
        cache_root: str,
        compute_if_missing: bool = False,
        flow_extractor: Optional[Callable[[str, str], Dict[int, np.ndarray]]] = None,
    ):
        self.cache_root = cache_root
        self.compute_if_missing = compute_if_missing
        self.flow_extractor = flow_extractor
        self._locks = {}
        self._global_lock = threading.Lock()

    def _video_cache_dir(self, video_id: str) -> str:
        return os.path.join(self.cache_root, video_id)

    def _flow_path(self, video_id: str, idx: int) -> str:
        # use 6-digit convention
        return os.path.join(self._video_cache_dir(video_id), f"flow_{idx:06d}.npz")

    def exists(self, video_id: str, idx: int) -> bool:
        return os.path.exists(self._flow_path(video_id, idx))

    def _ensure_video_dir(self, video_id: str):
        d = self._video_cache_dir(video_id)
        if not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)

    def _acquire_lock_for(self, video_id: str):
        with self._global_lock:
            if video_id not in self._locks:
                self._locks[video_id] = threading.Lock()
            return self._locks[video_id]

    def get_flow(self, video_id: str, idx: int) -> Optional[np.ndarray]:
        """
        Return flow field for forward flow from frame idx -> idx+1.
        Flow is expected to be a numpy array of shape (H, W, 2) or similar.
        Returns None if not available and cannot be computed.
        """
        path = self._flow_path(video_id, idx)
        if os.path.exists(path):
            try:
                data = np.load(path)
                # allow variety: 'flow' key, or single array in .npz
                if "flow" in data:
                    return data["flow"]
                # fallback: first array in file
                for v in data.files:
                    return data[v]
            except Exception:
                return None

        # missing file
        if self.compute_if_missing and self.flow_extractor is not None:
            # ensure one thread computes for a video at a time
            lock = self._acquire_lock_for(video_id)
            with lock:
                # another thread might have created it while waiting
                if os.path.exists(path):
                    return self.get_flow(video_id, idx)
                # run the extractor for the whole video (recommended)
                print(f"[FlowCache] Computing flows for video {video_id} on demand...")
                video_folder = video_id  # the extractor may expect the path; caller can override by closure
                out_dir = self._video_cache_dir(video_id)
                self._ensure_video_dir(video_id)
                try:
                    result = self.flow_extractor(video_folder, out_dir)
                    # if extractor returned a dict mapping idx->flow, save them
                    if isinstance(result, dict):
                        for k, arr in result.items():
                            save_path = self._flow_path(video_id, k)
                            np.savez_compressed(save_path, flow=arr)
                        # return the requested one if present
                        if idx in result:
                            return result[idx]
                    # otherwise, if extractor already wrote files to out_dir, try loading again
                    if os.path.exists(path):
                        return self.get_flow(video_id, idx)
                except Exception as e:
                    print(f"[FlowCache] Flow extraction failed for {video_id}: {e}")
                    return None
        return None

    def get_flows_for_window(self, video_id: str, center_idx: int, radius: int) -> Dict[int, Optional[np.ndarray]]:
        """
        Return a dict of flows for the temporal window centered at center_idx.
        We return forward flows for indices center_idx - radius ... center_idx + radius - 1
        (i.e., flow i represents i -> i+1). Missing entries are None.
        """
        flows = {}
        for i in range(center_idx - radius, center_idx + radius + 1):
            # usually we want flow from i -> i+1, so last index may be missing
            flows[i] = self.get_flow(video_id, i)
        return flows


if __name__ == "__main__":
    # quick demo usage (this demo uses a no-op extractor that returns random arrays)
    def fake_extractor(video_folder: str, out_dir: str):
        # produce 10 fake flows
        os.makedirs(out_dir, exist_ok=True)
        out = {}
        for i in range(10):
            arr = np.random.randn(64, 64, 2).astype(np.float32)
            out[i] = arr
            np.savez_compressed(os.path.join(out_dir, f"flow_{i:06d}.npz"), flow=arr)
        return out

    fc = FlowCache("/tmp/flow_cache", compute_if_missing=True, flow_extractor=fake_extractor)
    f = fc.get_flow("video_000", 3)
    print("Loaded flow shape:", None if f is None else f.shape)
