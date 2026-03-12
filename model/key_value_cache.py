"""
models/lwt/key_value_cache.py

Simple key-value cache used for streaming / short-term memory in LWT.
It stores a FIFO list of key and value tensors (per layer optionally) and
supports appending new entries and pruning old ones.

API:
    cache = KeyValueCache(max_len=32)
    cache.append(keys, values)         # keys/values are tensors (B, N_k, C)
    kv = cache.get_all()               # returns concatenated (B, total_N, C)
    cache.prune_to(max_len_new)        # shrink stored history
"""

from typing import List, Optional, Tuple
import torch


class KeyValueCache:
    def __init__(self, max_len: int = 32, device: Optional[torch.device] = None):
        self.max_len = max_len
        # store list of (keys, values) where keys/values are tensors shaped (B, N, C)
        self.keys: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.device = device

    def append(self, keys: torch.Tensor, values: torch.Tensor):
        """
        Append keys and values for a new time step.
        keys/values expected shape: (B, N, C)
        """
        # move to cache device if specified
        if self.device is not None and keys.device != self.device:
            keys = keys.to(self.device)
            values = values.to(self.device)
        self.keys.append(keys.detach())
        self.values.append(values.detach())
        self._prune_if_needed()

    def _prune_if_needed(self):
        # prune oldest if exceed max_len
        while len(self.keys) > self.max_len:
            self.keys.pop(0)
            self.values.pop(0)

    def get_all(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Concatenate stored keys/values along token dimension.
        Returns keys_cat, values_cat with shapes (B, total_N, C)
        or (None, None) if cache empty.
        """
        if len(self.keys) == 0:
            return None, None
        # keys/values are lists of (B, N_i, C) -- need to ensure consistent B and C
        keys_cat = torch.cat(self.keys, dim=1)
        values_cat = torch.cat(self.values, dim=1)
        return keys_cat, values_cat

    def clear(self):
        self.keys = []
        self.values = []

    def prune_to(self, max_len: int):
        """Prune to at most max_len stored frames (FIFO)."""
        self.max_len = max_len
        self._prune_if_needed()

    def __len__(self):
        return len(self.keys)


if __name__ == "__main__":
    import torch
    cache = KeyValueCache(max_len=3)
    for i in range(5):
        k = torch.randn(2, 16, 64)
        v = torch.randn(2, 16, 64)
        cache.append(k, v)
        print("Cached frames:", len(cache))
    kk, vv = cache.get_all()
    print("Concatenated shapes:", None if kk is None else kk.shape)
