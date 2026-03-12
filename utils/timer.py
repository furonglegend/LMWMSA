"""
utils/timer.py

Lightweight timing utilities for benchmarking or profiling code paths:
 - Timer context manager that returns elapsed seconds
 - Simple decorator `timed` to annotate function runtimes

Example:
    with Timer("encode") as t:
        ...
    print(t.elapsed)
"""

import time
from typing import Optional, Callable, Any
from functools import wraps

class Timer:
    def __init__(self, name: Optional[str] = None, silent: bool = False):
        self.name = name
        self.silent = silent
        self.start_ts = None
        self.end_ts = None
        self.elapsed = None

    def __enter__(self):
        self.start_ts = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_ts = time.time()
        self.elapsed = self.end_ts - self.start_ts
        if not self.silent:
            n = f" [{self.name}]" if self.name else ""
            print(f"[TIMER]{n} Elapsed {self.elapsed:.4f} sec")

def timed(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        out = fn(*args, **kwargs)
        t1 = time.time()
        print(f"[TIMED] {fn.__name__} took {(t1 - t0):.4f} sec")
        return out
    return wrapper
