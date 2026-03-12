"""
infer/benchmarks.py

Utilities to benchmark inference throughput, latency and memory for LWT models.
Provides:
 - run_throughput: measures images/sec over N frames
 - run_latency: measures per-frame median/95th latency (ms)
 - simple memory snapshot helper (GPU) using torch.cuda.memory_reserved

These are lightweight benchmarking helpers intended for development & reporting.
"""

import time
import torch
import numpy as np
from typing import Callable, Dict, Any, Tuple


def run_throughput(model: Callable, sample_fn: Callable, device: torch.device, warmup: int = 10, steps: int = 100) -> Dict[str, Any]:
    """
    Measure throughput (frames/sec) for a model.

    Args:
      model: callable that accepts a batch tensor and returns output
      sample_fn: callable producing input batch tensors (already on device)
      device: torch.device
      warmup: number of warmup iterations to ignore
      steps: number of measured iterations

    Returns:
      dict with fps (frames/sec), avg_time_ms, std_time_ms
    """
    # warmup
    for _ in range(warmup):
        inp = sample_fn().to(device)
        with torch.no_grad():
            _ = model(inp)

    times = []
    for _ in range(steps):
        inp = sample_fn().to(device)
        t0 = time.time()
        with torch.no_grad():
            _ = model(inp)
            if device.type == "cuda":
                torch.cuda.synchronize()
        t1 = time.time()
        times.append((t1 - t0) * 1000.0)  # ms

    times = np.array(times)
    avg_ms = float(times.mean())
    std_ms = float(times.std())
    fps = 1000.0 / avg_ms
    return {"fps": fps, "avg_ms": avg_ms, "std_ms": std_ms, "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}


def run_latency_per_frame(process_fn: Callable, inputs: list, device: torch.device) -> Dict[str, Any]:
    """
    Measure per-frame latencies by running process_fn(frame_tensor) for each input in list.
    Returns latency statistics.
    """
    latencies = []
    for t, frame in enumerate(inputs):
        inp = frame.to(device)
        t0 = time.time()
        with torch.no_grad():
            _ = process_fn(inp)
            if device.type == "cuda":
                torch.cuda.synchronize()
        t1 = time.time()
        latencies.append((t1 - t0) * 1000.0)
    arr = np.array(latencies)
    return {
        "count": len(arr),
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "std_ms": float(arr.std()),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max())
    }


def gpu_memory_snapshot(device: torch.device = torch.device("cuda")) -> Dict[str, Any]:
    """
    Return approximate GPU memory usage stats (in MB). Works only when CUDA is available.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"error": "CUDA unavailable or device not cuda."}
    idx = device.index if device.index is not None else torch.cuda.current_device()
    reserved = torch.cuda.memory_reserved(idx) / (1024 ** 2)
    allocated = torch.cuda.memory_allocated(idx) / (1024 ** 2)
    max_reserved = torch.cuda.max_memory_reserved(idx) / (1024 ** 2)
    return {"reserved_mb": reserved, "allocated_mb": allocated, "max_reserved_mb": max_reserved}


if __name__ == "__main__":
    # demo: benchmark a trivial model
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            # simulate some work
            return x * 0.5

    model = DummyModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def sample_fn():
        return torch.randn(1, 3, 64, 64).to(device)

    print("Running throughput demo...")
    print(run_throughput(model, sample_fn, device, warmup=2, steps=10))
    print("GPU snapshot:", gpu_memory_snapshot(device))
