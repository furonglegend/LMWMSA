# Our method — Controllable Video Super-Resolution (Code Skeleton)

This repository contains a structured code skeleton to implement **Our method**, a motion-aware, controllable Latent World Transformer (LWT) for video super-resolution. The code is organized to support:

- Full curriculum training (dense teacher → sparse-causal adaptation → student distillation).
- Low-cost sparse-causal LWT direct training.
- Streaming and batch inference with a short-term key-value cache.
- Motion reliability estimation (confidence), residual correction, and motion fusion.
- Adaptive block-level sparse attention with soft top-k training and hard top-k inference.
- User controls: global scalar (`gamma`) and spatial gate (`g_t`).

> NOTE: This repo provides readable, extensible skeletons for each module. Many modules include placeholders / stubs for easy integration. Replace the stubs with your project's encoder/teacher/decoder and loss implementations.

## Repo layout (high-level)
