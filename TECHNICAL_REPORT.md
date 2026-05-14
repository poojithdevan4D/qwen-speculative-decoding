# Technical Report: Speculative Decoding for Qwen2.5
**Project Overview**: A from-scratch implementation of the "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al.) paper, optimized for consumer-grade hardware (NVIDIA RTX 3050 4GB).

---

## Executive Summary
This project demonstrates a bit-identical speculative decoding pipeline for the Qwen2.5-1.5B (Target) and Qwen2.5-0.5B (Draft) models. By leveraging 4-bit NF4 quantization for the target model and a custom inference loop with `DynamicCache` management, we achieved a peak speedup of **2.0x** in greedy mode and enabled real-time streaming with adaptive lookahead.

---

## Day-by-Day Development Journey

### Day 1: Foundation & Correctness
*   **Environment Setup**: Established a CUDA-hardened environment using PyTorch 2.6 and `bitsandbytes` 4-bit quantization.
*   **Model Architecture**: Mapped a 1.5B Target model (NF4) and a 0.5B Draft model (FP16) into 4GB of VRAM, leaving ~1.5GB headroom for KV caches.
*   **The KV Cache Trap**: Discovered that naive speculative decoding causes token divergence. Identified the root cause as "stale KV cache"—where rejected tokens remain in the K/V tensors.
*   **Verification**: Implemented `verify.py` to assert bit-identicality between speculative and baseline outputs. Fixed the issue by implementing precise `DynamicCache.crop()` logic for both models.

### Day 2: Polish & Performance Telemetry
*   **Senior Review Alignment**: Refined the benchmarking methodology. Explained why reporting speedup against a manual greedy baseline is more scientifically honest than comparing against optimized library-level `model.generate()`.
*   **Visualization**: Built a Matplotlib-based charting suite to map the "Inverted-U" curve of speedup across lookahead values ($K \in \{1, 2, 4, 6, 8\}$).
*   **VRAM Telemetry**: Integrated `pynvml` to track peak memory usage, confirming a tight ~2.1GB footprint.
*   **Competitive Analysis**: Benchmarked against HuggingFace's `assisted_generation`, finding our implementation to be within 16% of production-grade library performance.

### Day 3: Probabilistic Rejection Sampling
*   **Algorithmic Expansion**: Moved beyond greedy decoding to support creative generation.
*   **Rejection Sampling**: Implemented the $p(x)/q(x)$ acceptance ratio logic. This ensures that even when draft tokens are sampled randomly, the final output follows the target model's distribution exactly.
*   **Correction Distribution**: Implemented the mathematical resample-correction: $p'(x) = \text{norm}(\max(0, p(x) - q(x)))$ for rejected tokens.

### Day 4+: Advanced Engine Features
*   **Dynamic K (Adaptive Lookahead)**: Implemented an auto-tuning controller. The engine now monitors acceptance rates in real-time—increasing $K$ during predictable sequences and throttling $K$ back during high-entropy generation to save power and bandwidth.
*   **Streaming API**: Refactored the core engine into a Python Generator. This enables "Time-to-First-Token" (TTFT) optimization and real-time UI updates, which are critical for interactive LLM applications.

---

## Key Technical Achievements

| Feature | Implementation Detail |
| :--- | :--- |
| **Quantization** | 4-bit NF4 for Target, FP16 for Draft (VRAM efficient) |
| **Correctness** | Guaranteed bit-identical greedy output |
| **Lookahead** | Dynamic/Adaptive $K$ (1 to 16) |
| **Sampling** | Rejection Sampling with $p/q$ correction |
| **Streaming** | Generator-based real-time yielding |

---

## Technical Challenges & Solutions

### 1. The Divergence Problem
**Issue**: Speculative output differed from baseline at the 5th token.
**Solution**: Standardized the baseline to use the exact same manual loop and `DynamicCache` management as the speculative path. This proved that the divergence was due to off-by-one errors in cache truncation, not model math.

### 2. Bandwidth Bottlenecking
**Issue**: On a 3050 Laptop, the memory bandwidth is limited. Running two models simultaneously can actually be slower than one.
**Solution**: Optimized the draft phase to be extremely lightweight and implemented Dynamic K to ensure we don't waste bandwidth on draft tokens that are likely to be rejected.

---

## Final Results
*   **Optimal K**: 6 (in initial tests) or 8 (in stabilized tests).
*   **Peak Speedup**: 2.0x (Greedy), 1.1x (Sampling).
*   **VRAM Usage**: 2.1 GB.
*   **Repository**: [github.com/poojithdevan4D/qwen-speculative-decoding](https://github.com/poojithdevan4D/qwen-speculative-decoding)
