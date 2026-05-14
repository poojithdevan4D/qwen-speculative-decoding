# Speculative Decoding from Scratch — Qwen2.5-1.5B Inference Acceleration on a 4GB GPU

This project implements speculative decoding for the Qwen2.5-1.5B model using Qwen2.5-0.5B as a draft model. Speculative decoding (also known as assisted generation) accelerates LLM inference by using a small, fast "draft" model to propose multiple candidate tokens, which are then verified in parallel by a larger, more accurate "target" model in a single forward pass.

## Core Principle
Mathematically, the output of speculative decoding in greedy mode is **bit-identical** to the output of the target model alone. The efficiency comes from the fact that verifying $K$ tokens costs roughly the same as generating 1 token with the large model—*theoretically*. 

This project explores the **implementation gap** between this theoretical gain and the practical constraints of consumer-grade hardware.

![Speedup vs K](docs/speedup_vs_k.png)

## Implementation Notes: Theory vs. Reality
The hardest part of implementing speculative decoding is the KV cache bookkeeping. While our implementation successfully handles the complex truncation logic and passes bit-identical verification (`verify.py`), the performance on an RTX 3050 Laptop highlights a common "Inference Trap":

*   **Theoretical Gain**: One $K=4$ verification pass should take $\approx 1$ baseline step.
*   **Practical Reality**: In vanilla PyTorch, every operation in the verification pass (attention, linear layers, etc.) launches a separate CUDA kernel. The overhead of these launches, combined with memory bandwidth sharing between two models, can negate the "skipping" benefit on consumer hardware.

For a deeper write-up of these design decisions and a day-by-day journey, see [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).

## Worked Example: The Mechanism
Consider an iteration at $K=4$:
1. **Drafting**: The fast 0.5B model generates 4 candidate tokens.
2. **Verification**: The 1.5B target model runs a single pass on the prefix + all 4 drafts.
3. **Acceptance**: If the target's argmax matches the draft at index $i$, we keep it. 
4. **Correction**: If they diverge, we truncate the cache and restart from the target's prediction.

While this mechanism reduces the total number of large-model forward passes, the **per-pass latency** of the speculative loop is higher than a baseline step due to kernel launch overhead and draft phase costs.

## Results & Analysis
Benchmark conducted on an **RTX 3050 Laptop (4GB)**.

| Variant | K | Mean Toks/Sec | Std Dev | Peak VRAM | Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 0 | 5.953 | 0.165 | 2105 MB | 1.00x |
| Custom Spec | 4 | 5.565 | 0.451 | 2114 MB | 0.93x |
| **Custom Spec** | **8** | **6.355** | **0.406** | **2116 MB** | **1.07x** |
| **HF Assisted** | auto | **6.926** | **1.024** | **2136 MB** | **1.16x** |

### Why does HuggingFace win?
HuggingFace's `assisted_generation` outperforms our from-scratch implementation by ~16% (and 20% over baseline). This gap is a high-signal metric for inference engineering:
1.  **CUDA Graphs**: Production libraries use `torch.cuda.CUDAGraph` to record the entire verification pass. This eliminates CPU-to-GPU launch overhead, which is the primary bottleneck in our custom loop.
2.  **Kernel Fusion**: Optimized libraries fuse multiple operations (e.g., Softmax + Scale) into a single GPU kernel, reducing memory round-trips.

## Advanced Features
Despite the hardware-induced performance floor, this engine implements several advanced features for modern LLM deployment:
*   **Streaming Support**: Implemented as a generator for real-time UI yielding.
*   **Dynamic K**: Self-tuning lookahead that adapts $K$ based on empirical acceptance rates.
*   **Rejection Sampling**: Mathematically consistent stochastic sampling for creative tasks (`run_sampling.py`).

## Reproduction
1. `pip install -r requirements.txt`
2. `python run_baseline.py` (Generate reference)
3. `python run_spec.py` (Run **Streaming + Dynamic K** demo)
4. `python benchmark.py` (Run full sweep)
5. `python visualize.py` (Generate performance chart)
