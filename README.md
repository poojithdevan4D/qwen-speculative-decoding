# Speculative Decoding from Scratch — Qwen2.5-1.5B Inference Acceleration on a 4GB GPU

This project implements speculative decoding for the Qwen2.5-1.5B model using Qwen2.5-0.5B as a draft model. Speculative decoding (also known as assisted generation) accelerates LLM inference by using a small, fast "draft" model to propose multiple candidate tokens, which are then verified in parallel by a larger, more accurate "target" model in a single forward pass.

## Core Principle
Mathematically, the output of speculative decoding in greedy mode is **bit-identical** to the output of the target model alone. This implementation guarantees that property, as verified by `verify.py`. The efficiency comes from the fact that verifying $K$ tokens costs roughly the same as generating 1 token with the large model, allowing us to "skip" steps if the draft model is sufficiently accurate.

## Hardware & Software Setup
- **GPU**: NVIDIA RTX 3050 Laptop (4GB VRAM)
- **Target Model**: Qwen2.5-1.5B-Instruct (4-bit NF4 via `bitsandbytes`)
- **Draft Model**: Qwen2.5-0.5B-Instruct (FP16)
- **Environment**: Python 3.11, PyTorch 2.6.0+cu124, Transformers 5.8.1

## Results
Benchmark conducted with a 128-token generation limit. Baseline is a manual greedy loop using `DynamicCache` for fair comparison.

| Variant | K (Lookahead) | Mean Toks/Sec | Std Dev | Acceptance Rate | Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 0 | 2.57 | 0.26 | 100% | 1.00x |
| Spec Decode | 1 | 2.36 | 0.33 | 70.7% | 0.92x |
| Spec Decode | 2 | 2.97 | 0.27 | 58.5% | 1.16x |
| Spec Decode | 4 | 3.02 | 1.12 | 51.8% | 1.18x |
| **Spec Decode** | **6** | **5.15** | **0.17** | **41.4%** | **2.00x** |
| Spec Decode | 8 | 4.74 | 0.19 | 36.4% | 1.85x |

### Acceptance Rate Analysis
The acceptance rate naturally declines as $K$ increases, as the draft model is less likely to correctly predict long sequences of tokens. However, the throughput peaks at **K=6**, where we achieve a **2.00x speedup** over baseline greedy decoding. At $K=8$, the overhead of the draft phase and the cost of rejections start to outweigh the parallelization gains.

## Production Considerations
- **Where it helps**: Low-latency requirements on consumer hardware, high-acceptance scenarios (e.g., code generation, predictable prose).
- **Where it fails**: High-entropy generation (e.g., creative writing) where acceptance rate drops significantly, or when the draft model is too large relative to the target model.

## References
- Leviathan et al., "Fast Inference from Transformers via Speculative Decoding", 2023. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)

## Reproduction
1. `pip install -r requirements.txt`
2. `python run_baseline.py` (Generate reference)
3. `python run_spec.py` (Run speculative decoding)
4. `python verify.py` (Confirm bit-identical correctness)
5. `python benchmark.py` (Run full sweep)
