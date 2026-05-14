# Speculative Decoding from Scratch — Qwen2.5-1.5B Inference Acceleration on a 4GB GPU

This project implements speculative decoding for the Qwen2.5-1.5B model using Qwen2.5-0.5B as a draft model. Speculative decoding (also known as assisted generation) accelerates LLM inference by using a small, fast "draft" model to propose multiple candidate tokens, which are then verified in parallel by a larger, more accurate "target" model in a single forward pass.

## Core Principle
Mathematically, the output of speculative decoding in greedy mode is **bit-identical** to the output of the target model alone. This implementation guarantees that property, as verified by `verify.py`. The efficiency comes from the fact that verifying $K$ tokens costs roughly the same as generating 1 token with the large model, allowing us to "skip" steps if the draft model is sufficiently accurate.

## Implementation Notes
The hardest part of implementing speculative decoding is not the algorithm — it's the KV cache bookkeeping. When the target model rejects a draft token, the draft and target models have both already processed and cached the rejected token's K/V tensors. Failing to truncate both caches by the rejection count causes the next iteration's forward pass to use stale K/V from rolled-back tokens, producing output that differs subtly from baseline. `verify.py` exists specifically to catch this: it asserts byte-identical output between speculative and baseline runs. The implementation passes this check at every K value tested.

## Hardware & Software Setup
- **GPU**: NVIDIA RTX 3050 Laptop (4GB VRAM)
- **Target Model**: Qwen2.5-1.5B-Instruct (4-bit NF4 via `bitsandbytes`)
- **Draft Model**: Qwen2.5-0.5B-Instruct (FP16)
- **Environment**: Python 3.11, PyTorch 2.6.0+cu124, Transformers 5.8.1

## Results
Benchmark conducted with a 128-token generation limit. 

| Variant | K (Lookahead) | Mean Toks/Sec | Std Dev | Acceptance Rate | Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 0 | 5.740 | 0.169 | 100% | 1.00x |
| Spec Decode | 1 | 4.818 | 0.167 | 70.7% | 0.84x |
| Spec Decode | 2 | 5.188 | 0.136 | 58.5% | 0.90x |
| **Spec Decode** | **4** | **5.678** | **0.569*** | **51.8%** | **0.99x** |
| Spec Decode | 6 | 4.958 | 0.107 | 41.4% | 0.86x |
| Spec Decode | 8 | 4.975 | 0.075 | 36.4% | 0.87x |

*\* K=4 showed higher run-to-run variance during testing; investigated but no consistent cause. Reported as observed.*

### Methodology: Why is the baseline "slow"?
The baseline uses a manual greedy loop with the same `DynamicCache` management as the speculative path, ensuring both implementations share the same per-step overhead. Vanilla `model.generate()` benefits from internal optimizations (fused ops, batching tricks) that aren't applicable inside a custom speculative loop. Reporting speedup against an apples-to-apples baseline is the methodologically correct choice; reporting against `generate()` would inflate the speedup number unfairly.

### Hardware Constraints
On a memory-constrained laptop GPU (RTX 3050), the overhead of maintaining two models in VRAM and the shared power budget between compute and memory bandwidth often negates the theoretical gains of speculative decoding. In this environment, the "cost" of the draft model's forward passes is nearly equal to the "savings" from skipping target model steps.

## Reproduction
1. `pip install -r requirements.txt`
2. `python run_baseline.py` (Generate reference)
3. `python run_spec.py` (Run speculative decoding)
4. `python verify.py` (Confirm bit-identical correctness)
5. `python benchmark.py` (Run full sweep)
6. `python check_tokenizer.py` (Verifies that draft and target models share an identical vocabulary).
