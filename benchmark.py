import torch
import time
import json
import os
import numpy as np
import pandas as pd
import pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache
from spec_decode import speculative_decode

def run_baseline_bench(model, tokenizer, input_ids, max_new_tokens=128):
    cache = DynamicCache()
    cur_input = input_ids
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    num_tokens = 0
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(cur_input, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(-1)
            num_tokens += 1
            cur_input = next_token.view(1, 1)
            if next_token == tokenizer.eos_token_id:
                break
    end_time = time.time()
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2
    return num_tokens / (end_time - start_time), peak_vram

def run_spec_bench(target_model, draft_model, tokenizer, input_ids, k, max_new_tokens=128):
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    # Handle generator
    generated = list(speculative_decode(target_model, draft_model, tokenizer, input_ids, max_new_tokens, k))
    end_time = time.time()
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2
    tps = len(generated) / (end_time - start_time)
    return tps, peak_vram

def run_hf_bench(target_model, draft_model, tokenizer, input_ids, max_new_tokens=128):
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    outputs = target_model.generate(
        input_ids,
        assistant_model=draft_model,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    end_time = time.time()
    num_tokens = outputs.shape[-1] - input_ids.shape[-1]
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2
    return num_tokens / (end_time - start_time), peak_vram

def main():
    target_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    draft_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print("Loading models for Benchmark...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(target_model_id)
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_id, quantization_config=bnb_config, device_map="cuda", trust_remote_code=True
    )
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_id, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True
    )
    
    prompt = "Explain how post-training quantization reduces the memory footprint of a transformer model. Be specific about what data structures shrink."
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(target_model.device)
    
    results = []
    
    # 1. Baseline
    print("\nBenchmarking Baseline...")
    tps_list, vram_list = [], []
    for i in range(5):
        tps, vram = run_baseline_bench(target_model, tokenizer, input_ids)
        tps_list.append(tps); vram_list.append(vram)
    mean_baseline = np.mean(tps_list)
    results.append({"variant": "baseline", "k": 0, "tps": mean_baseline, "vram_mb": np.max(vram_list)})
    
    # 2. Custom Spec
    for k in [4, 8]:
        print(f"\nBenchmarking Custom Speculative (k={k})...")
        tps_list, vram_list = [], []
        for i in range(5):
            tps, vram = run_spec_bench(target_model, draft_model, tokenizer, input_ids, k)
            tps_list.append(tps); vram_list.append(vram)
        results.append({"variant": "custom_spec", "k": k, "tps": np.mean(tps_list), "vram_mb": np.max(vram_list)})
        
    df = pd.DataFrame(results)
    df['speedup'] = df['tps'] / mean_baseline
    print("\n" + df.to_string(index=False))

if __name__ == "__main__":
    main()
