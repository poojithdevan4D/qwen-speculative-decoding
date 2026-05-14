import torch
import time
import json
import os
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache
from spec_decode import speculative_decode

def run_baseline_bench(model, tokenizer, input_ids, max_new_tokens=128):
    cache = DynamicCache()
    cur_input = input_ids
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
    return num_tokens / (end_time - start_time)

def run_spec_bench(target_model, draft_model, tokenizer, input_ids, k, max_new_tokens=128):
    start_time = time.time()
    generated, metrics = speculative_decode(target_model, draft_model, tokenizer, input_ids, max_new_tokens, k)
    end_time = time.time()
    tps = len(generated) / (end_time - start_time)
    return tps, metrics['acceptance_rate']

def main():
    target_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    draft_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print("Loading models for benchmarking...")
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
    
    # Baseline
    print("\nBenchmarking Baseline (Manual Greedy)...")
    tps_list = []
    for i in range(5):
        tps = run_baseline_bench(target_model, tokenizer, input_ids)
        tps_list.append(tps)
        print(f"  Run {i+1}: {tps:.2f} tokens/sec")
    
    mean_tps = np.mean(tps_list)
    std_tps = np.std(tps_list)
    results.append({
        "variant": "baseline",
        "k": 0,
        "mean_toks_per_sec": mean_tps,
        "std": std_tps,
        "acceptance_rate": 1.0,
        "speedup": 1.0
    })
    
    # Speculative
    for k in [1, 2, 4, 6, 8]:
        print(f"\nBenchmarking Speculative (k={k})...")
        tps_list = []
        acc_list = []
        for i in range(5):
            tps, acc = run_spec_bench(target_model, draft_model, tokenizer, input_ids, k)
            tps_list.append(tps)
            acc_list.append(acc)
            print(f"  Run {i+1}: {tps:.2f} tokens/sec, Acc: {acc:.2%}")
        
        m_tps = np.mean(tps_list)
        s_tps = np.std(tps_list)
        m_acc = np.mean(acc_list)
        results.append({
            "variant": "spec_decode",
            "k": k,
            "mean_toks_per_sec": m_tps,
            "std": s_tps,
            "acceptance_rate": m_acc,
            "speedup": m_tps / mean_tps
        })
        
    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/benchmark.csv", index=False)
    
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    print(df.to_string(index=False))
    print("="*50)

if __name__ == "__main__":
    main()
