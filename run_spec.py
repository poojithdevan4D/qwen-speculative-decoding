import torch
import time
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from spec_decode import speculative_decode

def main():
    target_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    draft_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print(f"Loading target model: {target_model_id} in 4-bit NF4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(target_model_id)
    
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_id,
        quantization_config=bnb_config,
        device_map="cuda",
        trust_remote_code=True
    )
    
    print(f"Loading draft model: {draft_model_id} in FP16...")
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_id,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True
    )
    
    prompt = "Explain how post-training quantization reduces the memory footprint of a transformer model. Be specific about what data structures shrink."
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(target_model.device)
    
    print(f"Running speculative decoding (k=4)...")
    start_time = time.time()
    generated_tokens, metrics = speculative_decode(
        target_model, 
        draft_model, 
        tokenizer, 
        input_ids, 
        max_new_tokens=128, 
        k=4
    )
    end_time = time.time()
    
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    total_time = end_time - start_time
    num_tokens = len(generated_tokens)
    tokens_per_sec = num_tokens / total_time
    
    print("-" * 30)
    print(f"Generated Text:\n{generated_text}")
    print("-" * 30)
    print(f"Total time: {total_time:.2f}s")
    print(f"Tokens/sec: {tokens_per_sec:.2f}")
    print(f"Acceptance Rate: {metrics['acceptance_rate']:.2%}")
    print(f"Total proposed: {metrics['total_proposed']}")
    print(f"Total accepted: {metrics['total_accepted']}")
    print(f"Total steps: {metrics['steps']}")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    results_path = "results/spec_tokens.json"
    with open(results_path, "w") as f:
        json.dump(generated_tokens, f)
    print(f"Saved token IDs to {results_path}")

if __name__ == "__main__":
    main()
