import torch
import time
import json
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from spec_decode import speculative_decode

def main():
    target_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    draft_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print(f"Loading models for advanced speculative decoding demo...")
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
    
    print("\n" + "="*50)
    print("STREAMING SPECULATIVE DECODING (Dynamic K=Enabled)")
    print("="*50)
    
    generated_tokens = []
    start_time = time.time()
    
    # Iterate through the generator
    for token_id in speculative_decode(
        target_model, draft_model, tokenizer, input_ids, 
        max_new_tokens=128, k=4, dynamic_k=True
    ):
        generated_tokens.append(token_id)
        # Stream to console
        text = tokenizer.decode([token_id], skip_special_tokens=True)
        print(text, end="", flush=True)
    
    end_time = time.time()
    print("\n" + "="*50)
    
    total_time = end_time - start_time
    tps = len(generated_tokens) / total_time
    print(f"Total time: {total_time:.2f}s")
    print(f"Tokens/sec: {tps:.2f}")
    
    # Save results for verify.py
    os.makedirs("results", exist_ok=True)
    with open("results/spec_tokens.json", "w") as f:
        json.dump(generated_tokens, f)

if __name__ == "__main__":
    main()
