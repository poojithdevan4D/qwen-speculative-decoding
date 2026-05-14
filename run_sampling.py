import torch
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from spec_decode import speculative_decode

def main():
    target_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    draft_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print("Loading models for sampling test...")
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
    
    prompt = "Write a creative short story about a cat who discovers a secret portal in a cardboard box."
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(target_model.device)
    
    print("\nRunning speculative decoding with sampling (T=0.7, k=4)...")
    start_time = time.time()
    generated_tokens, metrics = speculative_decode(
        target_model, 
        draft_model, 
        tokenizer, 
        input_ids, 
        max_new_tokens=128, 
        k=4,
        do_sample=True,
        temperature=0.7
    )
    end_time = time.time()
    
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print("-" * 30)
    print("Generated Text:")
    print(output_text)
    print("-" * 30)
    
    total_time = end_time - start_time
    tps = len(generated_tokens) / total_time
    print(f"Total time: {total_time:.2f}s")
    print(f"Tokens/sec: {tps:.2f}")
    print(f"Acceptance Rate: {metrics['acceptance_rate']:.2%}")

if __name__ == "__main__":
    main()
