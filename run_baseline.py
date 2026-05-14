import torch
import time
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache

def main():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    
    print(f"Loading target model for manual baseline: {model_id}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="cuda",
        trust_remote_code=True
    )
    
    prompt = "Explain how post-training quantization reduces the memory footprint of a transformer model. Be specific about what data structures shrink."
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
    generated = []
    cache = DynamicCache()
    cur_input = input_ids
    
    print("Generating manual baseline (greedy loop)...")
    start_time = time.time()
    for _ in range(128):
        with torch.no_grad():
            out = model(cur_input, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(-1)
            generated.append(next_token.item())
            cur_input = next_token.view(1, 1)
            if next_token == tokenizer.eos_token_id:
                break
    end_time = time.time()
    
    total_time = end_time - start_time
    print(f"Total time: {total_time:.2f}s")
    
    os.makedirs("results", exist_ok=True)
    with open("results/baseline_tokens.json", "w") as f:
        json.dump(generated, f)
    print("Saved to results/baseline_tokens.json")

if __name__ == "__main__":
    main()
