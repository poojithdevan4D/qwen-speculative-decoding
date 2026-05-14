from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print(f"BOS: {tokenizer.bos_token_id}")
print(f"EOS: {tokenizer.eos_token_id}")
print(f"Encoded: {tokenizer.encode('Hello')}")
