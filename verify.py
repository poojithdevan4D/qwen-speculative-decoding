import json
import os

def main():
    baseline_path = "results/baseline_tokens.json"
    spec_path = "results/spec_tokens.json"
    
    if not os.path.exists(baseline_path):
        print(f"Error: {baseline_path} not found.")
        return
    if not os.path.exists(spec_path):
        print(f"Error: {spec_path} not found.")
        return
        
    with open(baseline_path, "r") as f:
        baseline_tokens = json.load(f)
    with open(spec_path, "r") as f:
        spec_tokens = json.load(f)
        
    print(f"Baseline tokens: {len(baseline_tokens)}")
    print(f"Speculative tokens: {len(spec_tokens)}")
    
    # Trim to common length if needed, but they should be same if max_new_tokens is 128
    # However, one might hit EOS earlier.
    common_len = min(len(baseline_tokens), len(spec_tokens))
    
    match = True
    for i in range(common_len):
        if baseline_tokens[i] != spec_tokens[i]:
            print(f"Divergence at index {i}:")
            print(f"  Baseline: {baseline_tokens[i]}")
            print(f"  Spec:     {spec_tokens[i]}")
            match = False
            break
            
    if match and len(baseline_tokens) == len(spec_tokens):
        print("SUCCESS: Speculative decoding output is BIT-IDENTICAL to baseline greedy.")
    elif match:
        print("PARTIAL SUCCESS: Tokens match up to common length, but total lengths differ.")
        print(f"  Baseline length: {len(baseline_tokens)}")
        print(f"  Spec length:     {len(spec_tokens)}")
    else:
        print("FAILURE: Speculative decoding output differs from baseline.")

if __name__ == "__main__":
    main()
