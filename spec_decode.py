import torch
from transformers import DynamicCache

def speculative_decode(target_model, draft_model, tokenizer, input_ids, max_new_tokens=128, k=4):
    """
    Implements greedy speculative decoding.
    
    Args:
        target_model: The large 'oracle' model (M_t)
        draft_model: The small 'draft' model (M_d)
        tokenizer: Shared tokenizer
        input_ids: Prompt token IDs (1, seq_len)
        max_new_tokens: Budget for generation
        k: Lookahead count (number of draft tokens per step)
        
    Returns:
        generated_ids: List of newly generated token IDs
        metrics: Dictionary containing acceptance statistics
    """
    
    generated = []
    last_token = None
    
    # Initialize DynamicCaches
    draft_cache = DynamicCache()
    target_cache = DynamicCache()
    
    # Initial prompt processing for target model to fill cache
    # But wait, the algorithm says:
    # "Target processes (prompt or last_token) + all K drafts in ONE pass"
    # If step 0, target_input = cat([input_ids, draft_seq], dim=-1)
    
    step = 0
    total_proposed = 0
    total_accepted = 0
    acceptance_counts = [] # Track how many accepted per step
    
    cur_input_ids = input_ids # Start with full prompt
    
    while len(generated) < max_new_tokens:
        # ---- DRAFT PHASE ----
        draft_tokens = []
        # Draft model takes either the full prompt (step 0) or the last accepted/bonus token
        if step == 0:
            draft_input = input_ids
        else:
            draft_input = last_token.view(1, 1)
            
        # Propose K tokens
        temp_input = draft_input
        for _ in range(k):
            with torch.no_grad():
                out = draft_model(temp_input, past_key_values=draft_cache, use_cache=True)
                draft_cache = out.past_key_values
                next_id = out.logits[:, -1, :].argmax(-1) # shape (1,)
                draft_tokens.append(next_id)
                temp_input = next_id.view(1, 1)
        
        draft_seq = torch.cat(draft_tokens, dim=-1).view(1, -1) # Ensure (1, K)
        total_proposed += k
        
        # ---- VERIFY PHASE ----
        # Target processes (prompt or last_token) + all K drafts in ONE pass
        if step == 0:
            target_input = torch.cat([input_ids, draft_seq], dim=-1)
        else:
            target_input = torch.cat([last_token.view(1, 1), draft_seq], dim=-1)
            
        with torch.no_grad():
            target_out = target_model(target_input, past_key_values=target_cache, use_cache=True)
            target_cache = target_out.past_key_values
            
        # Take logits at the last K+1 positions
        target_argmax = target_out.logits[:, -(k+1):, :].argmax(-1) # (1, K+1)
        
        # ---- ACCEPT/REJECT ----
        accepted = 0
        for i in range(k):
            if target_argmax[0, i] == draft_tokens[i][0]:
                accepted += 1
                generated.append(draft_tokens[i].item())
                if len(generated) >= max_new_tokens:
                    break
            else:
                break
        
        # DEBUG
        # print(f"Step {step}: Proposed {k}, Accepted {accepted}, Cache L (T/D): {target_cache.get_seq_length()}/{draft_cache.get_seq_length()}")
        
        total_accepted += accepted
        acceptance_counts.append(accepted)
        
        # Last token and bonus handling
        if accepted < k:
            next_token = target_argmax[0, accepted]
            if len(generated) < max_new_tokens:
                generated.append(next_token.item())
            
            # Truncate caches
            n_drop_target = k - accepted
            target_cache.crop(target_cache.get_seq_length() - n_drop_target)
            
            # Draft cache truncation: it grew by initial_input_len + k - 1
            # We want to drop all unaccepted draft tokens.
            n_drop_draft = max(0, k - 1 - accepted)
            draft_cache.crop(draft_cache.get_seq_length() - n_drop_draft)
        else:
            next_token = target_argmax[0, k]
            if len(generated) < max_new_tokens:
                generated.append(next_token.item())
            # If we accepted all K, we might want to update draft_cache with the last accepted token dK
            # so it's ready for the bonus token in the next step.
            with torch.no_grad():
                draft_model(draft_tokens[-1].view(1, 1), past_key_values=draft_cache, use_cache=True)
            
        last_token = next_token
        step += 1
        
        # Check EOS
        if next_token == tokenizer.eos_token_id:
            break
            
    metrics = {
        "total_proposed": total_proposed,
        "total_accepted": total_accepted,
        "acceptance_rate": total_accepted / total_proposed if total_proposed > 0 else 0,
        "steps": step,
        "acceptance_history": acceptance_counts
    }
    
    return generated, metrics
