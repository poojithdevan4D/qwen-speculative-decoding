import torch
from transformers import DynamicCache

def speculative_decode(target_model, draft_model, tokenizer, input_ids, max_new_tokens=128, k=4, 
                       do_sample=False, temperature=1.0, top_p=1.0):
    """
    Implements speculative decoding with optional probabilistic rejection sampling.
    """
    generated = []
    last_token = None
    
    draft_cache = DynamicCache()
    target_cache = DynamicCache()
    
    step = 0
    total_proposed = 0
    total_accepted = 0
    acceptance_counts = []
    
    while len(generated) < max_new_tokens:
        # ---- DRAFT PHASE ----
        draft_tokens = []
        draft_probs = [] # Store probabilities for rejection sampling
        
        if step == 0:
            draft_input = input_ids
        else:
            draft_input = last_token.view(1, 1)
            
        temp_input = draft_input
        for _ in range(k):
            with torch.no_grad():
                out = draft_model(temp_input, past_key_values=draft_cache, use_cache=True)
                draft_cache = out.past_key_values
                logits = out.logits[:, -1, :]
                
                if do_sample:
                    # Apply temperature
                    logits = logits / max(temperature, 1e-5)
                    # Apply Top-P if needed
                    # (Simplified Top-P for now, or just focus on Rejection Sampling)
                    probs = torch.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                    draft_probs.append(probs)
                else:
                    next_id = logits.argmax(-1, keepdim=True)
                    
                draft_tokens.append(next_id)
                temp_input = next_id.view(1, 1)
        
        draft_seq = torch.cat(draft_tokens, dim=-1).view(1, -1)
        total_proposed += k
        
        # ---- VERIFY PHASE ----
        if step == 0:
            target_input = torch.cat([input_ids, draft_seq], dim=-1)
        else:
            target_input = torch.cat([last_token.view(1, 1), draft_seq], dim=-1)
            
        with torch.no_grad():
            target_out = target_model(target_input, past_key_values=target_cache, use_cache=True)
            target_cache = target_out.past_key_values
            
        target_logits = target_out.logits[:, -(k+1):, :] # (1, K+1, Vocab)
        if do_sample:
            target_logits = target_logits / max(temperature, 1e-5)
            target_probs = torch.softmax(target_logits, dim=-1)
        
        # ---- ACCEPT/REJECT ----
        accepted = 0
        for i in range(k):
            token_id = draft_tokens[i].item()
            if not do_sample:
                # Greedy match
                if target_logits[0, i, :].argmax(-1) == token_id:
                    accepted += 1
                    generated.append(token_id)
                    if len(generated) >= max_new_tokens: break
                else:
                    break
            else:
                # Rejection Sampling
                p = target_probs[0, i, token_id].item()
                q = draft_probs[i][0, token_id].item()
                
                r = torch.rand(1).item()
                if r <= p / q:
                    accepted += 1
                    generated.append(token_id)
                    if len(generated) >= max_new_tokens: break
                else:
                    # Rejected: Correct the distribution for the bonus/next token
                    # p' = norm(max(0, p - q))
                    resample_probs = torch.clamp(target_probs[0, i, :] - draft_probs[i][0, :], min=0)
                    resample_probs = resample_probs / resample_probs.sum()
                    next_token = torch.multinomial(resample_probs.unsqueeze(0), num_samples=1)
                    break
        
        total_accepted += accepted
        acceptance_counts.append(accepted)
        
        # ---- BONUS TOKEN / NEXT STEP PREP ----
        if accepted < k:
            if not do_sample:
                next_token = target_logits[0, accepted, :].argmax(-1, keepdim=True)
            # else: next_token was already set in the rejection loop
            
            if len(generated) < max_new_tokens:
                generated.append(next_token.item())
            
            # Truncate caches
            target_cache.crop(target_cache.get_seq_length() - (k - accepted))
            draft_cache.crop(draft_cache.get_seq_length() - max(0, k - 1 - accepted))
        else:
            # All K accepted, get the bonus token from the last target logit
            if not do_sample:
                next_token = target_logits[0, k, :].argmax(-1, keepdim=True)
            else:
                next_token = torch.multinomial(target_probs[0, k, :].unsqueeze(0), num_samples=1)
                
            if len(generated) < max_new_tokens:
                generated.append(next_token.item())
                
            # Advance draft cache with the last accepted token
            with torch.no_grad():
                draft_model(draft_tokens[-1].view(1, 1), past_key_values=draft_cache, use_cache=True)
            
        last_token = next_token
        step += 1
        if next_token == tokenizer.eos_token_id: break
            
    return generated, {
        "total_proposed": total_proposed, "total_accepted": total_accepted,
        "acceptance_rate": total_accepted / total_proposed if total_proposed > 0 else 0,
        "steps": step, "acceptance_history": acceptance_counts
    }
