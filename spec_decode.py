import torch
from transformers import DynamicCache

def speculative_decode(target_model, draft_model, tokenizer, input_ids, max_new_tokens=128, k=4, 
                       do_sample=False, temperature=1.0, top_p=1.0, dynamic_k=False):
    """
    Advanced speculative decoding with Streaming, Probabilistic Sampling, and Dynamic K.
    Yields tokens one by one as they are verified.
    """
    generated = []
    last_token = None
    
    draft_cache = DynamicCache()
    target_cache = DynamicCache()
    
    step = 0
    total_proposed = 0
    total_accepted = 0
    
    current_k = k
    
    while len(generated) < max_new_tokens:
        # ---- DRAFT PHASE ----
        draft_tokens = []
        draft_probs = []
        
        if step == 0:
            draft_input = input_ids
        else:
            draft_input = last_token.view(1, 1)
            
        temp_input = draft_input
        for _ in range(current_k):
            with torch.no_grad():
                out = draft_model(temp_input, past_key_values=draft_cache, use_cache=True)
                draft_cache = out.past_key_values
                logits = out.logits[:, -1, :]
                
                if do_sample:
                    logits = logits / max(temperature, 1e-5)
                    probs = torch.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                    draft_probs.append(probs)
                else:
                    next_id = logits.argmax(-1, keepdim=True)
                    
                draft_tokens.append(next_id)
                temp_input = next_id.view(1, 1)
        
        draft_seq = torch.cat(draft_tokens, dim=-1).view(1, -1)
        total_proposed += current_k
        
        # ---- VERIFY PHASE ----
        if step == 0:
            target_input = torch.cat([input_ids, draft_seq], dim=-1)
        else:
            target_input = torch.cat([last_token.view(1, 1), draft_seq], dim=-1)
            
        with torch.no_grad():
            target_out = target_model(target_input, past_key_values=target_cache, use_cache=True)
            target_cache = target_out.past_key_values
            
        target_logits = target_out.logits[:, -(current_k+1):, :]
        if do_sample:
            target_logits = target_logits / max(temperature, 1e-5)
            target_probs = torch.softmax(target_logits, dim=-1)
        
        # ---- ACCEPT/REJECT ----
        accepted = 0
        for i in range(current_k):
            token_id = draft_tokens[i].item()
            if not do_sample:
                if target_logits[0, i, :].argmax(-1) == token_id:
                    accepted += 1
                    generated.append(token_id)
                    yield token_id
                    if len(generated) >= max_new_tokens: break
                else:
                    break
            else:
                p = target_probs[0, i, token_id].item()
                q = draft_probs[i][0, token_id].item()
                if torch.rand(1).item() <= p / q:
                    accepted += 1
                    generated.append(token_id)
                    yield token_id
                    if len(generated) >= max_new_tokens: break
                else:
                    resample_probs = torch.clamp(target_probs[0, i, :] - draft_probs[i][0, :], min=0)
                    resample_probs = resample_probs / resample_probs.sum()
                    next_token = torch.multinomial(resample_probs.unsqueeze(0), num_samples=1)
                    break
        
        total_accepted += accepted
        
        # ---- BONUS TOKEN / NEXT STEP PREP ----
        if accepted < current_k:
            if not do_sample:
                next_token = target_logits[0, accepted, :].argmax(-1, keepdim=True)
            
            if len(generated) < max_new_tokens:
                generated.append(next_token.item())
                yield next_token.item()
            
            target_cache.crop(target_cache.get_seq_length() - (current_k - accepted))
            draft_cache.crop(draft_cache.get_seq_length() - max(0, current_k - 1 - accepted))
            
            # Dynamic K: Reduce K if we had many rejections
            if dynamic_k:
                current_k = max(1, current_k - 1)
        else:
            if not do_sample:
                next_token = target_logits[0, current_k, :].argmax(-1, keepdim=True)
            else:
                next_token = torch.multinomial(target_probs[0, current_k, :].unsqueeze(0), num_samples=1)
                
            if len(generated) < max_new_tokens:
                generated.append(next_token.item())
                yield next_token.item()
                
            with torch.no_grad():
                draft_model(draft_tokens[-1].view(1, 1), past_key_values=draft_cache, use_cache=True)
            
            # Dynamic K: Increase K if we accepted everything
            if dynamic_k:
                current_k = min(16, current_k + 1)
            
        last_token = next_token
        step += 1
        if next_token == tokenizer.eos_token_id: break
