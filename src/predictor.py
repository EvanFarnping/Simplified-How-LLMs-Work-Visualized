import torch
import torch.nn.functional as F

class Predictor:
    """
    Responsible for Next-Token Prediction logic.
    Updated with Robust Stop Logic for Chat Models.
    """
    
    @staticmethod
    def get_top_k_tokens(model, tokenizer, text, k=10):
        """
        Predicts the NEXT token probabilities.
        """
        device = next(model.parameters()).device
        
        should_prepend_bos = True
        if tokenizer.bos_token and text.strip().startswith(tokenizer.bos_token):
            should_prepend_bos = False
            
        if hasattr(model, "tokenizer"): # TransformerLens
            inputs = model.to_tokens(text, prepend_bos=should_prepend_bos)
        else: # Standard HuggingFace Model
            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=should_prepend_bos).to(device)

        return Predictor._get_logits_and_probs(model, tokenizer, inputs, k, past_key_values=None)

    @staticmethod
    def predict_sequence(model, tokenizer, text, steps=5, k=5, temperature=0.0):
        """
        Runs a loop to predict the next N tokens.
        Uses KV Caching, Sampling, and SMART STOP LOGIC.
        """
        device = next(model.parameters()).device
        
        stop_ids = {tokenizer.eos_token_id}
        if tokenizer.pad_token_id is not None:
            stop_ids.add(tokenizer.pad_token_id)
            
        # Common end special tokens for Models (Phi-4, Llama, Chat, etc.)
        potential_stops = ["<|end|>", "<|im_end|>", "<|eot_id|>", "</s>"]
        for s in potential_stops:
            test_ids = tokenizer.encode(s, add_special_tokens=False)
            if len(test_ids) == 1:
                stop_ids.add(test_ids[0])

        should_prepend_bos = True
        if tokenizer.bos_token and text.strip().startswith(tokenizer.bos_token):
            should_prepend_bos = False

        if hasattr(model, "tokenizer"):
            inputs = model.to_tokens(text, prepend_bos=should_prepend_bos)
        else: 
            inputs = tokenizer(text, return_tensors="pt", 
                               add_special_tokens=should_prepend_bos).to(device)

        history = []
        past_key_values = None 
        print(f"   ...Generating {steps} steps...")
        # print(f"   ...Generating {steps} steps (Listening for stops: {stop_ids})...") For debug purposes.

        sampling_k = max(k, 50) 
        for _ in range(steps):
            top_tokens, top_probs, top_indices, past_key_values = Predictor._get_logits_and_probs(
                model, tokenizer, inputs, sampling_k,
                return_indices=True, 
                past_key_values=past_key_values,
                temperature=temperature 
            )

            # Sampling Logic
            if temperature > 1e-5: 
                safe_probs = top_probs / top_probs.sum()
                winner_idx_in_pool = torch.multinomial(torch.tensor(safe_probs), 1).item()
            else:
                winner_idx_in_pool = 0
            
            winner_token = top_tokens[winner_idx_in_pool]
            winner_prob = top_probs[winner_idx_in_pool]
            winner_real_index = top_indices[winner_idx_in_pool].item()

            visual_candidates = list(zip(top_tokens[:k], top_probs[:k]))
            winner_is_visible = any(token == winner_token for token, prob in visual_candidates)

            if not winner_is_visible:
                visual_candidates[-1] = (winner_token, winner_prob)

            step_data = {
                "chosen_token": winner_token,
                "chosen_prob": winner_prob,
                "candidates": visual_candidates
            }
            history.append(step_data)

            if winner_real_index in stop_ids:
                break
            
            # Prepare next input
            next_token_tensor = top_indices[winner_idx_in_pool].unsqueeze(0).unsqueeze(0).to(device)
            use_optimized_path = (past_key_values is not None)

            if hasattr(model, "run_with_cache"):
                inputs = torch.cat([inputs, next_token_tensor], dim=1)
                past_key_values = None 

            else:
                if use_optimized_path:
                    if isinstance(inputs, torch.Tensor):
                        inputs = next_token_tensor 
                    elif hasattr(inputs, "input_ids"):
                        if "attention_mask" in inputs:
                            new_mask_bit = torch.ones((1, 1), device=device, dtype=inputs["attention_mask"].dtype)
                            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], new_mask_bit], dim=1)
                        inputs["input_ids"] = next_token_tensor
                else:
                    if isinstance(inputs, torch.Tensor):
                        inputs = torch.cat([inputs, next_token_tensor], dim=1)
                    elif hasattr(inputs, "input_ids"):
                        inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token_tensor], dim=1)
                        if "attention_mask" in inputs:
                            new_mask_bit = torch.ones((1, 1), device=device, dtype=inputs["attention_mask"].dtype)
                            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], new_mask_bit], dim=1)

        return history

    @staticmethod
    def _get_logits_and_probs(model, 
                              tokenizer, 
                              inputs, k, 
                              return_indices=False, 
                              past_key_values=None, 
                              temperature=0.0):
        
        new_past_key_values = None

        with torch.no_grad():
            if hasattr(model, "run_with_cache"):
                logits = model(inputs) 
                new_past_key_values = None 

            else:
                use_cache_setting = True
                if hasattr(model, "config") and hasattr(model.config, "use_cache"):
                    use_cache_setting = model.config.use_cache

                if isinstance(inputs, torch.Tensor):
                    outputs = model(inputs, past_key_values=past_key_values, 
                                    use_cache=use_cache_setting)
                else:
                    outputs = model(**inputs, past_key_values=past_key_values, 
                                    use_cache=use_cache_setting)
                
                logits = outputs.logits
                new_past_key_values = outputs.past_key_values
        
        next_token_logits = logits[0, -1, :] 

        if temperature > 1e-5:
            next_token_logits = next_token_logits / temperature

        probs = torch.softmax(next_token_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k)
        
        top_tokens = tokenizer.convert_ids_to_tokens(top_indices.tolist())
        top_probs = top_probs.float().cpu().numpy()
        
        if return_indices:
            return top_tokens, top_probs, top_indices, new_past_key_values
        
        return top_tokens, top_probs