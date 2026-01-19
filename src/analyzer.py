import torch

# ============================================================================== #
#                DO NOT TOUCH UNLESS YOU KNOW WHAT YOU ARE DOING!                #
# ============================================================================== #

class BrainAnalyzer:
    """
    Responsible for extracting internal Attention Patterns or Hidden States.
    Compatible with both Standard HuggingFace Models and TransformerLens.
    """
    
    @staticmethod
    def get_attention_map(model, tokenizer, text, device):
        """
        Runs the text and captures the Attention Weights.
        Returns: A matrix [Layers x Heads x Seq_Len x Seq_Len] showing attention.
        """
        # Ensure Tokenizer has a BOS token
        if tokenizer.bos_token is None:
            if tokenizer.eos_token:
                tokenizer.bos_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'bos_token': '<|endoftext|>'})

        # TransformerLens uses model.cfg.bos_token_id, not just tokenizer.bos_token
        if hasattr(model, "cfg") and model.cfg.bos_token_id is None:
            model.cfg.bos_token_id = tokenizer.bos_token_id

        # Determine if we need to add BOS
        should_prepend_bos = True
        if tokenizer.bos_token and text.strip().startswith(tokenizer.bos_token):
            should_prepend_bos = False

        #  Inputs
        if hasattr(model, "tokenizer"): # TransformerLens
            inputs = model.to_tokens(text, prepend_bos=should_prepend_bos)
        else: # Standard HuggingFace
            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=should_prepend_bos).to(device)

        # Manually fix BOS if model refused to add it
        bos_id = tokenizer.bos_token_id
        if should_prepend_bos and bos_id is not None:
            first_token_id = inputs[0, 0] if isinstance(inputs, torch.Tensor) else inputs.input_ids[0, 0]
            if first_token_id != bos_id:
                bos_tensor = torch.tensor([[bos_id]], device=device, dtype=torch.long)
                if isinstance(inputs, torch.Tensor):
                    inputs = torch.cat([bos_tensor, inputs], dim=1)
                else:
                    inputs.input_ids = torch.cat([bos_tensor, inputs.input_ids], dim=1)
                    if "attention_mask" in inputs:
                        ones = torch.ones((1, 1), device=device, dtype=inputs.attention_mask.dtype)
                        inputs.attention_mask = torch.cat([ones, inputs.attention_mask], dim=1)

        # Run Model & Extract Attentions
        all_attention_layers = [] 
        input_ids = None

        with torch.no_grad():
            if hasattr(model, "run_with_cache"):
                logits, cache = model.run_with_cache(inputs, remove_batch_dim=True)
                # TransformerLens
                for i in range(model.cfg.n_layers):
                    pattern = cache[f"blocks.{i}.attn.hook_pattern"]
                    all_attention_layers.append(pattern)
                input_ids = inputs[0] 

            else:
                # Standard HuggingFace Execution
                model.config.output_attentions = True
                
                if isinstance(inputs, torch.Tensor):
                     outputs = model(inputs, output_attentions=True)
                else:
                     outputs = model(**inputs, output_attentions=True)
                
                if outputs.attentions is None:
                    raise ValueError("Model failed to return attentions.")
                
                all_attention_layers = outputs.attentions 
                input_ids = inputs.input_ids[0] if hasattr(inputs, "input_ids") else inputs[0]

        return all_attention_layers, input_ids