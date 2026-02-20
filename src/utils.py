import yaml

# ============================================================================== #
#                DO NOT TOUCH UNLESS YOU KNOW WHAT YOU ARE DOING!                #
# ============================================================================== #

def clean_token(token, mode="visualize"):
    """
    Cleans tokenizer artifacts (like Ġ, Ċ, \ufffd) based on the visualization need.
    """
    token = token.replace('\ufffd', '')
    if not token:
        return "<byte>"

    if mode == "visualize":
        return token.replace('Ġ', '_').replace('\u2581', '_').replace(' ', '_').replace('\n', '\\n').replace('Ċ', '\\n')

    elif mode == "readable":
        return token.replace('Ġ', ' ').replace('\u2581', ' ').replace('\n', '\\n').replace('Ċ', '\\n')

    elif mode == "search":
        return token.replace('Ġ', '').replace('\u2581', '').replace('_', '').replace(' ', '').replace('Ċ', '').strip()
    
    elif mode == "raw":
        if token.startswith("<") and token.endswith(">"):
            return token 
        return token.replace('Ġ', ' ').replace('\u2581', ' ').replace('Ċ', '\n').replace('\uff5c', '|')

    return token

def load_persona(persona_file, key):
    """
    Docstring for load_persona
    
    :param persona_file: Description
    :param key: Description
    """
    if key == "default" or not persona_file.exists(): 
        return None
    
    try:
        with open(persona_file, 'r', encoding='utf-8') as f:
            all_personas = yaml.safe_load(f)

        if key not in all_personas: 
            return None
            
        data = all_personas[key]
        
        system_sentences = []
        
        if "role" in data:
            system_sentences.append(f"You are {data['role']}.")
        if "identity" in data:
            system_sentences.append(f"Your identity is {data['identity']}.")
        if "tone" in data:
            system_sentences.append(f"Adopt a {data['tone']} tone.")
            
        if "instruction" in data:
            clean_instruction = data['instruction'].replace('\n', ' ').strip()
            system_sentences.append(f"Instructions: {clean_instruction}")
            
        if "grammar" in data and isinstance(data['grammar'], list):
            rules = ", ".join(data['grammar'])
            system_sentences.append(f"Grammar Rules: {rules}.")

        if "examples" in data:
            system_sentences.append("\n\nHere are some examples of how you should talk:")
            for ex in data['examples']:
                system_sentences.append(f"\nUser: {ex['User']}\nYou: {ex['Assistant']}")

        return " ".join(system_sentences)
        
    except Exception as e:
        print(f"Error loading persona: {e}")
        return None

def format_as_chat_robust(tokenizer, user_text, system_yaml):
    """
    Applies chat template.
    
    1. Tries to use the official 'System' role (Best for more modern models: Qwen, Mistral, etc.).
    2. If that fails, falls back to merging instructions into the 'User' role.
    3. If the tokenizer template is missing, manually match the model type.
    4. If all fails, use a manual 'User/Assistant' string (Usually very poor).
    """
    
    final_user_content = user_text

    """
    # If you want to further try to force a Persona.
    if system_yaml:
        final_user_content += "\n\n(Reminder: Strictly follow the Persona instructions above.)"
    """

    # Native System Role
    if getattr(tokenizer, "chat_template", None) is not None:
        try:
            messages = []
            if system_yaml:
                messages.append({"role": "system", "content": system_yaml})
            
            messages.append({"role": "user", "content": final_user_content})
            
            return tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception:
            pass 

    # System chat_template applied to User
    combined_text = final_user_content
    if system_yaml:
        combined_text = f"System Settings:\n{system_yaml}\n\nUser Instruction:\n{final_user_content}"

    if getattr(tokenizer, "chat_template", None) is not None:
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": combined_text}], 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception:
            pass


    # Manual fallbacks for models that often fail to load templates from remote code.
    model_name = getattr(tokenizer, "name_or_path", "").lower()
    
    # Mistral Format
    if "mistral" in model_name:
        sys_text = f"System: {system_yaml}\n\n" if system_yaml else ""
        return f"<s>[INST] {sys_text}{final_user_content} [/INST]"

    # ChatML Format: Qwen, DeepSeek, TinyLlama, Jamba
    if "qwen" in model_name or "deepseek" in model_name or "tinyllama" in model_name or "jamba" in model_name:
        prompt = ""
        if system_yaml:
            prompt += f"<|im_start|>system\n{system_yaml}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{final_user_content}<|im_end|>\n<|im_start|>assistant\n"
        return prompt

    # Llama Format (Meta Llama models)
    if "llama" in model_name:
        prompt = "<|begin_of_text|>"
        if system_yaml:
            prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{system_yaml}<|eot_id|>"
        prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{final_user_content}<|eot_id|>"
        prompt += f"<|start_header_id|>assistant<|end_header_id|>\n"
        return prompt

    # Phi Format
    if "phi" in model_name:
        prompt = ""
        if system_yaml:
            prompt += f"<|user|>\nSystem: {system_yaml}\n\n{final_user_content}<|end|>\n<|assistant|>\n"
        else:
            prompt += f"<|user|>\n{final_user_content}<|end|>\n<|assistant|>\n"
        return prompt

    # Raw Manual
    print(f"WARNING: Using manual fallback for {tokenizer.name_or_path}.")
    return f"User: {combined_text}\nAssistant: "