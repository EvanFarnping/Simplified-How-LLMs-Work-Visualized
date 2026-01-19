from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformer_lens import HookedTransformer

import torch
import gc

# ============================================================================== #
#                DO NOT TOUCH UNLESS YOU KNOW WHAT YOU ARE DOING!                #
# ============================================================================== #

# NOTE, Feel free to read the comments to learn more about models and design of this project.
MODEL_MAP = {
    ##### --- RESEARCH/VERY OLD MODELS (CPU Usage, Oriented Towards Limited Low-Tier Pre-2020 Hardware) --- #####

    # https://huggingface.co/openai-community/gpt2 
    # ~0.5 to 1 GB (Not quantized), 124M parameters.
    # Small and fast transformer model developed by OpenAI in 2019. 
    # One of the 1st widely available open-source LLMs.
    # Suitable for basic text generation, research, and low-resource environments to understand core LLM tech.
    # Fully open under MIT license.
    "GPT-2": { 
        "repo": "gpt2",
        "arch": None,
        "trust_remote_code": False,
        "load_in_4bit": False
    }, 

    # https://huggingface.co/EleutherAI/pythia-160m 
    # ~0.6 to 1 GB (Not quantized), 160M parameters.
    # Part of the Pythia suite from EleutherAI.
    # Designed for interpretability and research on LLM training dynamics.
    # Fully open, ideal for studying core LLM behavior in controlled settings.
    "Pythia-160M": {
        "repo": "EleutherAI/pythia-160m",
        "arch": None,
        "trust_remote_code": False,
        "load_in_4bit": False
    },

    # https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
    # ~1 to 1.5 GB (Not quantized), 500M parameters.
    # A much smaller version of the main Qwen2.5 model.
    # Used for highly specific tasks and research related to increasing model speed (Speculative Decoding).
    "Qwen2.5-0.5B": {
        "repo": "Qwen/Qwen2.5-0.5B-Instruct",
        "arch": None,
        "trust_remote_code": True,
        "load_in_4bit": False
    },
    
    ##### --- LIGHT WEIGHT MODELS (More CPU Usage, GPU Resource Access Is Good To Have) --- #####

    # https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0 
    # ~2.2 to 3 GB (Not quantized), 1.1B parameters.
    # Compact version of the Llama architecture.
    # Developed by researchers for efficient deployment on mobile or edge devices.
    # Supports chat fine-tuning, making it useful for quick prototyping and small-scale applications.
    # Same architecture and tokenizer as Meta's Llama 2 models.
    # Made by the StatNLP Research Group at the Singapore University of Technology and Design (SUTD). 
    "TinyLlama-1.1B": {
        "repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "arch": None,
        "trust_remote_code": False,
        "load_in_4bit": False
    },
    
    # https://huggingface.co/Qwen/Qwen3-1.7B 
    # ~3 to 4 GB (Not quantized), 1.7B parameters
    # High-performance multilingual model from Alibaba.
    # Strong in core coding, reasoning, and general language tasks.
    # Intended for Agentic AI/precise tasks.
    "Qwen3-1.7B": { 
        "repo": "Qwen/Qwen3-1.7B",
        "arch": "Qwen2ForCausalLM",
        "trust_remote_code": True,
        "load_in_4bit": False
    },

    # https://huggingface.co/microsoft/Phi-4-mini-instruct 
    # ~8 GB (Not quantized), 3.8B parameters.
    # Microsoft's efficient small model series.
    # Outperforms similar-sized models in reasoning and coding benchmarks.
    # Designed for on-device inference and fine-tuning, with strong performance in math and logic tasks.
    # This model approaches Gemini 1.5 Flash and even GPT o1-mini & GPT-4o in precise tasks (assuming 14B Phi version).
    "Phi-4-mini-4B": {
        "repo": "microsoft/Phi-4-mini-instruct",
        "arch": None,
        "trust_remote_code": False,
        "load_in_4bit": False
    },

    ##### --- MEDIUM WEIGHT MODELS (GPU Usage, GPU Resources Are Practically Required) --- #####
    # (GPU Usage, 1-2 Medium-to-High-End GPU Card Resources Usually Required)

    # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
    # ~14 GB, 5-7 GB quantized (4-bit), 7B parameters
    # Versatile model from Mistral AI.
    # Comparable and in many cases, better than early GPT-3.5 in general capabilities.
    # Known for fast inference and strong performance in instruction-following, coding, and multilingual tasks.
    "Mistral-7B": { 
        "repo": "mistralai/Mistral-7B-Instruct-v0.3",
        "arch": None,
        "trust_remote_code": False,
        "load_in_4bit": True
    },

    # https://huggingface.co/Qwen/Qwen2.5-14B-Instruct 
    # ~ 30GB, ~9 GB quantized (4-bit), 14B parameters.
    # Weaker reasoning than "deepseek-ai/DeepSeek-V2-Lite-Chat", but overall still good general abilities.
    # In terms of performance, it's a lot closer to GPT-3.5 performance overall than Mistral-7B.
    "Qwen2.5-14B": {
        "repo": "Qwen/Qwen2.5-14B-Instruct",
        "arch": None,
        "trust_remote_code": True,
        "load_in_4bit": True
    },

    # https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat
    # ~18 GB, 7-8 GB quantized (4-bit), 16B parameters.
    # Open-source reasoning model from DeepSeek.
    # Cutting edge model from China exploring new ways to develop LLM based models.
    # Uses Reinforcement Learning, Mixture of Experts, Multi-Head Latent Attention, ETC.
    # Positioned between advanced models like Claude 3.5 Sonnet and o1-mini in coding and reasoning applications.
    # This can be run on a 2080Ti (source Evan Farnping's Personal Computer).
    "DeepSeek-Lite": {
        "repo": "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "arch": None,
        "trust_remote_code": True,
        "load_in_4bit": True
    },

    ##### --- HEAVIER WEIGHT AND MORE MODERN MODELS (Closer to GPT-4, Gemini 2.5 Pro, (2023-2024 models), ETC.) --- #####
    # (GPU Usage, 1-3 High-End GPU Card Resources Usually Required) ---
    
    # https://huggingface.co/Qwen/Qwen2.5-32B-Instruct 
    # Mainstream model from Alibaba (China).
    # When tuned well, can be comparable to GPT-4 to GPT-4o performance on various benchmarks.
    # ~60-65 GB, ~35-40 GB quantized (4-bit), 33B parameters
    "Qwen2.5-32B": {
        "repo": "Qwen/Qwen2.5-32B-Instruct",
        "arch": None,
        "trust_remote_code": True,
        "load_in_4bit": True
    },

    # https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
    # Model from DeepSeek (China),
    # ~60+ GB, 30-40+ GB quantized (4-bit), 33B parameters (effective via MoE)
    # Consistently competitive against OpenAI-o1-mini (GPT o1-mini).
    # Focused on reasoning and coding, not necessarily for chatting.
    "DeepSeek-R1": {
        "repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "arch": None,
        "trust_remote_code": True,
        "load_in_4bit": True
    },

    # https://huggingface.co/ai21labs/AI21-Jamba2-Mini
    # ~60+ GB, 25-30+ GB quantized (4-bit), 12B active parameters (52B total)
    # Hybrid MoE-Mamba model from AI21 Labs. Very experimental design/different design than what is mainstream.
    # Allegedly, performance is comparable and even better vs. original GPT-4, even GPT-4o in some cases.
    # Was released on Jan 8th 2026, very new, claims are skeptical.
    "Jamba-2-Mini": {
        "repo": "ai21labs/AI21-Jamba2-Mini",
        "arch": None,
        "trust_remote_code": True,
        "load_in_4bit": True
    },

    # https://huggingface.co/Qwen/Qwen2.5-72B-Instruct
    # 120GB+, 60-80GB quantized (4-bit), 73B parameters.
    # Considered to be stronger than GPT-3.5 & GPT-4, even GPT-4 Turbo in some cases.
    "Qwen2.5-72B": {
        "repo": "Qwen/Qwen2.5-72B-Instruct",
        "arch": None,
        "trust_remote_code": True,
        "load_in_4bit": True
    },

    # NOTE SPECIAL
    # https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct 
    # Example of a "small" modern model at the cutting edge. 109B Parameters (17B+ Active). 200+ GB of VRAM, 
    # A very modern and cutting edge model made by Meta/FaceBook. (Likely need to share contact info).
    # Essentially the same performance, if not, very similar performance to GPT-4 and GPT-4o.
    "Llama-4-Scout": {
        "repo": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "arch": None,
        "trust_remote_code": True,
        "load_in_4bit": True
    }

    # NOTE. 
    # Once you start going beyond 50B+ parameter models, 
    # you're dealing with very advanced models aiming to compete aginst mainstream models.
    # For example, most big and the most advance models have special designs and trillions of parameters. 
}

class ModelManager:
    def __init__(self):
        self.current_model_name = None
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, selection_name):
        """
        Loads a model dynamically based on hardware availability.
        """
        if selection_name == self.current_model_name and self.model is not None:
            return self.model, self.tokenizer

        if self.model is not None:
            self.unload_model()

        config = MODEL_MAP[selection_name]
        print(f"🔄🔄🔄 Loading {selection_name} on {self.device.upper()}... 🔄🔄🔄")

        should_quantize = config.get("load_in_4bit", False)

        quantization_config = None
        if self.device == "cuda" and should_quantize:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_enable_fp32_cpu_offload=True
            )

        try:
            self.model = HookedTransformer.from_pretrained(
                config["arch"] if config["arch"] else config["repo"],
                hf_model=config["repo"] if config["arch"] else None,
                device=self.device,
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False,
                hf_model_kwargs={
                    "trust_remote_code": config["trust_remote_code"], 
                    "quantization_config": quantization_config,
                    "attn_implementation": "eager"
                }
            )
            self.tokenizer = self.model.tokenizer

        except Exception as e:
            print(f"TransformerLens failed. Using standard Transformers loading.")

            self.tokenizer = AutoTokenizer.from_pretrained(config["repo"], 
                                                           trust_remote_code=config["trust_remote_code"])

            active_device_map = "auto"
            if "DeepSeek" in selection_name or "Qwen" in selection_name: # Due to cuda and cpu swap being poor.
                active_device_map = "cuda"

            if self.device == "cuda":
                target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                target_dtype = torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                config["repo"],
                trust_remote_code=config["trust_remote_code"],
                quantization_config=quantization_config,
                device_map=active_device_map,
                attn_implementation="eager",
                dtype=target_dtype
            )
        
        # Deepseek has different caching logic that I don't want to deal with since it's very different than the other models
        if "DeepSeek" in selection_name:
             self.model.config.use_cache = False

        self.current_model_name = selection_name
        print(f"✅✅✅  {selection_name} Ready. ✅✅✅")
        return self.model, self.tokenizer

    def unload_model(self):
        """
        Forcefully clears the model from memory.
        
        [NOTE QUALITY OF LIFE FEATURE]:
        # Prevents Crashes:
        Most educational laptops, consumer GPUs (like an RTX 2080), educational environments (Student Google Colab Pro)
           only have enough memory to hold ONE large model at a time. If we tried to load 
           a second model (e.g., switching from Phi-3 to Mistral) without deleting the first, 
           the computer would run out of RAM and crash immediately.
        
        # Automation:
        Without this function, you would have to manually restart your 
           Python kernel or terminal every time you wanted to run a different visualization. 
           This function automates that "restart" process, allowing you to run all experiments 
           in a single smooth flow, especially for non-CS high school students who are not familiar
           with changing values across files and terminal editing on the spot.

        [NOTE TECHNICAL REALITY CHECK]:
        In a professional production environment (like ChatGPT or a server), you would 
        usually avoid doing this.
        
        Latency Cost: Loading a model takes time (10-30 seconds). Unloading and 
            reloading it constantly makes the application very slow.
        Standard Practice: Pros use a "Singleton Pattern" where the model stays 
            loaded in RAM forever, serving thousands of requests instantly.
        
        We sacrifice speed here to gain stability and ease of use for this specific project.
        """
        
        print("🧹 Cleaning up model memory...")
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        
        self.model = None
        self.tokenizer = None
        self.current_model_name = None
        
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        print("✨✨✨ Memory Cleared. :D ✨✨✨")