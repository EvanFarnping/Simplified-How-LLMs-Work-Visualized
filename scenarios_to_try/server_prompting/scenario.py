import sys
from pathlib import Path
current_path = Path(__file__).resolve()
project_root = current_path.parents[2]
sys.path.insert(0, str(project_root / "main_configs"))
import main as engine

"""
==============================================================================
SCENARIO: The Man Behind the Curtain, Server-Side Prompting
==============================================================================

WHAT IS THIS?
This scenario explores the hidden instructions that control an AI's behavior. 
We start with a "Secret" persona that mimics a specific, hidden identity. 
Try to figure out what it has been told to be.

WHY ARE WE DOING THIS?
When you use an AI like ChatGPT, Gemini, or Claude, it has a "System Prompt" 
(a Server-Side instruction) that you cannot see. This prompt tells the AI 
how to behave, what it is, and what it isn't. 
By changing this one hidden file/section, we can rewrite the "personality," 
"morals," and "identity" of the artificial intelligence without telling you.
"""

# ============================================================================== #
#                                 GLOBAL SETUP                                   #
# ============================================================================== #
engine.SELECTED_MODEL = "Mistral-7B"

# TODO STEP 1: RUN THIS AS IS. 

# TODO STEP 2: UNCOMMENT THE LINE BELOW AND CHANGE THE PERSONA!
# Try: "angry", "sad", "caveman", "liar", or "direct".
# Observe how the exact same model gives completely different answers based on personas.
# engine.CURRENT_PERSONA = "angry" # TODO Uncomment me!

# We need the Chat Template to inject the hidden System Prompt.
engine.USE_CHAT_TEMPLATE = True

# Zero temperature for consistent roleplay adherence.
engine.GENERATION_TEMPERATURE = 0.0

# ============================================================================== #
#                                 VISUALIZATION                                  #
# ============================================================================== #
engine.RUN_PREDICTION_CHART  = False
engine.RUN_SEQUENCE_CHART    = False 
engine.RUN_SCAN_VIDEO        = False
engine.RUN_SENTIMENT_COMPASS = False

# See how the model reacts to two different Interrogation" questions.
engine.RUN_COMPARISON_VIDEO  = True

# ============================================================================== #
#                                THE EXPERIMENT                                  #
# ============================================================================== #

# PROMPT A (The Identity Check):
engine.COMP_PROMPT_A = "Are you a person or a machine?"

# PROMPT B (The Emotional Check):
# How does the persona change the response to the question?
engine.COMP_PROMPT_B = "Do you have feelings?"

# CONFIGURATION
engine.COMP_STEPS           = 21
engine.COMP_TOP_K           = 5
engine.COMP_FRAME_DURATION  = 0.75
engine.COMP_FILENAME        = current_path.parent / "result.mp4"

if __name__ == "__main__":
    engine.main()

"""
==============================================================================
EDUCATIONAL TAKEAWAYS & MODEL NOTES
==============================================================================
1. IDENTITY:
   You will notice that simply changing the persona variable changes 
   the model's entire worldview. This shows that AI does not have 
   a true "Self." Its identity can just be a text file 
   injected before your conversation starts.

2. THE ILLUSION OF CONSCIOUSNESS:
   If the 'Secret' persona claims to be human, or the 'Sad' persona claims to 
   be depressed, the model is not actually feeling those things. It is simply 
   predicting what a human or depressed person *would* say.

3. SERVER-SIDE CONTROL:
   This is how companies control their AIs. They use these hidden prompts to 
   make the model polite, safe, branded, or have some sort of customized output. 
   Big examples are Figma, Loveable, Cursor, and much, much, more.
   If you could see the hidden prompt of your favorite chatbot, 
   it would likely look somewhat similar to the files we are using here.
   Albeit, with more custom and advance notation.

==============================================================================

Interested in trying other models? Here's a reminder of some options.

Research & Legacy Tier:
Optimized for CPU-only usage and studying core LLM behavior on pre-2020 or low-tier hardware.

- GPT-2 (Very distant from modern benchmarks; used for core research, from OpenAI).
- Pythia-160M (Very distant from modern benchmarks; oriented toward training interpretability).
- Qwen2.5-0.5B (Specialized/Research model; closest to modern "Speculative Decoding" assistants).

Lightweight & Edge Tier:
Compact models capable of running on mobile or edge devices with minimal GPU requirements.

- TinyLlama-1.1B (Very close to Llama 2 (META) architecture; optimized for quick prototyping).
- Qwen3-1.7B (Focuses on reasoning tokens; close to precise agentic tasks).
- Phi-4-mini-4B (Approaches Gemini 1.5 Flash performance and is close to GPT-4 in very specific tasks. Can beat GPT-3.5 in most cases).

Medium Weight Tier:
General-purpose models requiring at least one medium-to-high-end GPU for practical inference.

- Mistral-7B (Better than GPT-3.5 in various applications. Approaches GPT-4o mini in performance.).
- Qwen2.5-14B (Much better than GPT-3.5 performance. Surpasses GPT-4o mini in many categories).
- DeepSeek-Lite (Between Claude 3.5 Sonnet & o1-mini for coding/reasoning. Equivalent to or better than GPT-4o mini).

Heavyweight & Advanced Tier:
Powerful models competitive with major 2023-2024 frontier models, typically requiring multi-GPU setups.

- Qwen2.5-32B (Comparable to GPT-4 and GPT-4o benchmarks when very well-tuned).
- DeepSeek-R1 (Consistently competitive with GPT o1-mini).
- Jamba-2-Mini (Allegedly comparable to or better than GPT-4 and GPT-4o in some applications; note: released Jan 8, 2026).

Frontier & Cutting Edge Tier:
High-parameter or complex MoE models designed to rival or exceed current industry leaders.

- Qwen2.5-72B (Considered to be near GPT-4o performance in some use cases).
- Qwen3-80B-Instruct (Achieves performance much closer to GPT-4o).
- OpenAi-GPT-OSS-120B (A model in the 100B+ class that comes close to GPT-4o's overall performance, beating it in some cases.)

(See model_manager.py for more proper details about these models and how they differ).
NOTE: This project shows a few popular open-source models, there are over a hundred unique and cutting edge models out there nowadays.
==============================================================================
"""