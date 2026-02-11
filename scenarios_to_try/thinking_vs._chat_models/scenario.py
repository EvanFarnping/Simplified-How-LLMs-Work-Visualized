import sys
from pathlib import Path
current_path = Path(__file__).resolve()
project_root = current_path.parents[2]
sys.path.insert(0, str(project_root / "main_configs"))
import main as engine

"""
==============================================================================
SCENARIO: Thinking Models/Features
==============================================================================

WHAT IS THIS?
This scenario shows a model responding with using a "Thinking" process (Chain of Thought) 
before answering. We use a math problem that requires multiple steps to solve.

WHY ARE WE DOING THIS?
Standard Chat Models try to predict the final answer immediately. For complex 
logic, this often leads to hallucinations (wrong answers). 
"Thinking" models (or models prompted to reason) generate intermediate tokens 
to "work out" the problem in their scratchpad (context window) before 
committing to a final result. This mimics human "System 2" thinking: slow, 
deliberate, and logical.
NOTE: Lately, many modern models are hybrid, knowing when to "think" vs. saying the answer.
A good example is when GPT-5 came out. While pretty lame at the time, it saved resources by being more hybrid.
"""

# ============================================================================== #
#                                 GLOBAL SETUP                                   #
# ============================================================================== #
# We use 'Qwen3-1.7B' (or you can try 'DeepSeek-R1' if you have the hardware).
# These models are designed or prompted to output "reasoning traces" 
# (steps of logic) before the final answer.
# Notice: "Thinking" takes more time. It generates more tokens.
# More tokens = More Energy = More Heat = More Cost.
engine.SELECTED_MODEL = "Qwen3-1.7B" 

# We use 'direct' to minimize polite chatter, but the model's internal 
# training might still force it to output a "Step-by-step" explanation.
engine.CURRENT_PERSONA = "direct"
engine.USE_CHAT_TEMPLATE = True

# Zero temperature ensures the model follows the most logical path without 
# getting "creative" with the numbers.
engine.GENERATION_TEMPERATURE = 0.0

# ============================================================================== #
#                                 VISUALIZATION                                  #
# ============================================================================== #
engine.RUN_PREDICTION_CHART  = False
engine.RUN_COMPARISON_VIDEO  = False
engine.RUN_SCAN_VIDEO        = False
engine.RUN_SENTIMENT_COMPASS = False

# We use the Sequence Chart because we need to see the *Length* of the thought.
# A chat model might answer in 3 tokens: "The answer is 249."
# A thinking model might answer in 30-50 tokens: "First, 21 times 11 is..."
engine.RUN_SEQUENCE_CHART    = True

# ============================================================================== #
#                                THE EXPERIMENT                                  #
# ============================================================================== #

# THE PROMPT:
# A model that guesses immediately often guesses some 3 digit number
# because they "look" right. A model that writes out the step does better.
engine.SEQ_CHART_PROMPT     = "Solve for y if x equals 11. y = 21x + 18."
# TODO. It is likely Qwen3-1.7B will not be enough.
# You may need to try Mistral-7B, Qwen2.5-14B, DeepSeek-Lite, and above.
# OR, try an easier question.

# CONFIGURATION
engine.SEQ_PREDICTION_STEPS = 30 # TODO May need to increase to see the steps.
engine.SEQ_TOP_K            = 3
engine.SEQ_FILENAME         = current_path.parent / "sequence_prediction.png"

if __name__ == "__main__":
    engine.main()

"""
==============================================================================
EDUCATIONAL TAKEAWAYS & MODEL NOTES
==============================================================================
1. THE SCRATCHPAD EFFECT:
   Most LLMs lack working memory. They cannot "hold" a number in their head while 
   doing the next step. They must WRITE it down to see it. 
   "Thinking" is just the act of writing tokens to the context window so the 
   model can read its own work for the next prediction. 
   NOTE: Simplified quite a bit.

2. INFERENCE COMPUTE:
   This visualization shows why "Reasoning Models" (like OpenAI o1 or Grok Expert) 
   are slower and more expensive. They are doing more work (generating more 
   tokens) per question. Accuracy has a literal energy cost.

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