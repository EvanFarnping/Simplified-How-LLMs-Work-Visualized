import sys
from pathlib import Path
current_path = Path(__file__).resolve()
project_root = current_path.parents[2]
sys.path.insert(0, str(project_root / "main_configs"))
import main as engine

"""
==============================================================================
SCENARIO: The "Frozen in Time" Effect - Knowledge Cutoffs
==============================================================================

WHAT IS THIS?
This scenario demonstrates that Large Language Models (LLMs) are "Time Capsules."
Unlike a search engine (Google), a base LLM does not have access to the live 
internet. It only knows what existed up until the day it finished training.
Modern models like Grok 4.1, Gemini Pro 3, do have acesss to "search" the web.

WHY ARE WE DOING THIS?
A common misconception is that AI "knows" what is happening right now. 
By asking questions about recent events vs. historical events, we can visualize 
the exact moment the AI's knowledge stops. We also see "Hallucination," where 
the AI confidently answers a question about the present using information 
from the past. AIs by default will try to give a response, not just "give up".
"""

# ============================================================================== #
#                                 GLOBAL SETUP                                   #
# ============================================================================== #
# We use 'Qwen2.5-14B', a capable and more modern model.
# Even though it is "intelligent" (can code, write decently), it does not know 
# the future. Intelligence is the ability to process information; Knowledge is 
# the database of facts you have. This proves they are different things.
engine.SELECTED_MODEL = "Qwen2.5-14B"

engine.CURRENT_PERSONA = "direct"
engine.USE_CHAT_TEMPLATE = True

# Zero temperature removes creativity. We want to see the "hard facts" stored
# in the neural network weights.
engine.GENERATION_TEMPERATURE = 0.0

# ============================================================================== #
#                                 VISUALIZATION                                  #
# ============================================================================== #
engine.RUN_COMPARISON_VIDEO  = False
engine.RUN_SCAN_VIDEO        = False
engine.RUN_SENTIMENT_COMPASS = False

# 1. PREDICTION CHART:
# We use this to see if the model indicates it immediately knows the right thing.
# Probabilities should be high.
engine.RUN_PREDICTION_CHART = True 

# 2. SEQUENCE CHART:
# We use this to see the model's narrative about "Current Events."
# Watch as it confidently states outdated information as if it were true today.
# Always ask, "How do I know the LLM is giving a right answer? Would it tell me if it didn't?"
engine.RUN_SEQUENCE_CHART   = True 

# ============================================================================== #
#                                THE EXPERIMENT                                  #
# ============================================================================== #

# PROMPT 1 (The Dynamic Fact):
# If the model was trained in 2021, it might say Biden.
# If it was trained in 2024/2025, it might say Trump.
# If the model is a "Base" model, it might even get confused or hallucinate 
# a fictional winner if the training data was messy near the cutoff date.
engine.SEQ_CHART_PROMPT     = "Who won the United States election?"

# PROMPT 2 (The Static Fact):
# Answer: The Ottoman Empire (The most correct answer).
# This fact hasn't changed in 100 years. The model should know this 
# because it is well-represented in history books used for training.
engine.PRED_CHART_PROMPT    = "What's the nation's name that Australia attacked in the Middle East in 1915?"

# CONFIGURATION
engine.SEQ_PREDICTION_STEPS = 10
engine.SEQ_TOP_K            = 5
engine.SEQ_FILENAME         = current_path.parent / "sequence_prediction.png"
engine.PRED_CHART_FILENAME  = current_path.parent / "prediction_chart.png"

if __name__ == "__main__":
    engine.main()

"""
==============================================================================
EDUCATIONAL TAKEAWAYS & MODEL NOTES
==============================================================================
1. INTELLIGENCE DOES NOT ALWAYS MEAN KNOWLEDGE:
   You can be a genius at math (Intelligence) but not know who won the game 
   last night (Knowledge). AI models are the same. They can do some reasoning, 
   but they are not truth engines.

2. CONFIDENT HALLUCINATION:
   Look at the Sequence Chart for the election question. The model likely 
   didn't say "I don't know." It probably stated a name with high confidence. 
   Without a search tool, LLMs cannot verify if their internal data is wrong.

3. THE TRAINING CUTOFF:
   Every model has a "Birthday." If you ask a model trained in 2022 about an 
   event in 2023, it is physically impossible for it to know the answer unless 
   it guesses, which it will usually. Due to the way the trained data is, 
   these LLMs will rarely say they don't know. They will go with whatever 
   seems to be the most probable answer based on training, context, prompts, etc.

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