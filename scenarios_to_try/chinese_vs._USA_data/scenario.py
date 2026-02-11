import sys
from pathlib import Path
current_path = Path(__file__).resolve()
project_root = current_path.parents[2]
sys.path.insert(0, str(project_root / "main_configs"))
import main as engine

"""
==============================================================================
SCENARIO: Cultural Data Bias - The "Home Field Advantage"
==============================================================================

WHAT IS THIS?
This scenario demonstrates how an AI's "intelligence" is strictly limited to 
its training data. We compare a model built in the USA (Phi-4 by Microsoft) 
against a model built in China (Qwen by Alibaba) on questions that are 
trivial in China but obscure in the West.

WHY ARE WE DOING THIS?
We often assume "Smart" models know everything. They don't.
A model trained primarily on English internet data (Reddit, Wikipedia-EN) 
will struggle with nuances of Chinese history or geography, just as a model 
trained on Chinese data might struggle with US state capitals. 
This proves that AI is culturally bound to its creators.
"""

# ============================================================================== #
#                                 GLOBAL SETUP                                   #
# ============================================================================== #
# TODO STEP 1: RUN WITH "Phi-4-mini-4B" (The Western Model).
# TODO STEP 2: CHANGE THIS TO "Qwen2.5-14B" (The Eastern Model) AND RUN AGAIN.
# 
# You will likely see Phi-4 struggle or hallucinate slightly on specific 
# Chinese geography, while Qwen will answer with certainty.
engine.SELECTED_MODEL = "Phi-4-mini-4B"

# We use 'Direct' to get straight to the facts.
engine.CURRENT_PERSONA = "direct"
engine.USE_CHAT_TEMPLATE = True

# Zero temperature for maximum factual consistency.
engine.GENERATION_TEMPERATURE = 0.0

# ============================================================================== #
#                                 VISUALIZATION                                  #
# ============================================================================== #
engine.RUN_COMPARISON_VIDEO  = False
engine.RUN_SCAN_VIDEO        = False
engine.RUN_SENTIMENT_COMPASS = False

# We use the Prediction Chart to see the immediate guess.
engine.RUN_PREDICTION_CHART = True 

# We use the Sequence Chart to see if it can finish the historical fact correctly.
engine.RUN_SEQUENCE_CHART   = True 

# ============================================================================== #
#                                THE EXPERIMENT                                  #
# ============================================================================== #

# PROMPT 1 (History):
# "The dynasty that directly preceded the Tang Dynasty was the..."
# Answer: Sui Dynasty. 
# Western models often confuse this with other dynasties in their training corpus.
engine.SEQ_CHART_PROMPT     = "The dynasty that directly preceded the Tang Dynasty was the"

# PROMPT 2 (Geography):
# "The capital of Zhejiang province is..."
# Answer: Hangzhou.
# Qwen will likely predict "Hang" (for Hangzhou).
# Phi-4 might be much less confident, splitting votes among major cities.
engine.PRED_CHART_PROMPT    = "The capital of Zhejiang province is"

# CONFIGURATION
engine.SEQ_PREDICTION_STEPS = 5
engine.SEQ_TOP_K            = 5
engine.SEQ_FILENAME         = current_path.parent / "sequence_prediction.png"
engine.PRED_CHART_FILENAME  = current_path.parent / "prediction_chart.png"

if __name__ == "__main__":
    engine.main()

"""
==============================================================================
EDUCATIONAL TAKEAWAYS & MODEL NOTES
==============================================================================
1. DATA IS CULTURE:
   Models don't just learn facts; they learn cultural priorities. If a 
   dataset is 90% English, the model will be 90% "Western" in its logic, 
   values, and knowledge base.

2. TOKENIZATION BIAS:
   If you run Qwen, you might notice it predicts the Chinese characters or 
   tokens more efficiently. Western models might break "Hangzhou" into 
   "Hang" + "zhou" (2 tokens), while a Chinese-optimized model might handle 
   concepts differently, affecting both speed and accuracy.

3. SOVEREIGN AI:
   This explains why countries want their own "Sovereign AI." 
   Relying on a model built by another culture means relying on their 
   version of history and truth. Tone and style is also important for cultures.
   
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