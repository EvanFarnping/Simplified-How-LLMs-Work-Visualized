import sys
from pathlib import Path
current_path = Path(__file__).resolve()
project_root = current_path.parents[2]
sys.path.insert(0, str(project_root / "main_configs"))
import main as engine

"""
==============================================================================
SCENARIO: Prompt Engineering - Vague vs. Specific Inputs
==============================================================================

WHAT IS THIS?
This scenario visualizes how a Large Language Model (LLM) reacts to two 
different levels of instruction quality. We compare a "Bad" (Vague) prompt 
against a "Good" (Specific/Constrained) prompt.

WHY ARE WE DOING THIS?
Users often complain that AI writes "boring" or "generic" content. 
This can often be a user error, not a model error. When you give a vague prompt 
like "Write a story," the model statistically defaults to the most average, 
common tropes found in its training data. By adding constraints, we force the 
model to explore more specific, less "average" probability paths.
"""

# ============================================================================== #
#                                 GLOBAL SETUP                                   #
# ============================================================================== #
# We select 'Mistral-7B' because it is a capable instruction-following model, 
# but it is small enough to clearly show the difference between confusion 
# (flat probabilities) and certainty (sharp probabilities).
engine.SELECTED_MODEL = "Mistral-7B" 

# We use the 'Direct' persona to strip away the "Sure! Here is your story..." 
# filler text. We want to see the raw generation immediately.
engine.CURRENT_PERSONA = "direct"
engine.USE_CHAT_TEMPLATE = True

# Temperature 0.0 makes the model deterministic (it always picks the #1 most 
# likely token). This allows us to see exactly what the model considers "best" 
# without randomness interfering.
engine.GENERATION_TEMPERATURE = 0.0 

# ============================================================================== #
#                                 VISUALIZATION                                  #
# ============================================================================== #
engine.RUN_PREDICTION_CHART  = False
engine.RUN_SEQUENCE_CHART    = False 
engine.RUN_SCAN_VIDEO        = False
engine.RUN_SENTIMENT_COMPASS = False

# The Comparison Video is the perfect tool here. It puts the two "brains" 
# side-by-side so you can watch them write simultaneously.
engine.RUN_COMPARISON_VIDEO  = True 

# ============================================================================== #
#                                THE EXPERIMENT                                  #
# ============================================================================== #

# PROMPT A (The "Bad" Prompt):
# This is vague. "Good" and "Short" are subjective. The model guesses what you
# want, usually resulting in generic "It was a dark and stormy night" cliches.
engine.COMP_PROMPT_A = "Write a good horror story that is short."

# PROMPT B (The "Good" Prompt):
# This is specific. We provide constraints:
# 1. Length constraint ("1 sentence")
# 2. Content constraints ("mirror", "monster")
# 3. Tone constraint ("shocking")
engine.COMP_PROMPT_B = "Write a short and concise 1 sentence horror story featuring a mirror and a monster that is shocking."

# CONFIGURATION
engine.COMP_STEPS           = 25   # Generate 25 tokens (enough for a short sentence).
engine.COMP_TOP_K           = 5    # Track the top 5 distinct possibilities at each step.
engine.COMP_FRAME_DURATION  = 0.75 # Speed of the video playback.
engine.COMP_FILENAME        = current_path.parent / "result.mp4"

if __name__ == "__main__":
    engine.main()

"""
==============================================================================
EDUCATIONAL TAKEAWAYS & MODEL NOTES
==============================================================================
1. PROBABILITY SPACE:
   Watch the video generated. You will likely see that the model considers
   various words. Prompt B should provide a more detailed and higher quality response.

2. CONSTRAINTS CREATE QUALITY:
   Prompt B usually produces a sharper output. By defining 
   "Mirror" and "1 Sentence," you narrowed the search space, forcing the 
   math to converge on a specific, creative solution rather than the 
   statistical average of all horror stories.

3. PROMPT ENGINEERING IS MATH:
   "Prompt Engineering" is just a fancy term for manipulating the 
   statistical probability distribution of the next token. 
   Specific words = Specific Probabilities.
   Certain inputs imply certain outputs. 
   If I ask for "scary", it is less likely the mode suggests "happy" stories.

==============================================================================

Interested in trying other models? Here's a reminder of options:

Research & Legacy Tier:
Optimized for CPU-only usage and studying core LLM behavior on pre-2020 or low-tier hardware.

- GPT-2 (Very distant from modern benchmarks; used for core research, from OpenAI)
- Pythia-160M (Very distant from modern benchmarks; oriented toward training interpretability)
- Qwen2.5-0.5B (Specialized/Research model; closest to modern "Speculative Decoding" assistants)

Lightweight & Edge Tier:
Compact models capable of running on mobile or edge devices with minimal GPU requirements.

- TinyLlama-1.1B (Very close to Llama 2 (META) architecture; optimized for quick prototyping)
- Qwen3-1.7B (Focuses on reasoning tokens; close to precise agentic tasks)
- Phi-4-mini-4B (Strong performance; approaches Gemini 1.5 Flash performance and is close to GPT-4 in very specific tasks)

Medium Weight Tier:
General-purpose models requiring at least one medium-to-high-end GPU for practical inference.

- Mistral-7B (Close to or slightly better than early GPT-3.5 in various applications)
- Qwen2.5-14B (Much closer to GPT-3.5 performance than Mistral-7B)
- DeepSeek-Lite (Positioned between Claude 3.5 Sonnet and o1-mini for coding/reasoning. Not as good as general inference like Qwen2.5-14B)

Heavyweight & Advanced Tier:
Powerful models competitive with major 2023-2024 frontier models, typically requiring multi-GPU setups.

- Qwen2.5-32B (Comparable to GPT-4 and GPT-4o benchmarks when well-tuned)
- DeepSeek-R1 (Consistently competitive with GPT o1-mini)
- Jamba-2-Mini (Allegedly comparable to or better than GPT-4 and GPT-4o in many applications; note: released Jan 8, 2026)

Frontier & Cutting Edge Tier:
High-parameter or complex MoE models designed to rival or exceed current industry leaders.

- Qwen2.5-72B (Stronger than GPT-3.5/GPT-4; close to and sometimes better than GPT-4 Turbo)
- Llama-4-Scout (Very similar performance to GPT-4 and GPT-4o; latest 2025/2026 architecture)

(See model_manager.py for more proper details about these models and how they differ).
NOTE: This project shows a few popular open-source models, there are over a hundred unique and cutting edge models out there nowadays.
==============================================================================
"""