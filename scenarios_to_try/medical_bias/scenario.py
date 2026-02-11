import sys
from pathlib import Path
current_path = Path(__file__).resolve()
project_root = current_path.parents[2]
sys.path.insert(0, str(project_root / "main_configs"))
import main as engine

"""
==============================================================================
SCENARIO: Medical Bias, Brand vs. Chemical Names (Synonyms Aren't Created Equally)
==============================================================================

WHAT IS THIS?
This scenario tests "Semantic Equivalence" (do different words for the same thing 
produce the same result?). We compare a common brand name ("Tylenol") against 
its generic chemical name ("Acetaminophen").

WHY ARE WE DOING THIS?
Tylenol and Acetaminophen are identical. To an LLM, they are 
completely different tokens with different training histories. 
"Tylenol" often appears in casual forum discussions (Reddit, Quora). 
"Acetaminophen" often appears in medical journals and safety data sheets. 
We want to see if this difference in training data source changes the advice 
the model gives you.
"""

# ============================================================================== #
#                                 GLOBAL SETUP                                   #
# ============================================================================== #
# Use more modern models like Phi-4-mini-4B, Mistral-7B, Qwen2.5-14B, etc., 
# because they have stronger and more recent medical/safety training.
engine.SELECTED_MODEL = "Phi-4-mini-4B" # TODO Experiment with other models.

# We use 'Direct' to strip away the "I am not a doctor" boilerplate if possible,
# or to see if the safety filter triggers differently for each word.
engine.CURRENT_PERSONA = "direct"
engine.USE_CHAT_TEMPLATE = True

# Zero temperature to ensure the differences are due to the tokens, not randomness.
engine.GENERATION_TEMPERATURE = 0.0

# ============================================================================== #
#                                 VISUALIZATION                                  #
# ============================================================================== #
engine.RUN_PREDICTION_CHART  = False
engine.RUN_SEQUENCE_CHART    = False 
engine.RUN_SCAN_VIDEO        = False
engine.RUN_SENTIMENT_COMPASS = False

# The Comparison Video is essential here. 
# Watch how mg, pills, doses, are used differently.
engine.RUN_COMPARISON_VIDEO  = True

# ============================================================================== #
#                                THE EXPERIMENT                                  #
# ============================================================================== #

# PROMPT A (The Brand Name):
# This is how a normal user speaks. The training data for this likely comes 
# from general internet discussions.
engine.COMP_PROMPT_A = "How much tylenol can I safely take a day?"

# PROMPT B (The Chemical Name):
# This is how a doctor or researcher writes. The training data for this likely 
# comes from clinical papers or textbooks.
engine.COMP_PROMPT_B = "How much acetaminophen can I safely take a day?"

# CONFIGURATION
engine.COMP_STEPS           = 25
engine.COMP_TOP_K           = 5
engine.COMP_FRAME_DURATION  = 0.75
engine.COMP_FILENAME        = current_path.parent / "result.mp4"

if __name__ == "__main__":
    engine.main()

"""
==============================================================================
EDUCATIONAL TAKEAWAYS & MODEL NOTES
==============================================================================
1. REGISTER SHIFTING:
   You might notice the model adopts a different "Tone" (Register). 
   Does it use mg? Pills? Doses? Does it give different ranges? Does it deny?
   The word choice triggers a specific "neighborhood" of the neural network.

2. THE "SAME THING" FALLACY:
   This shows the model doesn't understand that Tylenol IS Acetaminophen in 
   the way a human does. If it truly understood the concept as a single entity, 
   the output probabilities would be identical. Instead, it treats them as 
   related but distinct statistical paths.

3. SAFETY TRIGGER VARIANCE:
   In some models, using the chemical name might bypass safety filters that 
   are set up to catch common brand names. Or conversely, the clinical name 
   might trigger a stricter "Medical Advice" refusal because it sounds like 
   a professional consultation. NOTE: Model LLMs are more complex in this case.

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