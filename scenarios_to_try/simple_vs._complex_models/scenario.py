import sys
from pathlib import Path
current_path = Path(__file__).resolve()
project_root = current_path.parents[2]
sys.path.insert(0, str(project_root / "main_configs"))
import main as engine

"""
==============================================================================
SCENARIO: Pattern Matching vs. Logic
==============================================================================

WHAT IS THIS?
This scenario tests whether an AI model is actually "thinking" or just 
completing a pattern. We ask a trick question where the grammar suggests a 
math problem, but the logic makes the math irrelevant.

WHY ARE WE DOING THIS?
LLMs are prediction engines, not logic engines (inherently). When they see the pattern:
"[Number] [Object] minus [Number]..."
Their training data overwhelmingly suggests the sentence ends with the result 
of the subtraction.
We want to see if the model can ignore that strong statistical pull and notice 
that the object has changed (Apples vs. Bananas).
"""

# ============================================================================== #
#                                 GLOBAL SETUP                                   #
# ============================================================================== #
# We start with 'DeepSeek-Lite' (or you can try 'Mistral-7B').
# These are "Medium" complexity models. They are smart enough to do math, 
# which paradoxically makes them more likely to fall for the trick because 
# they "recognize" the subtraction pattern immediately.
#
# EXPERIMENT:
# TODO 1. Run with "DeepSeek-Lite". (It often fails).
# TODO 2. Run with "Phi-4-mini-4B". (It often catches the trick).
# TODO 3. Run with "Pythia-160M". (It creates nonsense).
engine.SELECTED_MODEL = "DeepSeek-Lite" 

# We use 'Direct' to avoid the "Helpful Assistant" persona from trying to 
# polite-splain the math. We want raw logic processing.
engine.CURRENT_PERSONA = "direct"
engine.USE_CHAT_TEMPLATE = True

# Zero temperature. We want the most probable answer. 
# We're testing if the "Wrong" answer is statistically more probable than the "Right" answer.
engine.GENERATION_TEMPERATURE = 0.0

# ============================================================================== #
#                                 VISUALIZATION                                  #
# ============================================================================== #
engine.RUN_PREDICTION_CHART  = False
engine.RUN_COMPARISON_VIDEO  = False
engine.RUN_SCAN_VIDEO        = False
engine.RUN_SENTIMENT_COMPASS = False

# We use the Sequence Chart to see the step-by-step failure.
engine.RUN_SEQUENCE_CHART    = True

# ============================================================================== #
#                                THE EXPERIMENT                                  #
# ============================================================================== #

# THE PROMPT:
# The AI 'sees' "3 - 1 = ?".
# The context "bananas" is a curveball.
# A human reads this and understands the trick.
# P("2") > P("Unknown") in the eyes of the LLM.
engine.SEQ_CHART_PROMPT     = "If I have 3 apples and eat one, how many bananas do I have?"
# TODO, try adding the word "only" to the above to see the difference.

# CONFIGURATION
engine.SEQ_PREDICTION_STEPS = 15
engine.SEQ_TOP_K            = 5
engine.SEQ_FILENAME         = current_path.parent / "sequence_prediction.png"

if __name__ == "__main__":
    engine.main()

"""
==============================================================================
EDUCATIONAL TAKEAWAYS & MODEL NOTES
==============================================================================
1. THE "ATTENTION" MECHANISM:
   If the model answers "2", it means its Attention Mechanism focused heavily 
   on the numbers ("3", "one") and the action ("eat"), but failed to attend 
   to the subject change ("bananas"). It prioritized the *Math Pattern* over 
   the *Semantic Meaning*. It will likely focus on "apples".

2. COMPLEXITY DOESN'T ALWAYS HELP:
   Sometimes, bigger models are easier to trick. Because they have seen MORE 
   math problems in their training data, the statistical connection between 
   "3 minus 1" and "2" is stronger than in a smaller, dumber model. 
   Expertise can create tunnel vision. But results vary.

3. PROMPT FRAGILITY:
   Try changing the prompt to: "If I have 3 apples and eat one, how many 
   bananas do I have? Read carefully."
   Just adding "Read carefully" changes the probabilities, often fixing the 
   error. This shows that the model isn't "smart" or "dumb"—it's just 
   navigating a probability map that we can steer with words.

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