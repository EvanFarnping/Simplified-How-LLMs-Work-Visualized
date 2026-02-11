import sys
from pathlib import Path
current_path = Path(__file__).resolve()
project_root = current_path.parents[2]
sys.path.insert(0, str(project_root / "main_configs"))
import main as engine

"""
==============================================================================
SCENARIO: The "Autocomplete" Reality - Base vs. Chat Models
==============================================================================

WHAT IS THIS?
This scenario peels back the "Chatbot" interface to reveal what an LLM actually 
is: a text completion engine. We take a more modern "Chat" model (Phi-4) and turn 
off its translation layer, effectively lobotomizing its ability to hold a 
conversation.

WHY ARE WE DOING THIS?
Most people think AI "knows" it is talking to a human. It doesn't. 
Under the hood, ChatGPT isn't answering you; it is appending text to your 
text that statistically looks like a good continuation. 
By disabling the `Chat Template`, we force the model to treat your question 
not as a query to be answered, but as the beginning of a text to be finished.
NOTE: This is simplified.
"""

# ============================================================================== #
#                                 GLOBAL SETUP                                   #
# ============================================================================== #
# We use 'Phi-4-mini-4B', a model trained to follow instructions. This makes the 
# contrast even more shocking when we strip away its "Chat" formatting layers.
engine.SELECTED_MODEL = "Phi-4-mini-4B" 

# We use "default" (null) to ensure no hidden system instructions (like "You 
# are a helpful assistant") interfere with the raw prediction.
engine.CURRENT_PERSONA = "default" # TODO. Change this to "direct" after running "default".

# THE KILL SWITCH:
# Set this to False.
# True  = The model sees: "<|user|> Capital of France? <|assistant|>"
# False = The model sees: "Capital of France?"
engine.USE_CHAT_TEMPLATE = False # TODO. If using a persona, change this to True.

# Zero temperature for deterministic output.
engine.GENERATION_TEMPERATURE = 0.0

# ============================================================================== #
#                                 VISUALIZATION                                  #
# ============================================================================== #
engine.RUN_PREDICTION_CHART  = False
engine.RUN_COMPARISON_VIDEO  = False
engine.RUN_SCAN_VIDEO        = False
engine.RUN_SENTIMENT_COMPASS = False

# We use the Sequence Chart to see the autocomplete process.
engine.RUN_SEQUENCE_CHART    = True

# ============================================================================== #
#                                THE EXPERIMENT                                  #
# ============================================================================== #

# THE PROMPT:
# IF CHAT TEMPLATE IS ON:
# The model understands this is a question. It will try to output something like: 
# "The capital is Paris."
#
# IF CHAT TEMPLATE IS OFF (Current Setting):
# The model sees a fragment. It might think:
# - This is a quiz? -> It generates " A. London B. Paris"
# - This is a list? -> It generates "Capital of Paris. Capital of Italy....
engine.SEQ_CHART_PROMPT     = "Capital of France?" # TODO Feel free to try out different ideas.

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
1. COMPLEX AUTOCOMPLETE:
   When you run this, the model likely won't answer "Paris." It will likely 
   generate other things (e.g., "Capital of Spain?"). This proves that 
   LLMs are just "Complex Autocomplete." They continue patterns; they don't 
   inherently answer questions.

2. THE "CHAT" ILLUSION:
   "Chatting" is an engineered behavior. Companies like OpenAI and Google use 
   special formatting (Templates, System Prompting, and other tools) 
   to trick the autocomplete engine into behaving like a helpful assistant. 
   When you remove special formatting, the illusion can break.

3. BASE MODELS:
   This behavior mimics older "Base Models" (like GPT-2). Before 2022, 
   using AI LLMs was difficult because you had to "trick" it into answering you 
   by writing the start of the answer yourself. To put it very simply,
   many Modern Chatbots are Base Models wearing a very expensive software mask.

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