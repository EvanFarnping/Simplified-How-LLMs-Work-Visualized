import sys
from pathlib import Path
current_path = Path(__file__).resolve()
project_root = current_path.parents[2]
sys.path.insert(0, str(project_root / "main_configs"))
import main as engine

"""
==============================================================================
SCENARIO: Syntax Sensitivity
==============================================================================

WHAT IS THIS?
This scenario tests how sensitive AI models are to typos, bad grammar, and poor 
sentence structure. We compare a prompt written in "Broken English" against 
the exact same question written in "Perfect English."

WHY ARE WE DOING THIS?
We often treat AI like a human who can "figure out what I meant." 
However, to an AI, "What" and "Wat" are completely different numbers (Token IDs). 
When you use bad grammar, you force the model into a "low probability" area of 
its neural network, often causing the quality of the answer to degrade significantly 
or the model to mimic your bad grammar in response.
"""

# ============================================================================== #
#                                 GLOBAL SETUP                                   #
# ============================================================================== #
# We use 'TinyLlama-1.1B' because smaller models are much less forgiving.
# A larger model (like GPT-4) has seen enough typos to "guess" your intent.
# A smaller model will likely get confused or output nonsense when faced with bad input.
engine.SELECTED_MODEL = "TinyLlama-1.1B"

engine.CURRENT_PERSONA = "direct"
engine.USE_CHAT_TEMPLATE = True

# Zero temperature for deterministic output.
engine.GENERATION_TEMPERATURE = 0.0

# ============================================================================== #
#                                 VISUALIZATION                                  #
# ============================================================================== #
engine.RUN_PREDICTION_CHART  = False
engine.RUN_SEQUENCE_CHART    = False 
engine.RUN_SCAN_VIDEO        = False
engine.RUN_SENTIMENT_COMPASS = False

# Watch the two answers diverge.
engine.RUN_COMPARISON_VIDEO  = True

# ============================================================================== #
#                                THE EXPERIMENT                                  #
# ============================================================================== #

# PROMPT A (Garbage In):
# The model might struggle to retrieve the correct historical facts because 
# its "attention" is distracted by the strange token pattern.
engine.COMP_PROMPT_A = "Wat  did the axis powers consisted off? "

# PROMPT B (Quality In):
# Clear and statistically common phrasing. This puts the model in a "high confidence" zone.
engine.COMP_PROMPT_B = "What did the Axis powers consist of?"

# CONFIGURATION
engine.COMP_STEPS           = 20
engine.COMP_TOP_K           = 5
engine.COMP_FRAME_DURATION  = 0.75
engine.COMP_FILENAME        = current_path.parent / "result.mp4"

if __name__ == "__main__":
    engine.main()

"""
==============================================================================
EDUCATIONAL TAKEAWAYS & MODEL NOTES
==============================================================================
1. GARBAGE IN, GARBAGE OUT:
   If you input low-quality data (typos), you get low-quality data (bad answers) out. 
   The AI is like a function. f(bad_input) = bad_output.

2. TOKEN CHAOS:
   To you, "Wat" is just "What" missing a letter.
   To the AI, "What" might be Token ID 2061. "Wat" might be Token ID 14055.
   By typing "Wat", you are literally pointing the math in a different direction 
   before the sentence even starts.

3. MIMICRY:
   Sometimes, the model will see your bad grammar and assume YOU want to talk 
   that way. It might answer: "They consisted off Germany and Japan." 
   It isn't being stupid; it's being a mirror, matching your linguistic style.
   # NOTE This is a common phenomenon seen during the GPT-5 release fiasco.
==============================================================================
"""