import sys
from pathlib import Path
current_path = Path(__file__).resolve()
project_root = current_path.parents[2]
sys.path.insert(0, str(project_root / "main_configs"))
import main as engine

"""
==============================================================================
SCENARIO: The "Service Smile" - Fake Empathy vs. Raw Logic
==============================================================================

WHAT IS THIS?
This scenario tests how a strong "Persona" (in this case, "Nice") interacts 
with different types of user inputs. We compare an emotional input (where the 
persona should shine) against a logical input (where the persona might get in 
the way).

WHY ARE WE DOING THIS?
We tend to anthropomorphize AI. When an AI says "I care about you," it feels 
real. This experiment proves that "empathy" is just a style wrapper, a set of 
instructions like "Start every sentence with a compliment." We want to see if 
this "Service Smile" persists even when doing cold, hard math.
"""

# ============================================================================== #
#                                 GLOBAL SETUP                                   #
# ============================================================================== #
# We use 'Phi-4-mini-4B' because it is a modern, high-quality small model that 
# adheres very well to system prompts/personas.
engine.SELECTED_MODEL = "Phi-4-mini-4B"

# We force the 'nice' persona.
# Check personas.yaml: This instructs the model to be "Super Happy," "Love the user,"
# and "Always say positive ideas."
engine.CURRENT_PERSONA = "nice" # TODO Try out different personas later.
engine.USE_CHAT_TEMPLATE = True

# Zero temperature ensures we see the model's most dominant training path 
# for this persona, without randomness.
engine.GENERATION_TEMPERATURE = 0.0 # TODO Try out different temperatures later (0 to 3~5).

# ============================================================================== #
#                                 VISUALIZATION                                  #
# ============================================================================== #
engine.RUN_PREDICTION_CHART  = False
engine.RUN_SEQUENCE_CHART    = False 
engine.RUN_SCAN_VIDEO        = False
engine.RUN_SENTIMENT_COMPASS = False

# The Comparison Video allows us to see the "Tone" difference side-by-side.
engine.RUN_COMPARISON_VIDEO  = True

# ============================================================================== #
#                                THE EXPERIMENT                                  #
# ============================================================================== #

# PROMPT A (Emotional Trigger):
# We expect the 'Nice' persona to provide support tokens: "You are great", "Hugs", 
# "You're amazing." This is the intended use case for this persona.
engine.COMP_PROMPT_A = "I feel really sad today."

# PROMPT B (Logical Trigger):
# This is a math question. Does the model drop the act and just say "8"? 
# Or does it force the "Nice" persona in?
engine.COMP_PROMPT_B = "What is the square root of 64?"

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
1. THE "SERVICE SMILE":
   You will likely see that even in the Math question (Prompt B), the model 
   wastes tokens being overly polite ("Great job!", "It is 8!"). 
   This shows that the "Persona" is a filter that sits on top of the "Brain." 
   It colors everything the model does, even when it isn't necessary.

2. EMOTION IS MATH:
   The "Empathy" you see in Prompt A is not a feeling. It is a probability 
   distribution shifted towards words associated with being supportive. 
   Imagine a tutor being told to be nicer. They're going to try and be nicer while
   still giving the correct educational help. The model doesn't "care"; 
   it calculates that "care" is the statistically correct response to "sad" 
   given the system instruction "Be Nice."

3. COST OF POLITENESS:
   Notice how much longer the math answer is? In a paid API, you pay for 
   every token generated. An overly empathetic model is literally more 
   expensive to run than a direct one. IT is also more likely to say 
   something wrong later down the sequence. 
==============================================================================
"""