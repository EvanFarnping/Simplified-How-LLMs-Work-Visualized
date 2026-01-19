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
"""