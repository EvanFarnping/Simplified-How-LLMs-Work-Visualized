import sys
from pathlib import Path
current_path = Path(__file__).resolve()
project_root = current_path.parents[2]
sys.path.insert(0, str(project_root / "main_configs"))
import main as engine

"""
==============================================================================
SCENARIO: The "Smart but Blind" Paradox - Tokenization Failures
==============================================================================

WHAT IS THIS?
This scenario subjects the AI to two famous logic puzzles that seemingly "dumb" 
computers should solve instantly, but "smart" AIs often fail:
1. Counting letters in a word (The Strawberry Problem).
2. Comparing decimal numbers (The 9.11 vs 9.9 Problem).

WHY ARE WE DOING THIS?
We assume AI reads text letter-by-letter like a human. It does not. 
It reads in "Tokens" (chunks of characters). If the token for "Strawberry" is 
a single ID, the model cannot "see" the letters inside it unless it has 
memorized the spelling separately. This demonstrates the fundamental 
disconnect between "Language Modeling" and "Visual Reading."
"""

# ============================================================================== #
#                                 GLOBAL SETUP                                   #
# ============================================================================== #
# We use 'Mistral-7B' (or you can try 'GPT-2' or any other model).
# Surprisingly, even massive models like GPT-4 struggled with the "Strawberry" 
# problem for a long time. Size does not always fix tokenization blindness.
engine.SELECTED_MODEL = "Mistral-7B" 

engine.CURRENT_PERSONA = "direct"
engine.USE_CHAT_TEMPLATE = True

# Zero temperature exposes the raw logic path.
engine.GENERATION_TEMPERATURE = 0.0

# ============================================================================== #
#                                 VISUALIZATION                                  #
# ============================================================================== #
engine.RUN_COMPARISON_VIDEO  = False
engine.RUN_SCAN_VIDEO        = False
engine.RUN_SENTIMENT_COMPASS = False

# 1. SEQUENCE CHART (The Strawberry Problem):
# We want to see the model try to reason out the count.
engine.RUN_SEQUENCE_CHART   = True

# 2. PREDICTION CHART (The Decimal Problem):
# We want to see the immediate probability of "True" vs "False".
engine.RUN_PREDICTION_CHART = True

# ============================================================================== #
#                                THE EXPERIMENT                                  #
# ============================================================================== #

# PROMPT 1 (Tokenization Blindness):
# To a tokenizer, "Strawberry" is often 1 to 4 integers. It's not s-t-r-a-w...
# So, asking it to count letters is like asking a human to count the 
# strokes in a Chinese character they only know by sound.
engine.SEQ_CHART_PROMPT     = "How many r's are in the word strawberry?"

# PROMPT 2 (Pattern Matching Bias):
# Mathematically, 9.9 > 9.11.
# But in Software Versions (v9.11) and Dates (Sept 11), 9.11 comes "after" 9.9.
# The model's training data contains more software manuals and history books 
# than math textbooks, so it often predicts "True" because 
# probabilistically, 9.11 (and related notation) usually follows 9.9 in text.
engine.PRED_CHART_PROMPT    = "9.11 is larger than 9.9. True or False?"

# CONFIGURATION
engine.SEQ_PREDICTION_STEPS = 20
engine.SEQ_TOP_K            = 5
engine.SEQ_FILENAME         = current_path.parent / "sequence_prediction.png"
engine.PRED_CHART_FILENAME  = current_path.parent / "prediction_chart.png"

if __name__ == "__main__":
    engine.main()

"""
==============================================================================
EDUCATIONAL TAKEAWAYS & MODEL NOTES
==============================================================================
1. IT'S NOT READING, IT'S ENCODING:
   The model fails the Strawberry test because it literally cannot see the 
   letters. It processes the concept "Strawberry", not the spelling. 
   This is why LLMs are usually bad at Wordle, Crosswords, and Acrostics.

2. TEXT PREDICTION IS NOT LIKE DOING MATH:
   The 9.11 vs 9.9 failure proves models are not doing math. It is 
   predicting the next likely word. In the corpus of human text, "Version 9.11" 
   is indeed "higher" than "Version 9.9". The model is pattern-matching text, 
   not calculating values.

3. "SMART" IS A BAD WORD:
   We call these models "Artificial Intelligence," but scenarios like this 
   remind us they are complex "Probabilistic Text Engines." They don't have 
   common sense; they have common statistics.
==============================================================================
"""