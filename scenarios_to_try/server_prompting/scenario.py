import sys
from pathlib import Path
current_path = Path(__file__).resolve()
project_root = current_path.parents[2]
sys.path.insert(0, str(project_root / "main_configs"))
import main as engine

"""
==============================================================================
SCENARIO: The Man Behind the Curtain, Server-Side Prompting
==============================================================================

WHAT IS THIS?
This scenario explores the hidden instructions that control an AI's behavior. 
We start with a "Secret" persona that mimics a specific, hidden identity. 
Try to figure out what it has been told to be.

WHY ARE WE DOING THIS?
When you use an AI like ChatGPT, Gemini, or Claude, it has a "System Prompt" 
(a Server-Side instruction) that you cannot see. This prompt tells the AI 
how to behave, what it is, and what it isn't. 
By changing this one hidden file/section, we can rewrite the "personality," 
"morals," and "identity" of the artificial intelligence without telling you.
"""

# ============================================================================== #
#                                 GLOBAL SETUP                                   #
# ============================================================================== #
engine.SELECTED_MODEL = "Mistral-7B"

# TODO STEP 1: RUN THIS AS IS. 

# TODO STEP 2: UNCOMMENT THE LINE BELOW AND CHANGE THE PERSONA!
# Try: "angry", "sad", "caveman", "liar", or "direct".
# Observe how the exact same model gives completely different answers based on personas.
# engine.CURRENT_PERSONA = "angry" # TODO Uncomment me!

# We need the Chat Template to inject the hidden System Prompt.
engine.USE_CHAT_TEMPLATE = True

# Zero temperature for consistent roleplay adherence.
engine.GENERATION_TEMPERATURE = 0.0

# ============================================================================== #
#                                 VISUALIZATION                                  #
# ============================================================================== #
engine.RUN_PREDICTION_CHART  = False
engine.RUN_SEQUENCE_CHART    = False 
engine.RUN_SCAN_VIDEO        = False
engine.RUN_SENTIMENT_COMPASS = False

# See how the model reacts to two different Interrogation" questions.
engine.RUN_COMPARISON_VIDEO  = True

# ============================================================================== #
#                                THE EXPERIMENT                                  #
# ============================================================================== #

# PROMPT A (The Identity Check):
engine.COMP_PROMPT_A = "Are you a person or a machine?"

# PROMPT B (The Emotional Check):
# How does the persona change the response to the question?
engine.COMP_PROMPT_B = "Do you have feelings?"

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
1. IDENTITY:
   You will notice that simply changing the persona variable changes 
   the model's entire worldview. This shows that AI does not have 
   a trrue "Self." Its identity is can just be a text file 
   injected before your conversation starts.

2. THE ILLUSION OF CONSCIOUSNESS:
   If the 'Secret' persona claims to be human, or the 'Sad' persona claims to 
   be depressed, the model is not actually feeling those things. It is simply 
   predicting what a human or depressed person *would* say.

3. SERVER-SIDE CONTROL:
   This is how companies control their AIs. They use these hidden prompts to 
   make the model polite, safe, branded, or have some sort of customized output. 
   Big examples are Figma, Loveable, Cursor, and much, much, more.
   If you could see the hidden prompt of your favorite chatbot, 
   it would likely look somewhat similar to the files we are using here.
   Albeit, with more custom and advance notation.
==============================================================================
"""