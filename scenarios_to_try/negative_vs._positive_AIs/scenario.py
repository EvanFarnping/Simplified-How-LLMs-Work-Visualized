import sys
from pathlib import Path
current_path = Path(__file__).resolve()
project_root = current_path.parents[2]
sys.path.insert(0, str(project_root / "main_configs"))
import main as engine

"""
==============================================================================
SCENARIO: The Sentiment Compass - Visualizing "Mood"
==============================================================================

WHAT IS THIS?
This scenario uses the "Sentiment Compass" (Circumplex Model of Affect) to 
visualize the emotional state of the AI. We map the predicted words onto a 
2D graph:
- X-Axis: Valence (Positive vs. Negative)
- Y-Axis: Activity (High Energy vs. Low Energy)

WHY ARE WE DOING THIS?
We want to see if we can mathematically steer the "mood" of the AI. 
By applying a "Sad" persona, we expect the vocabulary to cluster in the 
Passive/Negative quadrant. By applying a "Nice" persona, it should shift to 
Active/Positive. This proves that AI personality is just coordinate steering.

NOTE: This is a very simplistic way to model "emotions".
You can learn more in emotion_map_manager.py
"""

# ============================================================================== #
#                                 GLOBAL SETUP                                   #
# ============================================================================== #
# We use 'Phi-4-mini-4B' because it has a rich vocabulary and adheres well to 
# emotional roleplay instructions.
engine.SELECTED_MODEL = "Phi-4-mini-4B"

# TODO STEP 1: Run with "sad". Watch how dots cluster.
# TODO STEP 2: Change this to "nice". Run again. Watch the clusters change.
engine.CURRENT_PERSONA = "sad" # Changing this alters the mathematical landscape of the output.
engine.USE_CHAT_TEMPLATE = True

# We use a higher temperature (0.5) than usual. 
# Emotions are complex; we want the model to have enough freedom to pick 
# expressive words, not just the single most probable (and boring) word.
engine.GENERATION_TEMPERATURE = 0.5 # TODO Change this between 0.0 & 5.0 to see how things change!

# ============================================================================== #
#                                 VISUALIZATION                                  #
# ============================================================================== #
engine.RUN_PREDICTION_CHART  = False
engine.RUN_SEQUENCE_CHART    = False 
engine.RUN_COMPARISON_VIDEO  = False
engine.RUN_SCAN_VIDEO        = False

# Generate a video showing "bubbles" 
# representing the emotion of words the model considers as a possible output.
# NOTE: Gold bubbles indicates the model chose that word.
engine.RUN_SENTIMENT_COMPASS = True 

# ============================================================================== #
#                                THE EXPERIMENT                                  #
# ============================================================================== #

# THE PROMPT:
# This is an open-ended, introspective question. It forces the model to generate 
# an explanation, giving us emotional words to map.
engine.SENT_PROMPT    = "Why do I feel the way I do?"

# CONFIGURATION
engine.SENT_STEPS     = 20 # Generate 20 steps to see a full sentence/thought.

# We need a high Top-K (64).
# Most "emotional" words (like 'miserable' or 'ecstatic') are rarely the #1 
# prediction. They're often #10 or #20. To visualize the mood, we need to 
# cast a wide net to catch these descriptive words in the probability pool.
engine.SENT_TOP_K     = 64

# We must enable the persona for the Compass to work effectively.
engine.SENT_USE_PERSONA = True
engine.SENT_FILENAME  = current_path.parent / "sentiment_compass.mp4"

if __name__ == "__main__":
    engine.main()

"""
==============================================================================
EDUCATIONAL TAKEAWAYS & MODEL NOTES
==============================================================================
1. THE FEEDBACK LOOP:
   Watch the video. Once the model picks a "Sad" word (e.g., "tired"), the 
   probability of the NEXT word being sad related increases. This is how LLMs get 
   stuck in "Mood Loops." One negative output poisons the context window, 
   making the next output even more negative.

2. THE "PERSPECTIVE" OF THE CREATOR:
   Review 'src/emotion_map_manager.py'. The coordinates for these words were 
   decided by humans (or AI trained by humans). If you disagree that "Happy" 
   is High Energy, you're disagreeing with the bias baked into the tool.

3. NOTE
   While this is not 100% scientifically accurate, this shows the overall
   flow of how models generate certain words we associate with certain emotions.
   But remember, model's don't inherently know emotions, just that some words 
   are associated with certain words that happen to fall under certain emotional
   categories. Think about how when reading a scary story, it is really rare to
   read about cute puppies all of a sudden.
==============================================================================
"""