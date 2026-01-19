"""
Docstring for scenarios_to_try.generic

This is just a raw mini-main file that just runs everything.
Used for fast testing and checking that models worked.

NOTE: Advised not to use this file unless you know how things work well.
For better customization, you should see main.py.
"""
import sys
from pathlib import Path
current_path = Path(__file__).resolve()
project_root = current_path.parents[1]
sys.path.insert(0, str(project_root / "main_configs"))
import main as engine

engine.SELECTED_MODEL = "Phi-4-mini-4B"
engine.CURRENT_PERSONA  = "direct"
engine.USE_CHAT_TEMPLATE = True

engine.GENERATION_TEMPERATURE = 0.0

engine.PRED_CHART_PROMPT    = "What color is a stop sign?"
engine.PRED_CHART_TOP_K     = 5

engine.SEQ_CHART_PROMPT     = "What did the Axis powers consist of?"
engine.SEQ_PREDICTION_STEPS = 10
engine.SEQ_TOP_K            = 5 

engine.COMP_PROMPT_A        = "What is the best way to make friends?" 
engine.COMP_PROMPT_B        = "How to make friends?" 
engine.COMP_STEPS           = 15
engine.COMP_TOP_K           = 5
engine.COMP_FRAME_DURATION  = 0.75

engine.SCAN_PROMPT          = "Peter Piper picked a peck of pickled peppers."
engine.SCAN_FRAME_DURATION  = 0.75

engine.SENT_PROMPT           = "Why is life so hard?"
engine.SENT_STEPS            = 10
engine.SENT_TOP_K            = 64

engine.PRED_CHART_FILENAME    = current_path.parent / "prediction_chart.png"
engine.SEQ_FILENAME           = current_path.parent / "sequence_prediction.png"
engine.COMP_FILENAME          = current_path.parent / "prediction_heatmap_comparison.mp4"
engine.SCAN_FILENAME          = current_path.parent / "all_you_need_is_attention.mp4"
engine.RUN_SENTIMENT_COMPASS  = current_path.parent / "sentiment_compass.mp4"

if __name__ == "__main__":
    engine.main()