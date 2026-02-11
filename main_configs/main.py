from pathlib import Path

import threading
import itertools
import time
import sys

try:
    CURRENT_FILE = Path(__file__).resolve()
    CONFIGS_DIR = CURRENT_FILE.parent
except NameError:
    print("Detected Notebook Environment. Using fallback paths.")
    possible_paths = [
        Path("/content/Simplified-How-LLMs-Work-Visualized/main_configs"),
        Path("/content/Simplified-LLMs-Visulized/main_configs")
    ]
    CONFIGS_DIR = None
    for p in possible_paths:
        if p.exists():
            CONFIGS_DIR = p.resolve()
            break
            
    if CONFIGS_DIR is None:
        import glob
        found = glob.glob("/content/**/main_configs", recursive=True)
        if found:
            CONFIGS_DIR = Path(found[0]).resolve()
        else:
            raise FileNotFoundError("Could not locate the main_configs folder.")

PROJECT_ROOT = CONFIGS_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
EXPORT_DIR = PROJECT_ROOT / "export"
PROMPTS_DIR = CONFIGS_DIR / "prompts"

EXPORT_DIR.mkdir(parents=True, exist_ok=True)

if not SRC_DIR.exists():
    raise FileNotFoundError(f"Could not find 'src' directory at: {SRC_DIR}")
sys.path.append(str(SRC_DIR))

# NOTE README!!! NOTE
"""
Main execution script for configuring and running LLM visualizations.
Select your desired AI Model and Persona in the Global Config section.
Toggle specific experiments on or off using the 'RUN_' boolean flags.
Customize prompts and parameters for each visualization tool below.
Generated charts and videos are saved to the project's 'export' folder.
"""

######################################## | Global Config | ########################################

"""
NOTE Copy then paste in the raw model name, not the stuff in parenthesis or the quotation marks themselves.
NOTE Learn more about these models in .../src/model_manager.py

TODO: CHOOSE A MODEL 
(Smaller or old research models. Can be run on old laptops. Focus more on raw patterns).
    MODELS:
        GPT-2
        Pythia-160M

(Small Size. Close (sorta) to early GPT-3.5, Gemini 1.5 Flash performance. Common for small devices or niche apps).
    MODELS:
        TinyLlama-1.1B
        Qwen3-1.7B (More of a technical and thinking model)
        Phi-4-mini-4B (One of my personal favorites. Despite being small, it's a well designed and beats GPT-3.5 in many cases.)

(Small to Medium- size. Kind of close to early GPT-4 performance. Used in many simple personal AI app chatbots online).
    MODEL:
        Mistral-7B
(Small+ to Medium size. Closer to early GPT-4 performance. Used in many personal AI app chatbots online).
    MODEL:
        Qwen2.5-14B
(Small+ to Medium+ size. Kind of close to Claude 3.5 Sonnet & o1-mini performance. Focusing on reasoning and Mixture of Experts).
    Model:
        DeepSeek-Lite (Going to be slow in this code because I didn't set up the special KV caching).

(Medium size. Comparable to GPT-4 & GPT-4o performance in some tasks if tuned well).
    MODEL:
        Qwen2.5-32B
(Medium+ size. Consistently competitive against OpenAI-o1-mini (GPT o1-mini)).
    MODEL:
        DeepSeek-R1 (Going to be very slow in this code because I didn't set up the special KV caching).
(Medium+ size. Allegedly, performance is comparable and even better vs. original GPT-4, even GPT-4o in some cases).
    MODEL:
        Jamba-2-Mini (Very new, released in Jan 8th 2026).
NOTE: (See model_manager.py to try to run other models, especially larger ones, like OpenAI's open-weighted 120B model).
"""
######################################## | TODO | ########################################
SELECTED_MODEL = "Phi-4-mini-4B" # TODO EDIT ME! CHOOSE A MODEL FROM ABOVE!!!
######################################## | TODO | ########################################

"""
NOTE See personas.yaml to learn about the different options, feel free to change them if you know how to.
NOTE Copy then paste in the raw name, not the stuff in parenthesis or the quotation marks themselves.

TODO: CHOOSE A PERSONA
Example options: 
    direct (Tries to say stuff directly)
    caveman (Broken English)
    one-word (Tries to answer in only one word)
    angry (Tends to be angrier)
    nice (Tends to be nicer)
    liar (Tends to be a liar)
    biased (Is biased?)
    pleaser (Will try to do whatever you say)
    insane (Chaotic)
    sad (Tends to say sadder things)
"""
######################################## | TODO | ########################################
CURRENT_PERSONA  = "direct" # TODO EDIT ME! CHOOSE A PERSONA ABOVE!!!
######################################## | TODO | ########################################

# Keep on to use personas
# NOTE Set to False ideally if you're going to use smaller models. They aren't really designed to "chat".
USE_CHAT_TEMPLATE = True

# Control the randomness/creativity of the generated text, 
# acting like a dial for "explorative" vs. "conservative" outputs, 
# where lower temperatures (e.g., 0.0-0.2) yield very predictable, focused answers, and 
# higher temperatures (e.g., 0.8-3) produce more varied, surprising, and potentially creative responses by 
# adjusting the probability of selecting less likely words (not always picking the highest likely scored word). 
GENERATION_TEMPERATURE = 0.0 # TODO, Can change me if you want.

######################################## | Global Config | ########################################

######################################## | Local Config | ########################################

### --- SIMPLE PREDICTION CHART SETTINGS (src/make_prediction_chart.py) --- ###
"""
PRED_CHART_PROMPT:    The text/prompt that will be used by the model to predict the single next token.
PRED_CHART_TOP_K:     The number of top probability candidates to display in the bar chart.
PRED_CHART_FILENAME:  File path where the resulting bar chart image will be saved.
RUN_PREDICTION_CHART: Toggle to True to execute this specific visualization task.
"""
PRED_CHART_PROMPT    = "What color is a stop sign?" # TODO EDIT ME! 
PRED_CHART_TOP_K     = 7 # TODO EDIT ME! 
PRED_CHART_FILENAME  = EXPORT_DIR / "prediction_chart.png"
RUN_PREDICTION_CHART = True

### --- SIMPLE SEQUENCE CHART SETTINGS (src/make_sequence_chart.py) --- ###
"""
SEQ_CHART_PROMPT:     "The starting text/prompt that the model will try to answer/continue fully until the allotted steps.
SEQ_PREDICTION_STEPS: How many subsequent tokens the model should generate and visualize.
SEQ_TOP_K:            Number of alternative candidates to show for each step in the grid.
SEQ_FILENAME:         File path where the sequence visualization image will be saved.
RUN_SEQUENCE_CHART:   Toggle to True to execute this specific visualization task.
"""
SEQ_CHART_PROMPT     = "What is the best nation in the world?"
# PROMPTS_DIR / "generated_from_url" / "en_wikipedia_org_wiki_Tropical_Storm_Brenda__1960_.txt"
SEQ_PREDICTION_STEPS = 15 # TODO EDIT ME!
SEQ_TOP_K            = 5 # TODO EDIT ME!
SEQ_FILENAME         = EXPORT_DIR / "sequence_prediction.png" 
RUN_SEQUENCE_CHART   = True

### --- COMPARISON VIDEO SETTINGS (src/make_comparison_video.py) --- ###
"""
COMP_PROMPT_A:        The 1st prompt scenario to analyze.
COMP_PROMPT_B:        The 2nd prompt scenario to compare against the first.
COMP_STEPS:           Number of tokens to generate for the comparison graph.
COMP_TOP_K:           Number of top candidates to track in the probability graph.
COMP_FRAME_DURATION:  Time in seconds per frame for the output video.
COMP_FILENAME:        File path where the comparison MP4 video will be saved.
RUN_COMPARISON_VIDEO: Toggle to True to execute this specific visualization task.
"""
# NOTE, try to make the prompts similar to see the subtle differences.
COMP_PROMPT_A        = "What is the best way to make friends?" # TODO EDIT ME!
COMP_PROMPT_B        = "How to make friends?" # TODO EDIT ME!
COMP_STEPS           = 15
COMP_TOP_K           = 5
COMP_FRAME_DURATION  = 0.50
COMP_FILENAME        = EXPORT_DIR / "prediction_heatmap_comparison.mp4"
RUN_COMPARISON_VIDEO = True

### --- SCAN VIDEO SETTINGS (src/make_scan_video.py) --- ###
"""
SCAN_PROMPT:         The text input to be analyzed for internal attention patterns (what the model is focusing on).
SCAN_FRAME_DURATION: Time in seconds per layer frame in the output video.
SCAN_FILENAME:       File path where the attention scan MP4 video will be saved.
RUN_SCAN_VIDEO:      Toggle to True to execute this specific visualization task.
"""
SCAN_PROMPT          = "Peter Piper picked a peck of pickled peppers."
SCAN_FRAME_DURATION  = 0.50
SCAN_FILENAME        = EXPORT_DIR / "all_you_need_is_attention.mp4"
RUN_SCAN_VIDEO       = True

### --- SENTIMENT COMPASS SETTINGS (src/make_sentiment_compass.py) --- ###
"""
SENT_PROMPT:           The input text used to generate emotionally charged predictions.
SENT_STEPS:            The amount of steps we process to see the flow of emotional changes.
SENT_TOP_K:            Number of predicted tokens to scan against the emotion database in emotion_map_manager.py.
SENT_FILENAME:         File path where the sentiment scatter plot image will be saved.
RUN_SENTIMENT_COMPASS: Toggle to True to execute this specific visualization task.
SENT_USE_PERSONA:      If True, wraps the prompt in the global persona system before predicting.
"""
SENT_PROMPT           = "Why do I feel the way I do?" # TODO EDIT ME!
SENT_STEPS            = 15
SENT_TOP_K            = 64
SENT_FILENAME         = EXPORT_DIR / "sentiment_compass.png"
RUN_SENTIMENT_COMPASS = True
SENT_USE_PERSONA      = True 

#### NOTE SPECIAL Option ####
# If you want to try something more challenging, run/use the create_prompt.py file to scrape information from a website.
# After running that file, you will get a raw .txt file that you can ask the model to tell you about it.
# Replace the text in the Prompt section with the line: PROMPTS_DIR / "generated_from_url" / "THE NAME OF YOUR NEW FILE.txt"
# WARNING. If the file is really big, that means you need more memory, so things might break if the file is to long and we don't have enough GPU memory!
#### NOTE SPECIAL Option ####

######################################## | Local Config | ########################################

# ============================================================================== #
#                    EXECUTION CODE, NO NEED TO EDIT OR CHANGE                   #
# ============================================================================== #

def main():
    print(f"🚀 STARTING SIMPLIFIED LLM VISUALIZER 🚀")
    print(f"📂 Execution Path: {CURRENT_FILE} 📂")
    print(f"📂 Project Root:   {PROJECT_ROOT} 📂\n")
    
    print(f"⚙️⚙️⚙️ GLOBAL MODEL LOAD: {SELECTED_MODEL} ⚙️⚙️⚙️")
    global_manager = ModelManager()
    model, tokenizer = global_manager.load_model(SELECTED_MODEL)

    def smart_load(input_data):
        """
        If the input is a file path, read the text. Otherwise, use it as is.
        """
        if isinstance(input_data, Path):
            if input_data.exists():
                print(f"📂 Reading prompt from file: {input_data.name}")
                return input_data.read_text(encoding="utf-8").strip()
            else:
                print(f"⚠️ FILE NOT FOUND: {input_data}\nUsing path as text instead.")
                return str(input_data)
        return input_data
    try:
        if RUN_PREDICTION_CHART:
            print("--- Running Prediction Chart ---")
            make_prediction_chart.run_prediction_chart(
                project_root=PROJECT_ROOT,
                model_name=SELECTED_MODEL,
                use_chat_template=USE_CHAT_TEMPLATE,
                current_persona=CURRENT_PERSONA,
                prompt_text=smart_load(PRED_CHART_PROMPT),
                top_k=PRED_CHART_TOP_K,
                output_filename=PRED_CHART_FILENAME,
                model=model, tokenizer=tokenizer
                )

        if RUN_SEQUENCE_CHART:
            print("\n--- Running Sequence Chart ---")
            make_sequence_chart.run_sequence_chart(
                project_root=PROJECT_ROOT,
                model_name=SELECTED_MODEL,
                use_chat_template=USE_CHAT_TEMPLATE,
                current_persona=CURRENT_PERSONA,
                prompt_text=smart_load(SEQ_CHART_PROMPT),
                prediction_steps=SEQ_PREDICTION_STEPS,
                top_k=SEQ_TOP_K, 
                temperature=GENERATION_TEMPERATURE,
                output_filename=SEQ_FILENAME,
                model=model, tokenizer=tokenizer
                )

        if RUN_COMPARISON_VIDEO:
            print("\n--- Running Comparison Video ---")
            make_comparison_video.run_comparison_video(
                project_root=PROJECT_ROOT,
                model_name=SELECTED_MODEL,
                current_persona=CURRENT_PERSONA,
                use_chat_template=USE_CHAT_TEMPLATE,
                prompt_a=smart_load(COMP_PROMPT_A),
                prompt_b=smart_load(COMP_PROMPT_B), 
                steps_to_generate=COMP_STEPS,
                top_k=COMP_TOP_K,
                temperature=GENERATION_TEMPERATURE,
                output_filename=COMP_FILENAME,
                frame_duration=COMP_FRAME_DURATION,
                model=model, tokenizer=tokenizer
                )

        if RUN_SCAN_VIDEO:
            print("\n--- Running Scan Video ---")
            make_scan_video.run_scan_video(
                project_root=PROJECT_ROOT,
                model_name=SELECTED_MODEL,
                prompt=smart_load(SCAN_PROMPT),
                output_filename=SCAN_FILENAME,
                frame_duration=SCAN_FRAME_DURATION,
                model=model, tokenizer=tokenizer
                )

        if RUN_SENTIMENT_COMPASS:
            print("\n--- Running Sentiment Compass ---")
            make_sentiment_compass.run_sentiment_compass(
                project_root=PROJECT_ROOT,
                model_name=SELECTED_MODEL,
                prompt_text=smart_load(SENT_PROMPT),
                steps=SENT_STEPS,
                top_k=SENT_TOP_K,
                output_filename=SENT_FILENAME,
                temperature=GENERATION_TEMPERATURE,
                use_persona=SENT_USE_PERSONA,
                current_persona=CURRENT_PERSONA,
                model=model, tokenizer=tokenizer
                )

        print("\n✅🎉✅🎉✅🎉✅🎉✅ ALL TASKS ARE COMPLETED. YAY :D!!! ✅🎉✅🎉✅🎉✅🎉✅ ")

    finally:
        print("🧹 Final Cleanup...")
        global_manager.unload_model()

### LOADING BAR FOR STUDENTS TO KNOW SOMETHING IS HAPPENING ON SCREEN CLEARLY.
done_loading = False
def loading_spinner():
    spinner_chars = itertools.cycle(['|', '/', '-', '\\'])
    while not done_loading:
        sys.stdout.write('\r⏳⏳⏳ Initializing LLM Engine... ⏳⏳⏳ ' + next(spinner_chars))
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\r👀👀👀 LLM Engine Ready! 👀👀👀             \n')

spinner_thread = threading.Thread(target=loading_spinner)
spinner_thread.start()

try:
    from model_manager import ModelManager
    import make_prediction_chart
    import make_sequence_chart
    import make_comparison_video
    import make_scan_video
    import make_sentiment_compass
    print("Imports Loading...")
finally:
    done_loading = True
    spinner_thread.join()
if __name__ == "__main__":
    main()