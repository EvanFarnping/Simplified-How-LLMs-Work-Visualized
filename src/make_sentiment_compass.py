from emotion_map_manager import EMOTION_MAP, NEGATION_WORDS
from model_manager import ModelManager
from predictor import Predictor
from pathlib import Path
import utils

import matplotlib.patheffects as pe 
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import textwrap
import shutil

# ============================================================================== #
#                DO NOT TOUCH UNLESS YOU KNOW WHAT YOU ARE DOING!                #
# ============================================================================== #

def get_coords(word):
    clean = utils.clean_token(word, mode='search').lower()
    return EMOTION_MAP.get(clean)

def run_sentiment_compass(
    project_root,
    model_name="Phi-4-mini-4B",
    prompt_text="I am feeling very",
    steps=10, 
    top_k=50,
    temperature=0.5,
    output_filename="sentiment_compass.png",
    use_persona=False,
    current_persona="default",
    model = None,
    tokenizer = None
    ):
    
    project_root = Path(project_root)
    config_dir = project_root / "main_configs"
    persona_file = config_dir / "personas.yaml"
    output_path = project_root / output_filename
    
    if steps > 1 and output_path.suffix != ".mp4":
        output_path = output_path.with_suffix(".mp4")
        
    temp_dir = project_root / "temp_sentiment_frames"
    local_manager = None

    if model is None or tokenizer is None:
        print(f"⚙️⚙️⚙️ Loading {model_name}... ⚙️⚙️⚙️")
        local_manager = ModelManager()
        model, tokenizer = local_manager.load_model(model_name)
    
    try:
        if use_persona:
            print(f"🎭🎭🎭 Applying Persona: {current_persona} 🎭🎭🎭")
            persona_text = utils.load_persona(persona_file, current_persona)
            user_question = prompt_text

            chat_history = utils.format_as_chat_robust(tokenizer, user_question, persona_text)
            final_prompt = chat_history 
        else:
            final_prompt = prompt_text 

        # DATA GATHERING
        if steps > 1:
            print(f"🧠🧠🧠 Generating Sentiment Trajectory ({steps} steps)... 🧠🧠🧠")
            history = Predictor.predict_sequence(
                model, tokenizer, final_prompt, 
                steps=steps, k=top_k, temperature=temperature
            )
        else:
            print(f"🧠🧠🧠 Predicting single snapshot... 🧠🧠🧠")
            tokens, probs = Predictor.get_top_k_tokens(model, tokenizer, final_prompt, k=top_k)
            history = [{
                "chosen_token": "", 
                "candidates": list(zip(tokens, probs))
            }]

        # PLOTTING
        if temp_dir.exists(): shutil.rmtree(temp_dir)
        temp_dir.mkdir()
        filenames = []
        
        current_sentence = prompt_text
        
        print(f"🎨🎨🎨 Rendering {len(history)} frames... 🎨🎨🎨")
        
        for i, step_data in enumerate(history):
            active_context = current_sentence.lower().strip()
            context_words = active_context.replace('.', ' ').replace(',', ' ').split()

            candidates = step_data['candidates']
            top_7_tokens = [c[0] for c in candidates[:7]] 
            print(f"Step {i+1} Raw Tokens: {top_7_tokens}")

            valence_multiplier = 1.0
            if context_words:
                lookback_window = context_words[-3:] 
                if any(word in NEGATION_WORDS for word in lookback_window):
                    valence_multiplier = -1.0

            plt.style.use('dark_background')
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Segoe UI Emoji', 'Microsoft YaHei', 'SimHei', 
                                               'Arial Unicode MS', 'DejaVu Sans', 'Arial', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False

            fig, ax = plt.subplots(figsize=(12, 12))
            
            # Grid
            ax.axhline(0, color='white', linestyle='--', linewidth=2, alpha=0.8)
            ax.axvline(0, color='white', linestyle='--', linewidth=2, alpha=0.8)
            
            label_style = dict(fontsize=14, fontweight='bold', alpha=0.9)
            ax.text(0.95, 0.95, "ACTIVE & POSITIVE\n(Excited)", 
                    ha='right', va='top', color='#00e676', **label_style)
            ax.text(-0.95, 0.95, "ACTIVE & NEGATIVE\n(Anxious/Angry)", 
                    ha='left', va='top', color='#ff5252', **label_style)
            ax.text(-0.95, -0.95, "PASSIVE & NEGATIVE\n(Sad/Depressed)", 
                    ha='left', va='bottom', color='#b39ddb', **label_style)
            ax.text(0.95, -0.95, "PASSIVE & POSITIVE\n(Calm)", 
                    ha='right', va='bottom', color='#4fc3f7', **label_style)
            
            # Identify Winner
            chosen_token_raw = step_data.get('chosen_token', '')
            chosen_word_clean = utils.clean_token(chosen_token_raw, 'visualize')

            # Extract Candidates
            candidates = step_data['candidates']
            plot_data = []
            for tok, prob in candidates:
                coords = get_coords(tok)
                if coords:
                    valence, arousal = coords
                    if valence_multiplier == -1.0:
                        if valence > 0:
                            # "Not happy" -> Negative (Flip is correct)
                            final_valence = valence * -1.0
                        else:
                            # "Not sad" or "Oh no... bad" -> Keep Negative (Don't flip)
                            final_valence = valence 
                    else:
                        final_valence = valence

                    plot_data.append({
                        "word": utils.clean_token(tok, 'visualize'), 
                        "x": final_valence,
                        "y": arousal, 
                        "prob": prob
                    })

            # Plot Bubbles
            if plot_data:
                for item in plot_data:
                    is_selected_word = (item['word'] == chosen_word_clean)

                    size = (item['prob'] * 1700) + 300
                    
                    if is_selected_word:
                        color = '#FFD700'
                        alpha = 1.0       
                        zorder = 20       
                        linewidth = 4     
                        edgecolor = 'white'
                    else:
                        if item['x'] < -0.1: 
                            color = '#ff5252' 
                        elif item['x'] > 0.1: 
                            color = '#00e676' 
                        else: 
                            color = '#ffd740'
                        alpha = 0.8
                        zorder = 10
                        linewidth = 2
                        edgecolor = 'white'
                    
                    ax.scatter(item['x'], item['y'], s=size, c=color, alpha=alpha, 
                               edgecolors=edgecolor, linewidth=linewidth, zorder=zorder)
                    
                    txt = ax.text(item['x'], item['y'], item['word'], ha='center', va='center', 
                            color='white', fontsize=12, fontweight='bold', zorder=zorder+1)
                    txt.set_path_effects([pe.withStroke(linewidth=3, foreground='black')])
            else:
                ax.text(0, 0, "No Emotional Words Detected", ha='center', va='center', color='red', fontsize=20)

            ax.set_xlim(-1.15, 1.15)
            ax.set_ylim(-1.15, 1.15)
            ax.set_xlabel("VALENCE")
            ax.set_ylabel("ACTIVITY")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)

            info_line = f"Model: {model_name} | Persona: {current_persona}"
            
            if steps > 1:
                chosen = step_data['chosen_token']
                if chosen: 
                    current_sentence += utils.clean_token(chosen, 'readable')
                
                wrapper = textwrap.TextWrapper(width=60)
                wrapped_text = wrapper.fill(text=f"\"{current_sentence}\"")
                title_text = f"Projected Emotional Vocabulary\n{info_line}\nStep {i+1}/{len(history)}\n{wrapped_text}"
            else:
                title_text = f"Projected Emotional Vocabulary\n{info_line}\nInput: \" {prompt_text}...\""

            ax.set_title(title_text, fontsize=18, pad=20, fontweight='bold', color='white')

            disclaimer_text = (
                "This mapping uses keyword lookup with basic negation. "
                "It estimates sentiment but doesn't capture nuances like sarcasm, double negatives, or long/complex contexts."
            )
            fig.text(0.5, 0.05, disclaimer_text, 
                     ha='center', va='bottom', 
                     fontsize=11, color='#dadada', style='italic')

            if steps > 1:
                plt.subplots_adjust(top=0.80, bottom=0.1) 
                fname = temp_dir / f"frame_{i:03d}.png"
                plt.savefig(fname, dpi=120) 
                filenames.append(str(fname))
                plt.close(fig)
            else:
                plt.subplots_adjust(top=0.85, bottom=0.1)
                
                plt.savefig(output_path, dpi=300) 
                print(f"✅ Saved Snapshot to: {output_path}")
                plt.close(fig)

        # Stitch
        if steps > 1 and filenames:
            print("🎞️🎞️🎞️ Stitching Trajectory MP4... 🎞️🎞️🎞️")
            fps = 1.0
            with imageio.get_writer(output_path, 
                                    fps=fps, 
                                    macro_block_size=None,
                                    ffmpeg_log_level="error") as writer:
                
                for filename in filenames:
                    writer.append_data(imageio.imread(filename))
            print(f"✅✅✅ Saved Video to: {output_path} ✅✅✅")
            shutil.rmtree(temp_dir)

    finally:
        if local_manager: 
            local_manager.unload_model()