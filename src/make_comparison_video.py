from matplotlib.colors import LinearSegmentedColormap
from model_manager import ModelManager
from predictor import Predictor
from pathlib import Path
import utils

import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np
import textwrap 
import shutil

# ============================================================================== #
#                DO NOT TOUCH UNLESS YOU KNOW WHAT YOU ARE DOING!                #
# ============================================================================== #

def run_comparison_video(
    project_root,
    model_name="Phi-3-mini",
    current_persona="default",
    use_chat_template=True,
    prompt_a="The best way to make friends is",
    prompt_b="The worst way to make friends is",
    steps_to_generate=10,
    top_k=10,
    temperature=0.7, 
    output_filename="prediction_heatmap_comparison.mp4",
    frame_duration=0.8,
    model = None,
    tokenizer = None
    ):

    project_root = Path(project_root)
    config_dir = project_root / "main_configs"
    persona_file = config_dir / "personas.yaml"
    output_path = project_root / output_filename
    temp_dir = project_root / "temp_compare_frames"

    local_manager = None

    if model is None or tokenizer is None:
        print(f"⚙️⚙️⚙️ Loading {model_name}... ⚙️⚙️⚙️")
        local_manager = ModelManager()
        model, tokenizer = local_manager.load_model(model_name)
    
    try:
        if use_chat_template:
            persona_text = utils.load_persona(persona_file, current_persona)
            text_a = utils.format_as_chat_robust(tokenizer, prompt_a, persona_text)
            text_b = utils.format_as_chat_robust(tokenizer, prompt_b, persona_text)
        else:
            text_a = prompt_a
            text_b = prompt_b

        if temp_dir.exists(): shutil.rmtree(temp_dir)
        temp_dir.mkdir()
        
        # GENERATE BOTH SEQUENCES
        print(f"⚡⚡⚡ Generating Scenario A... ⚡⚡⚡")
        history_a = Predictor.predict_sequence(
            model, tokenizer, text_a, 
            steps=steps_to_generate, k=top_k, temperature=temperature
        )
        
        print(f"⚡⚡⚡ Generating Scenario B... ⚡⚡⚡")
        history_b = Predictor.predict_sequence(
            model, tokenizer, text_b, 
            steps=steps_to_generate, k=top_k, temperature=temperature
        )
        
        # RENDER VIDEO FRAMES
        max_steps = max(len(history_a), len(history_b))
        
        cmap_green = LinearSegmentedColormap.from_list("custom_green", ["#000000", "#00e676"])
        cmap_red = LinearSegmentedColormap.from_list("custom_red", ["#000000", "#ff5252"])
        
        filenames = []

        fig_height = int(np.ceil(max(8.0, max_steps * 0.7)))

        # Ensure height is even to avoid pixel division
        if fig_height % 2 != 0:
            fig_height += 1

        print(f"🎨🎨🎨 Rendering {max_steps} comparison frames... 🎨🎨🎨")
        # print(f"{fig_height} inches)... 🎨🎨🎨") # Debug height to avoid division errors


        for step in range(max_steps): 
            plt.style.use('dark_background')
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Segoe UI Emoji',   
                                               'Arial Unicode MS', 'DejaVu Sans', 'Arial', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            fig, axes = plt.subplots(1, 2, figsize=(18, fig_height))
            
            header_inches = 2.2 
            top_margin = 1.0 - (header_inches / fig_height)
            
            if top_margin < 0.7: 
                top_margin = 0.7
            
            title_y_pos = 1.0 - (0.5 / fig_height)
            
            fig.suptitle(f"Comparison Generation | Model: {model_name} | Persona: {current_persona}", 
                         fontsize=20, fontweight='bold', color='white', y=title_y_pos)

            def draw_heatmap(ax, full_history, current_step_index, title, prompt_text, color_theme):
                 visible_history = full_history[:current_step_index+1]
                 
                 rows, cols = max_steps, top_k
                 matrix = np.zeros((rows, cols))
                 
                 for r, step_data in enumerate(visible_history):
                     candidates = step_data['candidates']
                     for c, (tok, prob) in enumerate(candidates):
                         matrix[r, c] = prob * 100
                 
                 active_cmap = cmap_green if color_theme == "Greens" else cmap_red
                 ax.imshow(matrix, cmap=active_cmap, vmin=0, vmax=100, aspect='auto')
                 
                 ax.set_xticks(np.arange(cols))
                 ax.set_yticks(np.arange(rows))
                 ax.set_xticklabels([f"#{k+1}" for k in range(cols)], 
                                    fontsize=12, fontweight='bold', color='white')
                 
                 step_labels = []
                 for k in range(rows):
                     if k < len(visible_history):
                         chosen = visible_history[k]['chosen_token']
                         step_labels.append(f"Step {k+1}\n\"{utils.clean_token(chosen, 'readable')}\"")
                     else:
                         step_labels.append(f"Step {k+1}")
                 
                 ax.set_yticklabels(step_labels, fontsize=12, fontweight='bold', color='white')
                 
                 wrapper = textwrap.TextWrapper(width=50) 
                 wrapped_prompt = wrapper.fill(text=f"Input: \"{prompt_text}\"")
                 
                 ax.set_title(f"{title}\n{wrapped_prompt}", 
                              fontsize=16, color='white', pad=23.5, linespacing=1.25)
                 
                 for r in range(len(visible_history)):
                     candidates = visible_history[r]['candidates']
                     for c in range(cols):
                         tok, prob = candidates[c]
                         val = prob * 100
                         text_color = "black" if val > 50 else "white"
                         ax.text(c, r, f"{utils.clean_token(tok, 'readable')}\n{val:.0f}%", 
                                 ha="center", va="center", color=text_color, 
                                 fontsize=10, fontweight='bold')
                                 
                 ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
                 ax.grid(False)

            draw_heatmap(axes[0], history_a, step, "SCENARIO A", prompt_a, "Greens")
            draw_heatmap(axes[1], history_b, step, "SCENARIO B", prompt_b, "Reds")
            
            plt.subplots_adjust(top=top_margin, bottom=0.05, wspace=0.3)
            
            fname = temp_dir / f"frame_{step:03d}.png"
            
            plt.savefig(fname, dpi=120) 
            plt.close(fig)
            filenames.append(str(fname))

        print("🎞️🎞️🎞️ Stitching MP4 Together... 🎞️🎞️🎞️")
        fps = 1 / frame_duration
        with imageio.get_writer(output_path, fps=fps, 
                                macro_block_size=None, 
                                ffmpeg_log_level="error") as writer:
            for filename in filenames:
                writer.append_data(imageio.imread(filename))
                
        shutil.rmtree(temp_dir)
        print(f"✅✅✅✅✅ Saved to: {output_path} ✅✅✅✅✅")
        
    finally:
        if local_manager: 
            local_manager.unload_model()