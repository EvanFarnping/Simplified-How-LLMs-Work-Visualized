from matplotlib.colors import LinearSegmentedColormap
from model_manager import ModelManager
from predictor import Predictor
from pathlib import Path
import utils

import matplotlib.pyplot as plt
import numpy as np

# ============================================================================== #
#                DO NOT TOUCH UNLESS YOU KNOW WHAT YOU ARE DOING!                #
# ============================================================================== #

def run_sequence_chart(
    project_root,
    model_name="Phi-3-mini",
    use_chat_template=True,
    current_persona="history-teacher",
    prompt_text="The Axis powers consist of?",
    prediction_steps=6,
    top_k=5,
    temperature=0.7,
    output_filename="sequence_prediction.png",
    model = None,
    tokenizer = None
    ):

    project_root = Path(project_root)
    config_dir = project_root / "main_configs"
    persona_file = config_dir / "personas.yaml"
    output_path = project_root / output_filename

    local_manager = None

    if model is None or tokenizer is None:
        print(f"⚙️⚙️⚙️ Loading {model_name}... ⚙️⚙️⚙️")
        local_manager = ModelManager()
        model, tokenizer = local_manager.load_model(model_name)
    
    try:
        if use_chat_template:
            print(f"💬💬💬 Chat Mode Active 💬💬💬")
            persona_text = utils.load_persona(persona_file, current_persona)
            final_prompt = utils.format_as_chat_robust(tokenizer, prompt_text, persona_text)
        else:
            final_prompt = prompt_text

        print(f"🔮🔮🔮 Generating sequence... 🔮🔮🔮")
        sequence_data = Predictor.predict_sequence(model, 
                                                   tokenizer, 
                                                   final_prompt, 
                                                   steps=prediction_steps, 
                                                   k=top_k, 
                                                   temperature=temperature)
        
        actual_steps = len(sequence_data)
        
        if actual_steps == 0:
            print("⚠️ No tokens generated. Skipping chart. ⚠️")
            return

        prediction_steps = actual_steps
        prob_grid = np.zeros((prediction_steps, top_k))
        text_grid = []
        step_labels = []
        
        for i, step in enumerate(sequence_data):
            winner = step["chosen_token"]
            step_labels.append(f"Step {i+1}\nSelected:\n'{utils.clean_token(winner, 'visualize')}'")
            row_text = []

            for j, (tok, prob) in enumerate(step["candidates"]):
                prob_grid[i, j] = prob * 100 
                row_text.append(f"'{utils.clean_token(tok, 'visualize')}'\n{prob*100:.1f}%")
            text_grid.append(row_text)

        plt.style.use('dark_background')
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Segoe UI Emoji',   
                                               'Arial Unicode MS', 'DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        chart_height = max(6.0, prediction_steps * 1.8)
        fig, ax = plt.subplots(figsize=(16, chart_height))
        
        cmap_green = LinearSegmentedColormap.from_list("custom_green", 
                                                       ["#000000", "#00e676"])
        active_cmap = cmap_green

        cax = ax.imshow(prob_grid, cmap=active_cmap, vmin=0, vmax=100, aspect='auto')
        
        ax.set_xticks(np.arange(top_k))
        ax.set_yticks(np.arange(prediction_steps))
        ax.set_xticklabels([f"#{k+1}" for k in range(top_k)], fontweight='bold', fontsize=14, color='white')
        ax.set_yticklabels(step_labels, fontweight='bold', fontsize=14, color='white')
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, pad=10)
        
        for i in range(prediction_steps):
            for j in range(top_k):
                prob = prob_grid[i, j]
                text_color = "black" if prob > 50 else "white"
                weight = 'bold' if j == 0 else 'normal'
                
                ax.text(j, i, text_grid[i][j], 
                        ha="center", va="center", color=text_color, 
                        fontweight=weight, fontsize=12)

        title_y_pos = 1.0 - (0.5 / chart_height)
        plt.suptitle(f"Sequence Generation | Model: {model_name} | Persona: {current_persona}", 
                     fontsize=24, fontweight='bold', color='white', y=title_y_pos)
        
        prompt_y_pos = 1.0 - (1.2 / chart_height)
        display_prompt = (prompt_text[:90] + '...') if len(prompt_text) > 90 else prompt_text
        fig.text(0.5, prompt_y_pos, f"Input: \"{display_prompt}\"", 
                 ha='center', va='top', fontsize=18, color='#dadada')

        top_margin = 1.0 - (2.2 / chart_height)
        if top_margin < 0.6: 
            top_margin = 0.6
            
        plt.subplots_adjust(top=top_margin)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

        print(f"✅✅✅✅✅ Saved to: {output_path} ✅✅✅✅✅")

    finally:
        if local_manager: 
            local_manager.unload_model()