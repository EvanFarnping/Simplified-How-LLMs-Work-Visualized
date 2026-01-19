from model_manager import ModelManager
from predictor import Predictor
from pathlib import Path
import utils

import matplotlib.pyplot as plt
import numpy as np

# ============================================================================== #
#                DO NOT TOUCH UNLESS YOU KNOW WHAT YOU ARE DOING!                #
# ============================================================================== #

def run_prediction_chart(
    project_root,
    model_name="Phi-3-mini", 
    use_chat_template=True, 
    current_persona="default",
    prompt_text="What color is a stop sign?", 
    top_k=10, 
    output_filename="prediction_chart.png",
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
        final_prompt = ""
        if use_chat_template:
            print(f"💬💬💬 Chat Mode: {current_persona} 💬💬💬")
            persona_text = utils.load_persona(persona_file, current_persona)
            final_prompt = utils.format_as_chat_robust(tokenizer, prompt_text, persona_text)
        else:
            print("📝📝📝 Completion Mode 📝📝📝")
            final_prompt = prompt_text

        print(f"🔮🔮🔮 Predicting... 🔮🔮🔮")
        tokens, probs = Predictor.get_top_k_tokens(model, tokenizer, final_prompt, k=top_k)
        labels = [f"'{utils.clean_token(t, 'visualize')}'" for t in tokens]
        
        plt.style.use('dark_background')
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Segoe UI Emoji',   
                                               'Arial Unicode MS', 'DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

        fig, ax = plt.subplots(figsize=(14, 8))
        y_pos = np.arange(len(labels))
        
        bar_color = '#00e676'
        bars = ax.barh(y_pos, probs * 100, align='center', color=bar_color)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=15, fontweight='bold', color='white')
        ax.invert_yaxis()
        
        ax.set_xlabel('Probability (%)', fontsize=16, fontweight='bold', labelpad=15)
        
        ax.set_title(f"Immediate Next Token Predictions\nModel: {model_name} | Persona: {current_persona}", 
                     fontsize=24, fontweight='bold', color='white', pad=45)

        display_prompt = (prompt_text[:90] + '...') if len(prompt_text) > 90 else prompt_text
        ax.text(0.5, 1.02, f"Input: \"{display_prompt}\"", transform=ax.transAxes, 
                ha='center', va='bottom', fontsize=20, color="#dadada")

        for bar in bars:
            width = bar.get_width()
            
            if width > 10:
                x_pos = width - 0.5
                ha = 'right'
                text_color = 'black'
            else:
                x_pos = width + 0.5
                ha = 'left'
                text_color = 'white'
            
            if width > 0.1:
                ax.text(x_pos, bar.get_y() + bar.get_height()/2, 
                        f'{width:.1f}%', 
                        ha=ha, va='center', color=text_color, 
                        fontsize=14, fontweight='bold')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)

        ax.tick_params(axis='x', labelsize=12, width=2)
        ax.tick_params(axis='y', width=2)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)

        print(f"✅✅✅✅✅ Saved to: {output_path} ✅✅✅✅✅")

    finally:
        if local_manager: 
            local_manager.unload_model()