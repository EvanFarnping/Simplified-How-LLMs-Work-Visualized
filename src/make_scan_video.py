from model_manager import ModelManager
from analyzer import BrainAnalyzer
from pathlib import Path
import utils

import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np
import shutil
import math
import io 

# ============================================================================== #
#                DO NOT TOUCH UNLESS YOU KNOW WHAT YOU ARE DOING!                #
# ============================================================================== #

def render_video(frames, output_path, fps):
    print(f"   🎞️ Stitching {output_path.name} from memory...")
    
    imageio.mimsave(output_path, frames, fps=fps, macro_block_size=None)
    print(f"   ✅ Saved: {output_path}")

def run_scan_video(
    project_root,
    model_name="Phi-3-mini",
    prompt="The quick brown fox jumps over the lazy doggo.",
    output_filename="brain_scan.mp4",
    frame_duration=2.0, 
    model = None,
    tokenizer = None
    ):

    project_root = Path(project_root)
    
    base_output_path = Path(output_filename)
    
    if not base_output_path.is_absolute():
        base_output_path = project_root / base_output_path

    out_grid = base_output_path.parent / f"{base_output_path.stem}_grid.mp4"
    out_avg = base_output_path.parent / f"{base_output_path.stem}_average.mp4"
    
    temp_grid_dir = project_root / "temp_grid_frames"
    temp_avg_dir = project_root / "temp_avg_frames"

    local_manager = None

    if model is None or tokenizer is None:
        print(f"⚙️⚙️⚙️ Loading {model_name}... ⚙️⚙️⚙️")
        local_manager = ModelManager()
        model, tokenizer = local_manager.load_model(model_name)
    
    try:
        print(f"🧠🧠🧠 Scanning text... 🧠🧠🧠")
        device = next(model.parameters()).device 
        
        attn_layers_list, token_ids = BrainAnalyzer.get_attention_map(
            model, tokenizer, prompt, device)
        
        raw_tokens = tokenizer.convert_ids_to_tokens(token_ids)
        tokens = [utils.clean_token(t, 'raw') for t in raw_tokens]
        
        num_layers = len(attn_layers_list)
        first_layer_shape = attn_layers_list[0].shape
        num_heads = first_layer_shape[1] if len(first_layer_shape) == 4 else first_layer_shape[0]
        
        if temp_grid_dir.exists(): 
            shutil.rmtree(temp_grid_dir)
        if temp_avg_dir.exists(): 
            shutil.rmtree(temp_avg_dir)

        # Initialize lists to hold images in RAM
        grid_frames = []
        avg_frames = []
        
        plt.style.use('dark_background')
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Segoe UI Emoji',   
                                           'Arial Unicode MS', 'DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Calculate Grid Dimensions for the Detailed View
        cols = 8
        rows = math.ceil(num_heads / cols)

        print(f"🎨🎨🎨 Rendering {num_layers} Layers... 🎨🎨🎨")

        for i in range(num_layers):
            print(f"   -> Layer {i+1}...")
            
            layer_tensor = attn_layers_list[i]
            
            if len(layer_tensor.shape) == 4:
                layer_matrix = layer_tensor[0].float().cpu().numpy()
            else:
                layer_matrix = layer_tensor.float().cpu().numpy()

            #RENDER GRID VIEW
            fig_grid, axes = plt.subplots(rows, cols, figsize=(24, 12))
            axes = axes.flatten()
            
            for h in range(len(axes)):
                if h < num_heads:
                    ax = axes[h]
                    layer_data = np.nan_to_num(layer_matrix[h], nan=0.0, posinf=1.0, neginf=0.0)
                    ax.imshow(layer_data, cmap='viridis', aspect='auto')
                    ax.set_title(f"Head {h}", fontsize=10, color='white', pad=2)
                    ax.axis('off')
                else:
                    axes[h].axis('off')

            plt.suptitle(f"Layer {i+1} / {num_layers} | Model: {model_name}\n(Attention Head Grid)", 
                         fontsize=20, fontweight='bold', color='white', y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            grid_frames.append(imageio.imread(buf))
            buf.close()
            plt.close(fig_grid)

            #RENDER AVERAGED VIEW
            avg_data = np.mean(layer_matrix, axis=0) 
            avg_data = np.nan_to_num(avg_data, nan=0.0, posinf=1.0, neginf=0.0)

            fig_avg, ax_avg = plt.subplots(figsize=(10, 8))
            cax = ax_avg.imshow(avg_data, cmap='viridis')
            
            ax_avg.set_xticks(np.arange(len(tokens)))
            ax_avg.set_yticks(np.arange(len(tokens)))
            ax_avg.set_xticklabels(tokens, rotation=45, ha="right")
            ax_avg.set_yticklabels(tokens)
            ax_avg.set_title(f"Model: {model_name}\nLayer {i+1} / {num_layers} (Averaged)", fontsize=16)
            ax_avg.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
            
            bar = fig_avg.colorbar(cax, ax=ax_avg)
            bar.ax.set_title("Activity Level", fontsize=8)

            plt.tight_layout()
            
            buf_avg = io.BytesIO()
            plt.savefig(buf_avg, format='png', dpi=100)
            buf_avg.seek(0)
            avg_frames.append(imageio.imread(buf_avg))
            buf_avg.close()
            plt.close(fig_avg)

        fps = 1 / frame_duration
        render_video(grid_frames, out_grid, fps)
        render_video(avg_frames, out_avg, fps)
            
        print("✅✅✅✅✅ Done! Created 2 videos in the export folder. ✅✅✅✅✅")

    finally:
        if local_manager: 
            local_manager.unload_model()