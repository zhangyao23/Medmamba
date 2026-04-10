import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Optional

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def render_3d_heatmap(
    volume: np.ndarray,
    attention_weights: np.ndarray,
    threshold: float = 0.5,
    output_path: str = 'heatmap_3d.html'
):
    if not PLOTLY_AVAILABLE:
        print("plotly not available, skipping 3D rendering")
        return
    
    H, W, D = volume.shape
    Z, Y, X = np.meshgrid(
        np.arange(D), np.arange(H), np.arange(W), indexing='ij'
    )
    
    mask = attention_weights > threshold
    Z_filtered = Z.flatten()[mask.flatten()]
    Y_filtered = Y.flatten()[mask.flatten()]
    X_filtered = X.flatten()[mask.flatten()]
    weights_filtered = np.repeat(attention_weights, H * W)[mask.flatten()]
    
    fig = go.Figure(data=[go.Scatter3d(
        x=X_filtered,
        y=Y_filtered,
        z=Z_filtered,
        mode='markers',
        marker=dict(
            size=3,
            color=weights_filtered,
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Attention")
        )
    )])
    
    fig.update_layout(
        title="3D Tumor Heatmap",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z (Slice)"
        )
    )
    
    fig.write_html(output_path)
    print(f"3D heatmap saved to {output_path}")


def visualize_slice_sequence(
    slices: np.ndarray,
    attention: np.ndarray,
    output_dir: str
):
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (slice_img, att) in enumerate(zip(slices, attention)):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(slice_img, cmap='gray')
        ax.set_title(f'Slice {i}, Attention: {att:.3f}')
        ax.axis('off')
        plt.savefig(f'{output_dir}/slice_{i:03d}.png', bbox_inches='tight')
        plt.close()
    
    print(f"Slice sequence saved to {output_dir}")


def visualize_codebook_3d(
    model,
    dataloader,
    num_codes_to_show: int = 10,
    output_dir: str = 'codebook_viz'
):
    os.makedirs(output_dir, exist_ok=True)
    
    code_slices = defaultdict(list)
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            slices = batch['slices']
            features = model.backbone(slices)
            _, codes, _ = model.codebook(features)
            
            for i in range(codes.shape[0]):
                for j in range(codes.shape[1]):
                    code_id = codes[i, j].item()
                    slice_img = slices[i, j, 0].cpu().numpy()
                    code_slices[code_id].append(slice_img)
    
    for code_id in sorted(code_slices.keys())[:num_codes_to_show]:
        samples = code_slices[code_id][:20]
        
        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        fig.suptitle(f'Code #{code_id} (Count: {len(code_slices[code_id])})')
        
        for ax, img in zip(axes.flat, samples):
            ax.imshow(img, cmap='gray')
            ax.axis('off')
        
        plt.savefig(f'{output_dir}/code_{code_id}.png')
        plt.close()
    
    print(f"Codebook visualization saved to {output_dir}")
