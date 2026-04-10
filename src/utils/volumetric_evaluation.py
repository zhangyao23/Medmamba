import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path


class VolumetricEvaluator:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        output_dir: str = "volumetric_results"
    ):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.code_activation_history = {}
        
    def generate_3d_segmentation(
        self,
        patches: torch.Tensor,
        coords: torch.Tensor,
        labels: torch.Tensor,
        masks: torch.Tensor,
        volume_shape: Tuple[int, int, int]
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        self.model.eval()
        with torch.no_grad():
            logits, attention, vq_loss, codes, sorted_coords, _ = self.model(
                patches, coords, None, masks
            )
        
        B = patches.shape[0]
        results = []
        
        for b in range(B):
            mask = masks[b].bool()
            valid_codes = codes[b, mask]
            valid_coords = coords[b, mask]
            valid_attention = attention[b, mask]
            
            seg_mask = self.model.codebook.codes_to_segmentation_mask(valid_codes)
            
            volume_seg = np.zeros(volume_shape, dtype=np.uint8)
            volume_attention = np.zeros(volume_shape, dtype=np.float32)
            
            for i in range(len(valid_coords)):
                coord = valid_coords[i].cpu().numpy().astype(int)
                volume_seg[coord[0], coord[1], coord[2]] = seg_mask[i].item()
                volume_attention[coord[0], coord[1], coord[2]] = valid_attention[i].item()
            
            results.append({
                'segmentation': volume_seg,
                'attention': volume_attention,
                'codes': valid_codes.cpu().numpy(),
                'coords': valid_coords.cpu().numpy()
            })
        
        if len(results) == 1:
            return results[0]['segmentation'], results[0]['attention'], results[0]
        return results
    
    def calculate_tumor_burden(
        self,
        segmentation: np.ndarray,
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> Dict[str, float]:
        tumor_voxels = (segmentation > 0).sum()
        
        voxel_volume = np.prod(voxel_spacing)
        tumor_volume = tumor_voxels * voxel_volume
        
        total_voxels = segmentation.size
        tumor_ratio = tumor_voxels / total_voxels
        
        return {
            'tumor_voxels': int(tumor_voxels),
            'tumor_volume_mm3': float(tumor_volume),
            'tumor_ratio': float(tumor_ratio),
            'total_voxels': int(total_voxels)
        }
    
    def track_code_activations(
        self,
        codes: torch.Tensor,
        sample_id: str,
        coords: Optional[torch.Tensor] = None
    ):
        codes_np = codes.cpu().numpy().flatten()
        
        for code_id in np.unique(codes_np):
            code_id = int(code_id)
            if code_id not in self.code_activation_history:
                self.code_activation_history[code_id] = []
            
            activation_info = {
                'sample_id': sample_id,
                'count': int((codes_np == code_id).sum())
            }
            
            if coords is not None:
                code_positions = coords[codes == code_id].cpu().numpy()
                activation_info['positions'] = code_positions.tolist()
            
            self.code_activation_history[code_id].append(activation_info)
    
    def get_code_semantic_interpretation(
        self,
        code_id: int,
        top_k: int = 10
    ) -> Dict:
        if code_id not in self.code_activation_history:
            return {
                'code_id': code_id,
                'description': 'No activation history',
                'top_samples': []
            }
        
        history = self.code_activation_history[code_id]
        
        sorted_history = sorted(history, key=lambda x: x['count'], reverse=True)[:top_k]
        
        total_activations = sum(h['count'] for h in history)
        avg_activations = total_activations / len(history)
        
        return {
            'code_id': code_id,
            'total_activations': total_activations,
            'avg_activations_per_sample': avg_activations,
            'num_samples': len(history),
            'top_samples': sorted_history
        }
    
    def save_code_interpretation_report(self, filename: str = "code_interpretation.json"):
        report = {}
        
        for code_id in self.code_activation_history.keys():
            report[f"code_{code_id}"] = self.get_code_semantic_interpretation(code_id)
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Code interpretation report saved to: {output_path}")
        return output_path
    
    def generate_3d_point_cloud(
        self,
        segmentation: np.ndarray,
        attention: np.ndarray,
        threshold: float = 0.5,
        downsample_factor: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        tumor_mask = segmentation > 0
        
        high_attention = attention > threshold
        
        roi_mask = tumor_mask & high_attention
        
        points = np.argwhere(roi_mask)
        
        if downsample_factor > 1:
            points = points[::downsample_factor]
        
        colors = attention[roi_mask][::downsample_factor] if downsample_factor > 1 else attention[roi_mask]
        
        return points, colors
    
    def export_visualization_data(
        self,
        segmentation: np.ndarray,
        attention: np.ndarray,
        sample_id: str,
        format: str = 'npy'
    ):
        sample_dir = self.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        if format == 'npy':
            np.save(sample_dir / 'segmentation.npy', segmentation)
            np.save(sample_dir / 'attention.npy', attention)
        elif format == 'npz':
            np.savez_compressed(
                sample_dir / 'volume_data.npz',
                segmentation=segmentation,
                attention=attention
            )
        
        points, colors = self.generate_3d_point_cloud(segmentation, attention)
        np.save(sample_dir / 'point_cloud_points.npy', points)
        np.save(sample_dir / 'point_cloud_colors.npy', colors)
        
        metadata = {
            'sample_id': sample_id,
            'volume_shape': list(segmentation.shape),
            'tumor_burden': self.calculate_tumor_burden(segmentation),
            'num_points': len(points)
        }
        
        with open(sample_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Visualization data exported to: {sample_dir}")
        return sample_dir
