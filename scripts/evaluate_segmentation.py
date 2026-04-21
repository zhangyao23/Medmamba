import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from src.training import Config
from src.models.feature_extractor_3d import VolumetricFeatureExtractor
from src.models.vector_quantizer_3d import PartitionedVectorQuantizer
from src.models.spatial_scanner_3d import (
    ZOrderSpatialScanner,
    reorder_sequence,
    restore_sequence_order,
)
from src.models.hilbert_scanner import HilbertCurveSpatialScanner
from src.models.video_mamba import VideoMamba3D
from src.models.mil_head import AttentionMILHead
from src.models.self_correction import SelfCorrectionModule
from src.models.decoder_3d import VolumetricDecoder3D
from src.data.volumetric_dataset import compute_axis_starts


class Volumetric3DMIL(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        feature_dim = config.model['codebook']['embedding_dim']

        self.feature_extractor = VolumetricFeatureExtractor(
            arch=config.model['feature_extractor']['arch'],
            spatial_dims=config.model['feature_extractor']['spatial_dims'],
            n_input_channels=config.model['feature_extractor']['n_input_channels'],
            pretrained=config.model['feature_extractor']['pretrained'],
            frozen=config.model['feature_extractor']['frozen']
        )
        self.codebook = PartitionedVectorQuantizer(
            num_embeddings=config.model['codebook']['num_embeddings'],
            embedding_dim=config.model['codebook']['embedding_dim'],
            healthy_ratio=config.model['codebook']['healthy_ratio'],
            commitment_cost=config.model['codebook']['commitment_cost'],
            use_ema=config.model['codebook']['use_ema'],
            ema_decay=config.model['codebook']['ema_decay'],
            usage_balance_alpha=config.model['codebook'].get('usage_balance_alpha', 0.0),
            explore_prob=config.model['codebook'].get('explore_prob', 0.0)
        )
        scanner_type = config.model.get('spatial_scanner', {}).get('type', 'z_order')
        if scanner_type == 'hilbert':
            max_order = config.model['spatial_scanner'].get('hilbert_max_order', 8)
            self.spatial_scanner = HilbertCurveSpatialScanner(max_order=max_order)
        else:
            self.spatial_scanner = ZOrderSpatialScanner()

        self_correction_config = config.model.get('self_correction', {})
        self.use_self_correction = self_correction_config.get('enabled', False)
        if self.use_self_correction:
            self.self_correction = SelfCorrectionModule(
                entropy_threshold=self_correction_config.get('entropy_threshold', 0.7),
                neighbor_radius=self_correction_config.get('neighbor_radius', 2),
                spatial_weight=self_correction_config.get('spatial_weight', 0.3)
            )
        self.mamba = VideoMamba3D(
            d_model=feature_dim,
            d_state=config.model['mamba']['d_state'],
            d_conv=config.model['mamba']['d_conv'],
            expand=config.model['mamba']['expand'],
            num_layers=config.model['mamba']['num_layers'],
            bidirectional=config.model['mamba']['bidirectional']
        )
        self.mil_head = AttentionMILHead(
            input_dim=feature_dim,
            hidden_dim=config.model['mil']['hidden_dim'],
            num_classes=config.model['mil']['num_classes']
        )
        decoder_config = config.model.get('decoder', {})
        self.decoder_enabled = decoder_config.get('enabled', False)
        if self.decoder_enabled:
            self.decoder = VolumetricDecoder3D(
                input_dim=feature_dim,
                patch_size=tuple(config.data['patch_size']),
                output_channels=config.model['feature_extractor']['n_input_channels'],
                hidden_dims=decoder_config.get('hidden_dims', [512, 1024])
            )

    def forward(self, patches, coords, labels=None, masks=None, mini_batch_size=8):
        features = self.feature_extractor(patches, mini_batch_size=mini_batch_size)
        quantized, codes, vq_loss = self.codebook(features, labels, mask=masks)
        sorted_features, sorted_coords, sort_perm = self.spatial_scanner(quantized, coords)
        sorted_masks = reorder_sequence(masks, sort_perm)
        context = self.mamba(sorted_features, mask=sorted_masks)
        context_orig = restore_sequence_order(context, sort_perm)
        logits, attention = self.mil_head(context_orig, mask=masks)
        return logits, attention, codes, quantized


def load_volume(path, min_hu=-1024.0, max_hu=3071.0, adaptive_norm=True):
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    volume = img.get_fdata()
    if volume.ndim == 4:
        volume = volume[:, :, :, 0] if volume.shape[-1] == 1 else volume.mean(axis=-1)
    volume_raw = volume.copy()
    from src.data.volumetric_dataset import _normalize_volume
    volume = _normalize_volume(volume, path, min_hu, max_hu, adaptive_norm)
    return volume, volume_raw


def get_gt_mask_path(image_path):
    base = os.path.basename(image_path)
    name_noext = base.split('.nii')[0]
    label_dir = os.path.join(os.path.dirname(os.path.dirname(image_path)), 'labels')
    gt_path = os.path.join(label_dir, name_noext + '_seg.nii.gz')
    if os.path.exists(gt_path):
        return gt_path
    gt_path2 = os.path.join(label_dir, name_noext + '.nii.gz')
    if os.path.exists(gt_path2):
        return gt_path2
    return None


TUMOR_LABEL_MAP = {
    'Bladder_Tumor_00': 1,
    'Breast_Tumor_00': 1,
    'Cervix_Tumor_00': 2,
    'Colon_Tumor_00': 14,
    'Kidney_Tumor_00': 3,
    'Liver_Tumor_00': 14,
    'Lung_Tumor_00': 1,
    'Lung_Tumor_01': 5,
    'Pancreas_Tumor_00': 14,
    'Prostate_Tumor_00': 3,
    'Uterus_Tumor_00': 3,
}


def _get_tumor_label(path: str):
    for organ, label in TUMOR_LABEL_MAP.items():
        if organ in path:
            return label
    return None


def load_gt_mask(gt_path):
    img = nib.load(gt_path)
    img = nib.as_closest_canonical(img)
    mask = img.get_fdata()
    if mask.ndim == 4:
        mask = mask[:, :, :, 0]
    tumor_label = _get_tumor_label(gt_path)
    if tumor_label is not None:
        return (mask == tumor_label).astype(np.float32)
    return (mask > 0).astype(np.float32)


def compute_dice_iou(pred, gt, smooth=1e-6):
    pred_bin = (pred > 0.5).astype(np.float32)
    gt_bin = (gt > 0.5).astype(np.float32)
    intersection = (pred_bin * gt_bin).sum()
    dice = (2.0 * intersection + smooth) / (pred_bin.sum() + gt_bin.sum() + smooth)
    union = pred_bin.sum() + gt_bin.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    precision = (intersection + smooth) / (pred_bin.sum() + smooth)
    recall = (intersection + smooth) / (gt_bin.sum() + smooth)
    return {
        'dice': float(dice),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'pred_volume': float(pred_bin.sum()),
        'gt_volume': float(gt_bin.sum()),
        'intersection': float(intersection),
    }


def compute_patch_level_metrics(coords, seg_mask_np, attention_np, gt_mask_hwz,
                                patch_size):
    ph, pw, pd = patch_size[1], patch_size[2], patch_size[0]
    H, W, D = gt_mask_hwz.shape
    n = len(coords)
    gt_patch_labels = np.zeros(n, dtype=np.float32)
    gt_patch_overlap = np.zeros(n, dtype=np.float32)

    for i, (z, y, x) in enumerate(coords):
        z_end = min(z + pd, D)
        y_end = min(y + ph, H)
        x_end = min(x + pw, W)
        patch_gt = gt_mask_hwz[y:y_end, x:x_end, z:z_end]
        tumor_voxels = patch_gt.sum()
        gt_patch_labels[i] = 1.0 if tumor_voxels > 0 else 0.0
        gt_patch_overlap[i] = tumor_voxels / max(patch_gt.size, 1)

    pred_pos = seg_mask_np.astype(bool)
    gt_pos = gt_patch_labels.astype(bool)

    tp = int((pred_pos & gt_pos).sum())
    fp = int((pred_pos & ~gt_pos).sum())
    fn = int((~pred_pos & gt_pos).sum())
    tn = int((~pred_pos & ~gt_pos).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    att_on_gt_pos = float(attention_np[gt_pos].sum()) if gt_pos.any() else 0.0
    att_total = float(attention_np.sum())

    return {
        'patch_tp': tp, 'patch_fp': fp, 'patch_fn': fn, 'patch_tn': tn,
        'patch_precision': round(float(precision), 4),
        'patch_recall': round(float(recall), 4),
        'patch_f1': round(float(f1), 4),
        'gt_positive_patches': int(gt_pos.sum()),
        'has_gt_positive_patch': bool(gt_pos.any()),
        'pred_positive_patches': int(pred_pos.sum()),
        'total_patches': n,
        'attention_on_gt_tumor': round(att_on_gt_pos, 4),
        'attention_total': round(att_total, 4),
        'attention_gt_ratio': round(att_on_gt_pos / max(att_total, 1e-8), 4),
    }


def extract_patches(volume, patch_size=(32, 32, 32), stride=(32, 32, 32), max_patches=64):
    H, W, D = volume.shape
    patch_d, patch_h, patch_w = patch_size
    stride_d, stride_h, stride_w = stride
    patches = []
    coords = []
    z_starts = compute_axis_starts(D, patch_d, stride_d)
    y_starts = compute_axis_starts(H, patch_h, stride_h)
    x_starts = compute_axis_starts(W, patch_w, stride_w)
    for z in z_starts:
        z_end = min(z + patch_d, D)
        z_start = max(0, z_end - patch_d)
        for y in y_starts:
            y_end = min(y + patch_h, H)
            y_start = max(0, y_end - patch_h)
            for x in x_starts:
                x_end = min(x + patch_w, W)
                x_start = max(0, x_end - patch_w)
                patch = volume[y_start:y_end, x_start:x_end, z_start:z_end]
                if patch.shape != (patch_h, patch_w, patch_d):
                    patch_padded = np.zeros((patch_h, patch_w, patch_d), dtype=np.float32)
                    patch_padded[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
                    patch = patch_padded
                patch = np.transpose(patch, (2, 0, 1))
                patches.append(patch)
                coords.append((z_start, y_start, x_start))
    if len(patches) > max_patches:
        patches = patches[:max_patches]
        coords = coords[:max_patches]
    return patches, coords


def build_3d_maps(coords, seg_mask, attention, patch_size, volume_shape_dhw):
    D, H, W = volume_shape_dhw
    pd, ph, pw = patch_size
    seg_volume = np.zeros((D, H, W), dtype=np.float32)
    att_volume = np.zeros((D, H, W), dtype=np.float32)
    count_volume = np.zeros((D, H, W), dtype=np.float32)

    for i, (z, y, x) in enumerate(coords):
        z_end = min(z + pd, D)
        y_end = min(y + ph, H)
        x_end = min(x + pw, W)
        seg_volume[z:z_end, y:y_end, x:x_end] += float(seg_mask[i])
        att_volume[z:z_end, y:y_end, x:x_end] += float(attention[i])
        count_volume[z:z_end, y:y_end, x:x_end] += 1.0

    valid = count_volume > 0
    seg_volume[valid] /= count_volume[valid]
    att_volume[valid] /= count_volume[valid]
    return seg_volume, att_volume


def largest_connected_component_3d(binary_mask):
    from scipy import ndimage
    labeled, num_features = ndimage.label(binary_mask)
    if num_features <= 1:
        return binary_mask
    sizes = ndimage.sum(binary_mask, labeled, range(1, num_features + 1))
    largest_label = np.argmax(sizes) + 1
    return (labeled == largest_label).astype(binary_mask.dtype)


def visualize_slices(volume_norm, seg_volume, att_volume, save_path, sample_name,
                     label, pred_class, pred_prob, gt_volume=None,
                     dice_info=None, n_slices=8):
    D, H, W = volume_norm.shape
    slice_indices = np.linspace(0, D - 1, n_slices, dtype=int)

    n_rows = 4 if gt_volume is not None else 3
    fig, axes = plt.subplots(n_rows, n_slices, figsize=(3 * n_slices, 3.3 * n_rows))

    title = f'{sample_name}\nGT={label}, Pred={pred_class} (prob={pred_prob:.3f})'
    if dice_info:
        title += f'  |  Dice={dice_info["dice"]:.3f}, IoU={dice_info["iou"]:.3f}'
    fig.suptitle(title, fontsize=13, fontweight='bold')

    for col, z_idx in enumerate(slice_indices):
        axes[0, col].imshow(volume_norm[z_idx], cmap='gray', vmin=0, vmax=1)
        axes[0, col].set_title(f'z={z_idx}', fontsize=9)
        axes[0, col].axis('off')

        axes[1, col].imshow(volume_norm[z_idx], cmap='gray', vmin=0, vmax=1)
        seg_slice = seg_volume[z_idx]
        seg_overlay = np.ma.masked_where(seg_slice < 0.5, seg_slice)
        axes[1, col].imshow(seg_overlay, cmap='Reds', alpha=0.6, vmin=0, vmax=1)
        axes[1, col].set_title(f'Seg z={z_idx}', fontsize=9)
        axes[1, col].axis('off')

        axes[2, col].imshow(volume_norm[z_idx], cmap='gray', vmin=0, vmax=1)
        att_max = att_volume.max() if att_volume.max() > 0 else 1.0
        axes[2, col].imshow(att_volume[z_idx], cmap='jet', alpha=0.5, vmin=0, vmax=att_max)
        axes[2, col].set_title(f'Attn z={z_idx}', fontsize=9)
        axes[2, col].axis('off')

        if gt_volume is not None:
            axes[3, col].imshow(volume_norm[z_idx], cmap='gray', vmin=0, vmax=1)
            gt_slice = gt_volume[z_idx]
            gt_overlay = np.ma.masked_where(gt_slice < 0.5, gt_slice)
            axes[3, col].imshow(gt_overlay, cmap='Greens', alpha=0.6, vmin=0, vmax=1)
            pred_overlay = np.ma.masked_where(seg_volume[z_idx] < 0.5, seg_volume[z_idx])
            axes[3, col].imshow(pred_overlay, cmap='Reds', alpha=0.3, vmin=0, vmax=1)
            axes[3, col].set_title(f'GT+Pred z={z_idx}', fontsize=9)
            axes[3, col].axis('off')

    axes[0, 0].set_ylabel('Original', fontsize=11)
    axes[1, 0].set_ylabel('Codebook Seg', fontsize=11)
    axes[2, 0].set_ylabel('Attention', fontsize=11)
    if gt_volume is not None:
        axes[3, 0].set_ylabel('GT(green)\n+Pred(red)', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def visualize_codebook_stats(code_counts, healthy_mask, save_path):
    n_codes = len(code_counts)
    colors = ['tab:blue' if healthy_mask[i] else 'tab:red' for i in range(n_codes)]

    fig, ax = plt.subplots(figsize=(max(8, n_codes * 0.4), 5))
    ax.bar(range(n_codes), code_counts, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Code Index')
    ax.set_ylabel('Activation Count')
    ax.set_title('Codebook Usage (Blue=Healthy, Red=Cancer)')
    ax.set_xticks(range(n_codes))
    ax.set_xticklabels(range(n_codes), fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/volumetric_config.yaml')
    parser.add_argument('--checkpoint', type=str, default='volumetric_checkpoints/best_model.pth')
    parser.add_argument('--output_dir', type=str, default='segmentation_results')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--max_samples', type=int, default=20)
    parser.add_argument('--max_patches', type=int, default=256)
    parser.add_argument('--eval_stride', type=int, default=None,
                        help='Override stride for denser coverage (e.g. 16)')
    parser.add_argument('--only_positive', action='store_true')
    parser.add_argument('--only_negative', action='store_true')
    parser.add_argument('--balanced', action='store_true',
                        help='Sample equal positive and negative')
    parser.add_argument('--repartition', type=str, default=None,
                        choices=['enrichment', 'topk'],
                        help='Re-partition codebook at eval time')
    parser.add_argument('--repartition_threshold', type=float, default=0.7)
    parser.add_argument('--min_cancer_codes', type=int, default=None)
    parser.add_argument('--soft_seg', action='store_true',
                        help='Use soft cancer scores instead of binary partition')
    parser.add_argument('--soft_seg_threshold', type=float, default=0.3,
                        help='Threshold for soft segmentation binarization')
    parser.add_argument('--attention_combine', action='store_true',
                        help='Combine code seg with attention for segmentation')
    parser.add_argument('--attention_combine_weight', type=float, default=0.5,
                        help='Weight for attention in combined segmentation')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config = Config.from_yaml(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading model...")
    model = Volumetric3DMIL(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if missing:
        print(f"  Missing keys (using defaults): {missing}")
    model.codebook.phase1_complete = True
    if model.codebook.healthy_code_mask.any():
        model.codebook.use_dynamic_partition = True
        model.codebook.healthy_frozen = True
        model.codebook.frozen_code_mask = model.codebook.healthy_code_mask.clone()
    model.eval()
    ckpt_epoch = ckpt.get('epoch', -1)
    ckpt_metrics = ckpt.get('metrics', {})
    print(f"Loaded checkpoint: epoch={ckpt_epoch}, AUC={ckpt_metrics.get('auc', 'N/A')}")
    print(f"Dynamic partition: {model.codebook.use_dynamic_partition}")

    if args.repartition is not None:
        print(f"\n--- Re-partitioning codebook (strategy={args.repartition}) ---")
        model.codebook.repartition(
            threshold=args.repartition_threshold,
            min_cancer_codes=args.min_cancer_codes,
            strategy=args.repartition
        )
        model.codebook.use_dynamic_partition = True
        model.codebook.frozen_code_mask = model.codebook.healthy_code_mask.clone()
        print("--- Re-partition complete ---\n")

    if args.soft_seg:
        cancer_scores = model.codebook.get_cancer_scores()
        print(f"Soft segmentation enabled (threshold={args.soft_seg_threshold})")
        top5 = torch.topk(cancer_scores, k=min(5, len(cancer_scores)), largest=True)
        print(f"  Top cancer score codes: "
              + ", ".join([f"{i.item()}={s:.3f}" for i, s in zip(top5.indices, top5.values)]))

    healthy_mask = model.codebook.healthy_code_mask.cpu().numpy()
    num_embeddings = model.codebook.num_embeddings
    print(f"Codebook: {num_embeddings} codes, "
          f"healthy={healthy_mask.sum()}, cancer={num_embeddings - healthy_mask.sum()}")

    with open(config.data['test_json'], 'r') as f:
        entries = json.load(f)
    if args.only_positive:
        entries = [e for e in entries if e['label'] == 1]
    elif args.only_negative:
        entries = [e for e in entries if e['label'] == 0]
    if args.balanced:
        pos = [e for e in entries if e['label'] == 1]
        neg = [e for e in entries if e['label'] == 0]
        half = args.max_samples // 2
        entries = pos[:half] + neg[:half]
    else:
        entries = entries[:args.max_samples]
    print(f"Processing {len(entries)} samples "
          f"(pos={sum(1 for e in entries if e['label']==1)}, "
          f"neg={sum(1 for e in entries if e['label']==0)})")

    patch_size = tuple(config.data['patch_size'])
    if args.eval_stride is not None:
        stride = tuple([args.eval_stride] * 3)
    else:
        stride = tuple(config.data['stride'])
    max_patches = args.max_patches
    min_hu = config.data['min_hu']
    max_hu = config.data['max_hu']
    adaptive_norm = config.data.get('adaptive_norm', True)
    mini_batch_size = config.model['feature_extractor'].get('mini_batch_size', 8)

    global_code_counts = np.zeros(num_embeddings, dtype=np.int64)
    cancer_code_counts = np.zeros(num_embeddings, dtype=np.int64)
    results = []

    for idx, entry in enumerate(entries):
        volume_path = entry['image']
        label = entry['label']
        sample_name = os.path.basename(os.path.dirname(os.path.dirname(volume_path))) + \
                      '/' + os.path.splitext(os.path.basename(volume_path))[0]
        print(f"\n[{idx+1}/{len(entries)}] {sample_name} (label={label})")

        volume, volume_raw = load_volume(volume_path, min_hu, max_hu, adaptive_norm=adaptive_norm)
        H, W, D = volume.shape
        patches_list, coords_list = extract_patches(volume, patch_size, stride, max_patches)
        n_patches = len(patches_list)
        print(f"  Volume: HWD={H}x{W}x{D}, Patches: {n_patches}")

        patches_tensor = torch.stack(
            [torch.from_numpy(p).float() for p in patches_list]
        ).unsqueeze(0).unsqueeze(2).to(device)
        coords_tensor = torch.tensor(coords_list, dtype=torch.long).unsqueeze(0).to(device)
        masks_tensor = torch.ones(1, n_patches, dtype=torch.bool, device=device)

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
            logits, attention, codes, quantized = model(
                patches_tensor, coords_tensor, None, masks_tensor,
                mini_batch_size=mini_batch_size
            )

        probs = torch.softmax(logits, dim=1)
        pred_class = probs[0].argmax().item()
        pred_prob = probs[0, 1].item()

        codes_np = codes[0, :n_patches].cpu().numpy()
        attention_np = attention[0, :n_patches].cpu().numpy()

        if args.soft_seg:
            soft_scores = model.codebook.codes_to_segmentation_mask(
                codes[0:1, :n_patches], soft=True
            )
            soft_np = soft_scores[0].cpu().numpy()
            seg_mask_np = (soft_np >= args.soft_seg_threshold).astype(np.float32)
        elif args.attention_combine:
            binary_seg = model.codebook.codes_to_segmentation_mask(codes[0:1, :n_patches])
            binary_np = binary_seg[0].cpu().float().numpy()
            att_norm = attention_np.copy()
            att_max = att_norm.max()
            if att_max > 0:
                att_norm = att_norm / att_max
            w = args.attention_combine_weight
            combined = (1.0 - w) * binary_np + w * att_norm
            seg_mask_np = (combined >= 0.5).astype(np.float32)
        else:
            seg_mask = model.codebook.codes_to_segmentation_mask(codes[0:1, :n_patches])
            seg_mask_np = seg_mask[0].cpu().numpy().astype(np.float32)

        for c in codes_np:
            global_code_counts[c] += 1
        if label == 1:
            for c in codes_np:
                cancer_code_counts[c] += 1

        cancer_patches = seg_mask_np.sum()
        cancer_attention = (attention_np * seg_mask_np).sum()
        print(f"  Pred: class={pred_class}, prob(cancer)={pred_prob:.4f}")
        print(f"  Cancer patches: {int(cancer_patches)}/{n_patches} "
              f"({100*cancer_patches/n_patches:.1f}%)")
        print(f"  Attention on cancer patches: {cancer_attention:.4f} / {attention_np.sum():.4f}")

        unique_codes, counts = np.unique(codes_np, return_counts=True)
        code_str = ', '.join([f'{c}({"H" if healthy_mask[c] else "C"}):{n}'
                              for c, n in zip(unique_codes, counts)])
        print(f"  Codes: {code_str}")

        volume_dhw = np.transpose(volume, (2, 0, 1))
        D_vol, H_vol, W_vol = volume_dhw.shape
        seg_vol, att_vol = build_3d_maps(
            coords_list, seg_mask_np, attention_np, patch_size, (D_vol, H_vol, W_vol)
        )
        seg_vol_binary = (seg_vol > 0.5).astype(np.float32)
        seg_vol_binary = largest_connected_component_3d(seg_vol_binary)
        seg_vol = seg_vol * seg_vol_binary

        gt_vol_dhw = None
        dice_info = None
        dice_att_info = None
        patch_metrics = None
        gt_path = get_gt_mask_path(volume_path)
        if gt_path is not None:
            gt_mask_hwz = load_gt_mask(gt_path)
            gt_vol_dhw = np.transpose(gt_mask_hwz, (2, 0, 1))
            if gt_vol_dhw.shape == seg_vol.shape:
                dice_info = compute_dice_iou(seg_vol, gt_vol_dhw)
                att_weighted_seg = seg_vol * att_vol
                att_seg_max = att_weighted_seg.max()
                if att_seg_max > 0:
                    att_weighted_seg = att_weighted_seg / att_seg_max
                dice_att_info = compute_dice_iou(att_weighted_seg, gt_vol_dhw)
                patch_metrics = compute_patch_level_metrics(
                    coords_list, seg_mask_np, attention_np,
                    gt_mask_hwz, patch_size
                )
                gt_tumor_pct = 100 * gt_vol_dhw.sum() / gt_vol_dhw.size
                print(f"  GT mask: tumor voxels={int(gt_vol_dhw.sum())}/{gt_vol_dhw.size} "
                      f"({gt_tumor_pct:.2f}%)")
                print(f"  Voxel-level  -> Dice={dice_info['dice']:.4f}, "
                      f"IoU={dice_info['iou']:.4f}, "
                      f"Prec={dice_info['precision']:.4f}, "
                      f"Rec={dice_info['recall']:.4f}")
                print(f"  Patch-level  -> F1={patch_metrics['patch_f1']:.4f}, "
                      f"Prec={patch_metrics['patch_precision']:.4f}, "
                      f"Rec={patch_metrics['patch_recall']:.4f} "
                      f"(TP={patch_metrics['patch_tp']}, FP={patch_metrics['patch_fp']}, "
                      f"FN={patch_metrics['patch_fn']}, "
                      f"GT+={patch_metrics['gt_positive_patches']}/{n_patches})")
                print(f"  Attn on GT tumor patches: "
                      f"{patch_metrics['attention_on_gt_tumor']:.4f} / "
                      f"{patch_metrics['attention_total']:.4f} "
                      f"({100*patch_metrics['attention_gt_ratio']:.1f}%)")
            else:
                print(f"  GT shape mismatch: pred={seg_vol.shape} vs gt={gt_vol_dhw.shape}")
                gt_vol_dhw = None
        else:
            print(f"  GT mask not found")

        safe_name = sample_name.replace('/', '_').replace('\\', '_')
        vis_path = os.path.join(args.output_dir, f'{safe_name}_slices.png')
        visualize_slices(
            volume_dhw, seg_vol, att_vol, vis_path, sample_name,
            label, pred_class, pred_prob,
            gt_volume=gt_vol_dhw, dice_info=dice_info
        )
        print(f"  Saved: {vis_path}")

        sample_result = {
            'sample': sample_name,
            'label': label,
            'pred_class': pred_class,
            'pred_prob': round(pred_prob, 4),
            'n_patches': n_patches,
            'cancer_patches': int(cancer_patches),
            'cancer_ratio': round(float(cancer_patches / n_patches), 4),
            'cancer_attention_ratio': round(float(cancer_attention / max(attention_np.sum(), 1e-8)), 4),
            'unique_codes': len(unique_codes),
        }
        if dice_info is not None:
            sample_result['dice'] = round(dice_info['dice'], 4)
            sample_result['iou'] = round(dice_info['iou'], 4)
            sample_result['seg_precision'] = round(dice_info['precision'], 4)
            sample_result['seg_recall'] = round(dice_info['recall'], 4)
        if dice_att_info is not None:
            sample_result['dice_attn'] = round(dice_att_info['dice'], 4)
            sample_result['iou_attn'] = round(dice_att_info['iou'], 4)
            sample_result['seg_precision_attn'] = round(dice_att_info['precision'], 4)
            sample_result['seg_recall_attn'] = round(dice_att_info['recall'], 4)
        if patch_metrics is not None:
            sample_result.update(patch_metrics)
        results.append(sample_result)

    stats_path = os.path.join(args.output_dir, 'codebook_usage.png')
    visualize_codebook_stats(global_code_counts, healthy_mask, stats_path)
    print(f"\nCodebook usage chart saved: {stats_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    tp = sum(1 for r in results if r['label'] == 1 and r['pred_class'] == 1)
    tn = sum(1 for r in results if r['label'] == 0 and r['pred_class'] == 0)
    fp = sum(1 for r in results if r['label'] == 0 and r['pred_class'] == 1)
    fn = sum(1 for r in results if r['label'] == 1 and r['pred_class'] == 0)
    total = len(results)
    print(f"Classification: TP={tp}, TN={tn}, FP={fp}, FN={fn}, Total={total}")
    if total > 0:
        print(f"Accuracy: {(tp+tn)/total:.4f}")

    pos_results = [r for r in results if r['label'] == 1]
    neg_results = [r for r in results if r['label'] == 0]
    if pos_results:
        avg_cancer_ratio = np.mean([r['cancer_ratio'] for r in pos_results])
        avg_cancer_att = np.mean([r['cancer_attention_ratio'] for r in pos_results])
        print(f"\nPositive samples ({len(pos_results)}):")
        print(f"  Avg cancer patch ratio: {avg_cancer_ratio:.4f}")
        print(f"  Avg attention on cancer patches: {avg_cancer_att:.4f}")
    if neg_results:
        avg_cancer_ratio_neg = np.mean([r['cancer_ratio'] for r in neg_results])
        avg_cancer_att_neg = np.mean([r['cancer_attention_ratio'] for r in neg_results])
        print(f"\nNegative samples ({len(neg_results)}):")
        print(f"  Avg cancer patch ratio: {avg_cancer_ratio_neg:.4f}")
        print(f"  Avg attention on cancer patches: {avg_cancer_att_neg:.4f}")

    if pos_results and neg_results:
        print(f"\nSeparation quality:")
        pos_ratios = [r['cancer_ratio'] for r in pos_results]
        neg_ratios = [r['cancer_ratio'] for r in neg_results]
        print(f"  Positive cancer ratio: {np.mean(pos_ratios):.4f} +/- {np.std(pos_ratios):.4f}")
        print(f"  Negative cancer ratio: {np.mean(neg_ratios):.4f} +/- {np.std(neg_ratios):.4f}")

    dice_results = [r for r in results if 'dice' in r]
    if dice_results:
        print(f"\n--- Segmentation vs GT (Codebook) ---")
        pos_dice = [r for r in dice_results if r['label'] == 1]
        neg_dice = [r for r in dice_results if r['label'] == 0]
        all_dices = [r['dice'] for r in dice_results]
        all_ious = [r['iou'] for r in dice_results]
        print(f"  All samples ({len(dice_results)}):")
        print(f"    Dice: {np.mean(all_dices):.4f} +/- {np.std(all_dices):.4f}")
        print(f"    IoU:  {np.mean(all_ious):.4f} +/- {np.std(all_ious):.4f}")
        if pos_dice:
            pd = [r['dice'] for r in pos_dice]
            pi = [r['iou'] for r in pos_dice]
            pp = [r['seg_precision'] for r in pos_dice]
            pr = [r['seg_recall'] for r in pos_dice]
            print(f"  Positive ({len(pos_dice)}):")
            print(f"    Dice: {np.mean(pd):.4f} +/- {np.std(pd):.4f}")
            print(f"    IoU:  {np.mean(pi):.4f} +/- {np.std(pi):.4f}")
            print(f"    Prec: {np.mean(pp):.4f}, Rec: {np.mean(pr):.4f}")
        if neg_dice:
            nd = [r['dice'] for r in neg_dice]
            ni = [r['iou'] for r in neg_dice]
            print(f"  Negative ({len(neg_dice)}):")
            print(f"    Dice: {np.mean(nd):.4f} +/- {np.std(nd):.4f}")
            print(f"    IoU:  {np.mean(ni):.4f} +/- {np.std(ni):.4f}")

    patch_results = [r for r in results if 'patch_f1' in r]
    if patch_results:
        print(f"\n--- Patch-level Localization ---")
        pos_pm = [r for r in patch_results if r['label'] == 1]
        neg_pm = [r for r in patch_results if r['label'] == 0]
        gt_pos_pm = [r for r in patch_results if r.get('has_gt_positive_patch', False)]
        gt_empty_pm = [r for r in patch_results if not r.get('has_gt_positive_patch', False)]
        all_pf1 = [r['patch_f1'] for r in patch_results]
        all_pp = [r['patch_precision'] for r in patch_results]
        all_pr = [r['patch_recall'] for r in patch_results]
        all_att_gt = [r['attention_gt_ratio'] for r in patch_results]
        print(f"  All samples ({len(patch_results)}):")
        print(f"    Patch F1:   {np.mean(all_pf1):.4f} +/- {np.std(all_pf1):.4f}")
        print(f"    Patch Prec: {np.mean(all_pp):.4f} +/- {np.std(all_pp):.4f}")
        print(f"    Patch Rec:  {np.mean(all_pr):.4f} +/- {np.std(all_pr):.4f}")
        print(f"    Attn on GT: {np.mean(all_att_gt):.4f} +/- {np.std(all_att_gt):.4f}")
        if pos_pm:
            print(f"  Positive ({len(pos_pm)}):")
            print(f"    Patch F1:   {np.mean([r['patch_f1'] for r in pos_pm]):.4f}")
            print(f"    Patch Prec: {np.mean([r['patch_precision'] for r in pos_pm]):.4f}")
            print(f"    Patch Rec:  {np.mean([r['patch_recall'] for r in pos_pm]):.4f}")
            print(f"    Attn on GT: {np.mean([r['attention_gt_ratio'] for r in pos_pm]):.4f}")
        if neg_pm:
            print(f"  Negative ({len(neg_pm)}):")
            print(f"    Patch F1:   {np.mean([r['patch_f1'] for r in neg_pm]):.4f}")
            print(f"    Patch Prec: {np.mean([r['patch_precision'] for r in neg_pm]):.4f}")
            print(f"    Patch Rec:  {np.mean([r['patch_recall'] for r in neg_pm]):.4f}")
            print(f"    Attn on GT: {np.mean([r['attention_gt_ratio'] for r in neg_pm]):.4f}")
        if gt_pos_pm:
            print(f"  GT-positive ({len(gt_pos_pm)}):")
            print(f"    Patch F1:   {np.mean([r['patch_f1'] for r in gt_pos_pm]):.4f}")
            print(f"    Patch Prec: {np.mean([r['patch_precision'] for r in gt_pos_pm]):.4f}")
            print(f"    Patch Rec:  {np.mean([r['patch_recall'] for r in gt_pos_pm]):.4f}")
        if gt_empty_pm:
            print(f"  GT-empty ({len(gt_empty_pm)}):")
            print(f"    Patch F1:   {np.mean([r['patch_f1'] for r in gt_empty_pm]):.4f}")
            print(f"    Patch Prec: {np.mean([r['patch_precision'] for r in gt_empty_pm]):.4f}")
            print(f"    Patch Rec:  {np.mean([r['patch_recall'] for r in gt_empty_pm]):.4f}")

    dice_att_results = [r for r in results if 'dice_attn' in r]
    if dice_att_results:
        print(f"\n--- Segmentation vs GT (Attention-weighted) ---")
        pos_da = [r for r in dice_att_results if r['label'] == 1]
        neg_da = [r for r in dice_att_results if r['label'] == 0]
        all_da = [r['dice_attn'] for r in dice_att_results]
        all_ia = [r['iou_attn'] for r in dice_att_results]
        print(f"  All samples ({len(dice_att_results)}):")
        print(f"    Dice: {np.mean(all_da):.4f} +/- {np.std(all_da):.4f}")
        print(f"    IoU:  {np.mean(all_ia):.4f} +/- {np.std(all_ia):.4f}")
        if pos_da:
            pda = [r['dice_attn'] for r in pos_da]
            pia = [r['iou_attn'] for r in pos_da]
            ppa = [r['seg_precision_attn'] for r in pos_da]
            pra = [r['seg_recall_attn'] for r in pos_da]
            print(f"  Positive ({len(pos_da)}):")
            print(f"    Dice: {np.mean(pda):.4f} +/- {np.std(pda):.4f}")
            print(f"    IoU:  {np.mean(pia):.4f} +/- {np.std(pia):.4f}")
            print(f"    Prec: {np.mean(ppa):.4f}, Rec: {np.mean(pra):.4f}")
        if neg_da:
            nda = [r['dice_attn'] for r in neg_da]
            nia = [r['iou_attn'] for r in neg_da]
            print(f"  Negative ({len(neg_da)}):")
            print(f"    Dice: {np.mean(nda):.4f} +/- {np.std(nda):.4f}")
            print(f"    IoU:  {np.mean(nia):.4f} +/- {np.std(nia):.4f}")

    print(f"\nCodebook activation summary:")
    active_codes = (global_code_counts > 0).sum()
    healthy_active = sum(1 for i in range(num_embeddings) if global_code_counts[i] > 0 and healthy_mask[i])
    cancer_active = sum(1 for i in range(num_embeddings) if global_code_counts[i] > 0 and not healthy_mask[i])
    print(f"  Active codes: {active_codes}/{num_embeddings}")
    print(f"  Healthy codes active: {healthy_active}/{int(healthy_mask.sum())}")
    print(f"  Cancer codes active: {cancer_active}/{int(num_embeddings - healthy_mask.sum())}")

    report = {
        'checkpoint': args.checkpoint,
        'checkpoint_epoch': ckpt_epoch,
        'num_samples': total,
        'classification': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
        'codebook': {
            'num_embeddings': num_embeddings,
            'healthy': int(healthy_mask.sum()),
            'cancer': int(num_embeddings - healthy_mask.sum()),
            'active': int(active_codes),
        },
    }
    dice_results = [r for r in results if 'dice' in r]
    if dice_results:
        pos_d = [r for r in dice_results if r['label'] == 1]
        neg_d = [r for r in dice_results if r['label'] == 0]
        report['segmentation_codebook'] = {
            'mean_dice': round(float(np.mean([r['dice'] for r in dice_results])), 4),
            'mean_iou': round(float(np.mean([r['iou'] for r in dice_results])), 4),
        }
        if pos_d:
            report['segmentation_codebook']['positive_mean_dice'] = round(
                float(np.mean([r['dice'] for r in pos_d])), 4)
        if neg_d:
            report['segmentation_codebook']['negative_mean_dice'] = round(
                float(np.mean([r['dice'] for r in neg_d])), 4)
    dice_att_results = [r for r in results if 'dice_attn' in r]
    if dice_att_results:
        pos_da = [r for r in dice_att_results if r['label'] == 1]
        neg_da = [r for r in dice_att_results if r['label'] == 0]
        report['segmentation_attention'] = {
            'mean_dice': round(float(np.mean([r['dice_attn'] for r in dice_att_results])), 4),
            'mean_iou': round(float(np.mean([r['iou_attn'] for r in dice_att_results])), 4),
        }
        if pos_da:
            report['segmentation_attention']['positive_mean_dice'] = round(
                float(np.mean([r['dice_attn'] for r in pos_da])), 4)
        if neg_da:
            report['segmentation_attention']['negative_mean_dice'] = round(
                float(np.mean([r['dice_attn'] for r in neg_da])), 4)
    report['samples'] = results
    report_path = os.path.join(args.output_dir, 'results.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to: {report_path}")
    print(f"Visualizations saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
