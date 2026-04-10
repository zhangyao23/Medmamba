import argparse
import os
import sys
import json
import importlib.util

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from src.training import Config
from src.data.volumetric_loader import create_volumetric_dataloader
from src.utils.metrics import evaluate_bag_level

spec = importlib.util.spec_from_file_location(
    "train_ddp", os.path.join(os.path.dirname(__file__), "train_volumetric_ddp.py"))
train_ddp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_ddp)
Volumetric3DMIL = train_ddp.Volumetric3DMIL


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


def load_gt_mask_hwz(gt_path):
    if gt_path is None or not os.path.exists(gt_path):
        return None
    img = nib.load(gt_path)
    img = nib.as_closest_canonical(img)
    mask = img.get_fdata()
    if mask.ndim == 4:
        mask = mask[:, :, :, 0]
    tumor_label = _get_tumor_label(gt_path)
    if tumor_label is not None:
        return (mask == tumor_label).astype(np.float32)
    return (mask > 0).astype(np.float32)


def compute_patch_positive_labels(coords_np, gt_mask_hwz, patch_size):
    pd, ph, pw = patch_size
    H, W, D = gt_mask_hwz.shape
    n = len(coords_np)
    labels = np.zeros(n, dtype=np.float32)
    for i in range(n):
        z, y, x = int(coords_np[i, 0]), int(coords_np[i, 1]), int(coords_np[i, 2])
        z_e = min(z + pd, D)
        y_e = min(y + ph, H)
        x_e = min(x + pw, W)
        labels[i] = float(gt_mask_hwz[y:y_e, x:x_e, z:z_e].sum() > 0)
    return labels


def seg_metrics_at_thr(probs_list, gt_list, bag_filter, thr):
    tp = fp = fn = tn = 0
    for p, g, keep in zip(probs_list, gt_list, bag_filter):
        if not keep:
            continue
        pred = (p >= thr).astype(bool)
        gt = g.astype(bool)
        tp += int((pred & gt).sum())
        fp += int((pred & ~gt).sum())
        fn += int((~pred & gt).sum())
        tn += int((~pred & ~gt).sum())
    dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return {
        'dice': dice, 'sensitivity': sensitivity, 'specificity': specificity,
        'precision': precision, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }


def plot_roc_curve(labels_np, probs_np, output_path, model_name='v20'):
    fpr, tpr, thresholds = roc_curve(labels_np, probs_np)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    ax.plot(fpr, tpr, color='#2563eb', lw=2.5,
            label=f'{model_name} (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)

    key_thrs = [0.1, 0.2, 0.3, 0.35, 0.4, 0.5]
    for t in key_thrs:
        idx = np.argmin(np.abs(thresholds - t))
        ax.scatter(fpr[idx], tpr[idx], s=50, zorder=5)
        ax.annotate(f'thr={t}', (fpr[idx], tpr[idx]),
                    textcoords="offset points", xytext=(8, -8), fontsize=8)

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=13)
    ax.set_title(f'ROC Curve - {model_name} (Weakly Supervised)', fontsize=14)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ROC curve saved to {output_path}")
    return roc_auc


def plot_threshold_curves(labels_np, probs_np, output_path, model_name='v20'):
    thresholds = np.arange(0.01, 0.99, 0.01)
    accs, precs, recs, f1s, specs, fps_list = [], [], [], [], [], []

    n_neg = int((labels_np == 0).sum())
    n_pos = int((labels_np == 1).sum())

    for thr in thresholds:
        pred = (probs_np >= thr).astype(int)
        tp = int(((pred == 1) & (labels_np == 1)).sum())
        tn_val = int(((pred == 0) & (labels_np == 0)).sum())
        fp_val = int(((pred == 1) & (labels_np == 0)).sum())
        fn_val = int(((pred == 0) & (labels_np == 1)).sum())
        acc = (tp + tn_val) / len(labels_np) if len(labels_np) > 0 else 0
        prec = tp / (tp + fp_val) if (tp + fp_val) > 0 else 1.0
        rec = tp / (tp + fn_val) if (tp + fn_val) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        spec = tn_val / (tn_val + fp_val) if (tn_val + fp_val) > 0 else 1.0
        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        specs.append(spec)
        fps_list.append(fp_val)

    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    ax1 = axes[0]
    ax1.plot(thresholds, accs, label='Accuracy', lw=2, color='#059669')
    ax1.plot(thresholds, precs, label='Precision', lw=2, color='#dc2626')
    ax1.plot(thresholds, recs, label='Recall (Sensitivity)', lw=2, color='#2563eb')
    ax1.plot(thresholds, f1s, label='F1 Score', lw=2, color='#9333ea')
    ax1.plot(thresholds, specs, label='Specificity', lw=2, color='#ea580c', linestyle='--')
    ax1.set_xlabel('Threshold', fontsize=13)
    ax1.set_ylabel('Metric Value', fontsize=13)
    ax1.set_title(f'Classification Metrics vs Threshold - {model_name}', fontsize=14)
    ax1.legend(fontsize=11, loc='center left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([-0.02, 1.05])

    ax2 = axes[1]
    ax2.plot(thresholds, fps_list, lw=2, color='#dc2626')
    ax2.set_xlabel('Threshold', fontsize=13)
    ax2.set_ylabel('Number of False Positives', fontsize=13)
    ax2.set_title(f'False Positives vs Threshold - {model_name} (N_neg={n_neg})', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])

    fp_zero_thr = None
    for i, fp in enumerate(fps_list):
        if fp == 0:
            fp_zero_thr = thresholds[i]
            break
    if fp_zero_thr is not None:
        ax2.axvline(x=fp_zero_thr, color='#059669', linestyle='--', lw=1.5,
                     label=f'FP=0 @ thr={fp_zero_thr:.2f}')
        ax2.legend(fontsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Threshold curves saved to {output_path}")


def plot_seg_metrics_curves(seg_probs_list, seg_gt_list, seg_pos_bag, output_path, model_name='v20'):
    thresholds = np.arange(0.05, 0.96, 0.02)
    dices, sens_list, spec_list, prec_list = [], [], [], []

    for thr in thresholds:
        m = seg_metrics_at_thr(seg_probs_list, seg_gt_list, seg_pos_bag, thr)
        dices.append(m['dice'])
        sens_list.append(m['sensitivity'])
        spec_list.append(m['specificity'])
        prec_list.append(m['precision'])

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(thresholds, dices, label='Dice', lw=2.5, color='#2563eb')
    ax.plot(thresholds, sens_list, label='Sensitivity (Recall)', lw=2, color='#dc2626')
    ax.plot(thresholds, spec_list, label='Specificity', lw=2, color='#059669')
    ax.plot(thresholds, prec_list, label='Precision', lw=2, color='#ea580c', linestyle='--')

    best_idx = int(np.argmax(dices))
    ax.scatter(thresholds[best_idx], dices[best_idx], s=80, color='#2563eb', zorder=5)
    ax.annotate(f'Best Dice={dices[best_idx]:.4f}\n@thr={thresholds[best_idx]:.2f}',
                (thresholds[best_idx], dices[best_idx]),
                textcoords="offset points", xytext=(10, -15), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='black'))

    ax.set_xlabel('Threshold', fontsize=13)
    ax.set_ylabel('Metric Value', fontsize=13)
    ax.set_title(f'Patch-Level Segmentation Metrics vs Threshold - {model_name}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([-0.02, 1.05])
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Seg metrics curves saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_patches', type=int, default=None)
    parser.add_argument('--adaptive_norm', action='store_true', default=False)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--use_codebook_seg', action='store_true',
                        help='Force using codebook cancer_scores for patch seg instead of patch_seg_head')
    parser.add_argument('--use_recon_seg', action='store_true',
                        help='Use reconstruction error as patch seg signal')
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    ckpt_basename = os.path.splitext(os.path.basename(args.checkpoint))[0]
    model_name = args.model_name or ckpt_basename
    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(args.checkpoint)), 'eval_results', ckpt_basename)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    sd_keys = set(ckpt['model_state_dict'].keys())
    has_spatial_decoder = any('spatial_decoder' in k for k in sd_keys)
    has_voxel_decoder = any('voxel_decoder' in k for k in sd_keys)
    if has_voxel_decoder and not has_spatial_decoder:
        print("  Detected MLP voxel decoder in checkpoint, overriding config")
        config.model['voxel_seg']['decoder_type'] = 'mlp'
    elif has_spatial_decoder:
        config.model['voxel_seg']['decoder_type'] = 'spatial_unet'

    max_patches = args.max_patches or config.data.get('max_patches', 256)
    print(f"  max_patches={max_patches}, adaptive_norm={args.adaptive_norm}")

    print(f"Loading model from {args.checkpoint} ...")
    model = Volumetric3DMIL(config).to(device)
    model_sd = model.state_dict()
    ckpt_sd = ckpt['model_state_dict']
    filtered_sd = {}
    skipped_shape = []
    for k, v in ckpt_sd.items():
        if k in model_sd and model_sd[k].shape != v.shape:
            skipped_shape.append(k)
        else:
            filtered_sd[k] = v
    if skipped_shape:
        print(f"  Skipped {len(skipped_shape)} keys due to shape mismatch: {skipped_shape}")
    missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    old_decoder_keys = [k for k in unexpected if k.startswith('voxel_decoder.decoder.')]
    if old_decoder_keys and hasattr(model, 'voxel_decoder'):
        print("  Rebuilding old-style PatchVoxelDecoder from checkpoint weights...")
        sd = ckpt['model_state_dict']
        fc_w = sd['voxel_decoder.fc.weight']
        fc_b = sd['voxel_decoder.fc.bias']
        input_dim = fc_w.shape[1]
        total_fc_out = fc_w.shape[0]

        conv_layers = {}
        for k in sorted(old_decoder_keys):
            idx_str = k.split('voxel_decoder.decoder.')[1].split('.')[0]
            idx = int(idx_str)
            if idx not in conv_layers:
                conv_layers[idx] = {}
            param_name = k.split('.')[-1]
            conv_layers[idx][param_name] = sd[k]

        layers = []
        for idx in sorted(conv_layers.keys()):
            p = conv_layers[idx]
            if 'weight' in p and p['weight'].ndim == 5:
                in_ch, out_ch = p['weight'].shape[0], p['weight'].shape[1]
                ks = p['weight'].shape[2]
                layer = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=ks, stride=2, padding=1)
                layer.weight.data.copy_(p['weight'])
                layer.bias.data.copy_(p['bias'])
                layers.append(layer)
            elif 'weight' in p and p['weight'].ndim == 1:
                n_feat = p['weight'].shape[0]
                ng = min(32, n_feat)
                layer = nn.GroupNorm(ng, n_feat)
                layer.weight.data.copy_(p['weight'])
                layer.bias.data.copy_(p['bias'])
                layers.append(layer)
                layers.append(nn.ReLU(inplace=True))

        first_conv_in = None
        for l in layers:
            if isinstance(l, nn.ConvTranspose3d):
                first_conv_in = l.in_channels
                break

        vol_per_ch = total_fc_out // first_conv_in
        init_side = round(vol_per_ch ** (1.0 / 3.0))
        init_spatial = (init_side, init_side, init_side)

        class OldPatchVoxelDecoder(nn.Module):
            def __init__(self, fc, decoder_seq, init_sp, first_ch):
                super().__init__()
                self.fc = fc
                self.decoder = decoder_seq
                self.init_spatial = init_sp
                self.first_ch = first_ch
            def forward(self, seg_input):
                B, N, D_in = seg_input.shape
                x = self.fc(seg_input)
                sd, sh, sw = self.init_spatial
                x = x.view(B * N, self.first_ch, sd, sh, sw)
                x = self.decoder(x)
                x = x[:, 0]
                x = x.view(B, N, *x.shape[1:])
                return x

        fc_layer = nn.Linear(input_dim, total_fc_out)
        fc_layer.weight.data.copy_(fc_w)
        fc_layer.bias.data.copy_(fc_b)
        decoder_seq = nn.Sequential(*layers)
        old_decoder = OldPatchVoxelDecoder(
            fc_layer, decoder_seq, init_spatial,
            first_conv_in
        ).to(device)
        model.voxel_decoder = old_decoder
        model.use_spatial_decoder = False
        model.voxel_seg_enabled = True
        print(f"  Old decoder rebuilt: input_dim={input_dim}, init_spatial={init_spatial}")
    model.codebook.phase1_complete = True
    if model.codebook.healthy_code_mask.any():
        model.codebook.use_dynamic_partition = True
        model.codebook.healthy_frozen = True
        model.codebook.frozen_code_mask = model.codebook.healthy_code_mask.clone()
    model.eval()
    print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")

    val_loader = create_volumetric_dataloader(
        json_path=config.data['test_json'],
        batch_size=config.data['batch_size'],
        patch_size=tuple(config.data['patch_size']),
        stride=tuple(config.data['stride']),
        num_workers=config.data['num_workers'],
        shuffle=False,
        balanced_sampling=False,
        max_patches=max_patches,
        use_ddp=False,
        min_hu=config.data['min_hu'],
        max_hu=config.data['max_hu'],
        adaptive_norm=args.adaptive_norm
    )
    print(f"Test set: {len(val_loader.dataset)} samples, {len(val_loader)} batches")

    patch_size = tuple(config.data['patch_size'])
    mini_batch_size = config.model['feature_extractor'].get('mini_batch_size', 8)

    all_preds = []
    all_labels = []
    all_seg_probs = []
    all_seg_gt = []
    all_seg_is_pos_bag = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            patches = batch['patches']
            coords = batch['coords'].to(device)
            labels = batch['labels'].to(device)
            masks = batch['masks'].to(device)

            with torch.amp.autocast('cuda', enabled=True):
                outputs = model(
                    patches, coords, None, masks, mini_batch_size=mini_batch_size
                )
                logits = outputs[0]
                codes = outputs[3]
                recon_patches = outputs[6] if len(outputs) > 6 else None
                patch_seg_logits = outputs[8] if len(outputs) > 8 else None

            all_preds.append(logits)
            all_labels.append(labels)

            for b in range(labels.shape[0]):
                valid_n = int(masks[b].sum().item())
                if valid_n <= 0:
                    continue
                volume_path = batch['volume_paths'][b]
                gt_path = get_gt_mask_path(volume_path)
                gt_mask_hwz = load_gt_mask_hwz(gt_path)
                if gt_mask_hwz is None:
                    continue

                if args.use_recon_seg and recon_patches is not None:
                    orig = patches[b, :valid_n].float().to(recon_patches.device)
                    recon = recon_patches[b, :valid_n].float()
                    recon_err = (orig - recon).abs().mean(dim=(1, 2, 3, 4))
                    re_np = recon_err.cpu().numpy()
                    re_min, re_max = re_np.min(), re_np.max()
                    if re_max - re_min > 1e-8:
                        seg_prob = (re_np - re_min) / (re_max - re_min)
                    else:
                        seg_prob = np.zeros_like(re_np)
                elif args.use_codebook_seg:
                    cancer_scores = model.codebook.get_cancer_scores()
                    seg_prob = cancer_scores[codes[b, :valid_n]].float().cpu().numpy()
                elif patch_seg_logits is not None:
                    seg_prob = torch.sigmoid(patch_seg_logits[b, :valid_n, 0]).cpu().numpy()
                else:
                    seg_prob = model.codebook.codes_to_segmentation_mask(
                        codes[b:b+1, :valid_n]
                    )[0].cpu().numpy().astype(np.float32)

                coords_b = coords[b, :valid_n].cpu().numpy()
                gt_labels_b = compute_patch_positive_labels(coords_b, gt_mask_hwz, patch_size)

                all_seg_probs.append(seg_prob)
                all_seg_gt.append(gt_labels_b)
                all_seg_is_pos_bag.append(int(labels[b].item()) == 1)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = evaluate_bag_level(all_preds, all_labels, fixed_threshold=0.35)

    print("\n" + "=" * 70)
    print("CLASSIFICATION RESULTS (full test set)")
    print("=" * 70)
    print(f"  AUC:       {metrics['auc']:.4f}")
    print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"  Best F1:   {metrics['f1_best']:.4f} @ threshold={metrics['best_threshold']:.2f}")
    print(f"  Acc@best:  {metrics['accuracy_best']:.4f}")
    print(f"  --- Fixed threshold = {metrics['fixed_threshold']:.2f} ---")
    print(f"  Accuracy:  {metrics['accuracy_fixed']:.4f}")
    print(f"  Precision: {metrics['precision_fixed']:.4f}")
    print(f"  Recall:    {metrics['recall_fixed']:.4f}")
    print(f"  F1:        {metrics['f1_fixed']:.4f}")
    print(f"  Specificity: {metrics['specificity_fixed']:.4f}")

    probs_np = torch.softmax(all_preds, dim=1)[:, 1].cpu().numpy()
    labels_np = all_labels.cpu().numpy()
    for thr in [0.1, 0.2, 0.3, 0.35, 0.4, 0.5]:
        pred_bin = (probs_np >= thr).astype(int)
        tp = int(((pred_bin == 1) & (labels_np == 1)).sum())
        tn = int(((pred_bin == 0) & (labels_np == 0)).sum())
        fp = int(((pred_bin == 1) & (labels_np == 0)).sum())
        fn = int(((pred_bin == 0) & (labels_np == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        print(f"  thr={thr:.2f}: TP={tp}, FP={fp}, FN={fn}, TN={tn} | "
              f"Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, Spec={spec:.4f}")

    print("\n--- Saving predictions ---")
    np.savez(
        os.path.join(args.output_dir, 'predictions.npz'),
        cls_probs=probs_np,
        cls_labels=labels_np,
        cls_logits=all_preds.cpu().numpy(),
        auc=metrics['auc'],
        pr_auc=metrics['pr_auc'],
    )
    print(f"  Saved predictions.npz ({len(probs_np)} samples)")

    print("\n--- Generating classification plots ---")
    plot_roc_curve(labels_np, probs_np,
                   os.path.join(args.output_dir, 'roc_curve.png'), model_name)
    plot_threshold_curves(labels_np, probs_np,
                          os.path.join(args.output_dir, 'threshold_curves.png'), model_name)

    if len(all_seg_probs) > 0:
        print("\n" + "=" * 70)
        print("PATCH-LEVEL SEGMENTATION")
        print("=" * 70)

        best_dice = 0.0
        best_thr = 0.5
        for thr in np.arange(0.1, 0.95, 0.02):
            m = seg_metrics_at_thr(all_seg_probs, all_seg_gt, all_seg_is_pos_bag, thr)
            if m['dice'] > best_dice:
                best_dice = m['dice']
                best_thr = thr

        all_true = [True] * len(all_seg_probs)

        m_best = seg_metrics_at_thr(all_seg_probs, all_seg_gt, all_seg_is_pos_bag, best_thr)
        m_05 = seg_metrics_at_thr(all_seg_probs, all_seg_gt, all_seg_is_pos_bag, 0.5)

        print(f"  --- Best threshold = {best_thr:.2f} (pos bags only) ---")
        print(f"    Dice:        {m_best['dice']:.4f}")
        print(f"    Sensitivity: {m_best['sensitivity']:.4f}")
        print(f"    Specificity: {m_best['specificity']:.4f}")
        print(f"    Precision:   {m_best['precision']:.4f}")
        print(f"    TP={m_best['tp']}, FP={m_best['fp']}, FN={m_best['fn']}, TN={m_best['tn']}")

        print(f"  --- Fixed threshold = 0.50 (pos bags only) ---")
        print(f"    Dice:        {m_05['dice']:.4f}")
        print(f"    Sensitivity: {m_05['sensitivity']:.4f}")
        print(f"    Specificity: {m_05['specificity']:.4f}")
        print(f"    Precision:   {m_05['precision']:.4f}")
        print(f"    TP={m_05['tp']}, FP={m_05['fp']}, FN={m_05['fn']}, TN={m_05['tn']}")

        print(f"\n  --- Threshold sweep (pos bags) ---")
        print(f"  {'thr':>5s}  {'Dice':>7s}  {'Sens':>7s}  {'Spec':>7s}  {'Prec':>7s}  {'TP':>5s}  {'FP':>5s}  {'FN':>5s}")
        for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            m = seg_metrics_at_thr(all_seg_probs, all_seg_gt, all_seg_is_pos_bag, thr)
            print(f"  {thr:5.2f}  {m['dice']:7.4f}  {m['sensitivity']:7.4f}  "
                  f"{m['specificity']:7.4f}  {m['precision']:7.4f}  "
                  f"{m['tp']:5d}  {m['fp']:5d}  {m['fn']:5d}")

        print(f"\n  Samples with GT: {len(all_seg_probs)} "
              f"(pos={sum(all_seg_is_pos_bag)}, neg={len(all_seg_probs)-sum(all_seg_is_pos_bag)})")

        seg_probs_flat = np.concatenate([p for p, is_pos in zip(all_seg_probs, all_seg_is_pos_bag) if is_pos])
        seg_gt_flat = np.concatenate([g for g, is_pos in zip(all_seg_gt, all_seg_is_pos_bag) if is_pos])
        np.savez(
            os.path.join(args.output_dir, 'seg_predictions.npz'),
            seg_probs=seg_probs_flat,
            seg_gt=seg_gt_flat,
            best_dice=best_dice,
            best_thr=best_thr,
        )
        print(f"  Saved seg_predictions.npz ({len(seg_probs_flat)} patches)")

        print("\n--- Generating segmentation plots ---")
        plot_seg_metrics_curves(all_seg_probs, all_seg_gt, all_seg_is_pos_bag,
                                os.path.join(args.output_dir, 'seg_metrics_curves.png'),
                                model_name)

    print("\n" + "=" * 70)
    print("DONE - All results saved to: " + args.output_dir)
    print("=" * 70)


if __name__ == '__main__':
    main()
