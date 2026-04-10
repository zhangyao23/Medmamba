import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

from src.training import Config
from src.models.feature_extractor_3d import VolumetricFeatureExtractor
from src.models.baseline_heads import GAPHead, ABMILHead, TransMILHead, DSMILHead
from src.data.volumetric_loader import create_volumetric_dataloader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BaselineMILModel(torch.nn.Module):
    def __init__(self, config, aggregation_method='abmil'):
        super().__init__()
        self.aggregation_method = aggregation_method
        feature_dim = 512
        self.feature_extractor = VolumetricFeatureExtractor(
            arch=config.model['feature_extractor']['arch'],
            spatial_dims=config.model['feature_extractor']['spatial_dims'],
            n_input_channels=config.model['feature_extractor']['n_input_channels'],
            pretrained=config.model['feature_extractor']['pretrained'],
            frozen=config.model['feature_extractor']['frozen']
        )
        if aggregation_method == 'gap':
            self.head = GAPHead(input_dim=feature_dim, num_classes=2)
        elif aggregation_method == 'abmil':
            self.head = ABMILHead(input_dim=feature_dim, hidden_dim=256, num_classes=2)
        elif aggregation_method == 'transmil':
            self.head = TransMILHead(input_dim=feature_dim, num_classes=2, num_layers=2, nhead=8)
        elif aggregation_method == 'dsmil':
            self.head = DSMILHead(input_dim=feature_dim, num_classes=2)

    def forward(self, patches, coords, masks=None, mini_batch_size=8):
        features = self.feature_extractor(patches, mini_batch_size=mini_batch_size)
        logits, attention = self.head(features, mask=masks)
        return logits, attention, None


def build_v20_model(config, ckpt_state_dict=None):
    from src.models.vector_quantizer_3d import PartitionedVectorQuantizer
    from src.models.spatial_scanner_3d import ZOrderSpatialScanner
    from src.models.hilbert_scanner import HilbertCurveSpatialScanner
    from src.models.video_mamba import VideoMamba3D
    from src.models.mil_head import AttentionMILHead
    from src.models.self_correction import SelfCorrectionModule
    from src.models.decoder_3d import VolumetricDecoder3D
    from src.models.patch_voxel_decoder import PatchVoxelDecoder

    class Volumetric3DMIL(torch.nn.Module):
        def __init__(self, cfg, ckpt_state_dict=None):
            super().__init__()
            self.config = cfg
            self.codebook_after_mamba = cfg.model.get('codebook_after_mamba', False)
            feature_dim = cfg.model['codebook']['embedding_dim']

            self.feature_extractor = VolumetricFeatureExtractor(
                arch=cfg.model['feature_extractor']['arch'],
                spatial_dims=cfg.model['feature_extractor']['spatial_dims'],
                n_input_channels=cfg.model['feature_extractor']['n_input_channels'],
                pretrained=cfg.model['feature_extractor']['pretrained'],
                frozen=cfg.model['feature_extractor']['frozen']
            )
            self.codebook = PartitionedVectorQuantizer(
                num_embeddings=cfg.model['codebook']['num_embeddings'],
                embedding_dim=cfg.model['codebook']['embedding_dim'],
                healthy_ratio=cfg.model['codebook']['healthy_ratio'],
                commitment_cost=cfg.model['codebook']['commitment_cost'],
                use_ema=cfg.model['codebook']['use_ema'],
                ema_decay=cfg.model['codebook']['ema_decay'],
                usage_balance_alpha=cfg.model['codebook'].get('usage_balance_alpha', 0.0),
                explore_prob=cfg.model['codebook'].get('explore_prob', 0.0),
                gate_std_factor=cfg.model['codebook'].get('gate_std_factor', 1.5),
                min_cancer_fraction=cfg.model['codebook'].get('min_cancer_fraction', 0.1),
                lambda_code_nu=cfg.model['codebook'].get('lambda_code_nu', 0.5)
            )
            scanner_type = cfg.model.get('spatial_scanner', {}).get('type', 'z_order')
            if scanner_type == 'hilbert':
                self.spatial_scanner = HilbertCurveSpatialScanner(
                    max_order=cfg.model['spatial_scanner'].get('hilbert_max_order', 8))
            else:
                self.spatial_scanner = ZOrderSpatialScanner()

            sc_cfg = cfg.model.get('self_correction', {})
            self.use_self_correction = sc_cfg.get('enabled', False)
            if self.use_self_correction:
                self.self_correction = SelfCorrectionModule(
                    entropy_threshold=sc_cfg.get('entropy_threshold', 0.7),
                    neighbor_radius=sc_cfg.get('neighbor_radius', 2),
                    spatial_weight=sc_cfg.get('spatial_weight', 0.3))

            self.mamba = VideoMamba3D(
                d_model=feature_dim,
                d_state=cfg.model['mamba']['d_state'],
                d_conv=cfg.model['mamba']['d_conv'],
                expand=cfg.model['mamba']['expand'],
                num_layers=cfg.model['mamba']['num_layers'],
                bidirectional=cfg.model['mamba']['bidirectional'])

            self.mil_head = AttentionMILHead(
                input_dim=feature_dim,
                hidden_dim=cfg.model['mil']['hidden_dim'],
                num_classes=cfg.model['mil']['num_classes'])

            seg_head_config = cfg.model.get('seg_head', {})
            self.seg_head_type = seg_head_config.get('type', 'simple')

            seg_input_dim = self._infer_seg_input_dim(feature_dim, ckpt_state_dict)
            self.seg_use_multiscale = (seg_input_dim > feature_dim * 2)

            self.voxel_seg_enabled = self._check_voxel_decoder(ckpt_state_dict)
            if self.voxel_seg_enabled:
                voxel_input_dim = seg_input_dim
                pd, ph, pw = tuple(cfg.data['patch_size'])
                hidden_dims = cfg.model.get('voxel_seg', {}).get('hidden_dims', [256, 128, 64])
                n_up = len(hidden_dims)
                init_d, init_h, init_w = pd // (2 ** n_up), ph // (2 ** n_up), pw // (2 ** n_up)
                first_ch = hidden_dims[0]
                self.voxel_decoder = torch.nn.Module()
                self.voxel_decoder.fc = torch.nn.Linear(voxel_input_dim, first_ch * init_d * init_h * init_w)
                self.voxel_decoder.init_spatial = (init_d, init_h, init_w)
                self.voxel_decoder.first_ch = first_ch
                layers = []
                in_ch = first_ch
                for out_ch in hidden_dims[1:]:
                    layers.append(torch.nn.ConvTranspose3d(in_ch, out_ch, 4, 2, 1))
                    layers.append(torch.nn.GroupNorm(min(32, out_ch), out_ch))
                    layers.append(torch.nn.ReLU(inplace=True))
                    in_ch = out_ch
                layers.append(torch.nn.ConvTranspose3d(in_ch, 1, 4, 2, 1))
                self.voxel_decoder.decoder = torch.nn.Sequential(*layers)
                self.voxel_decoder.patch_size = (pd, ph, pw)

            if self.seg_head_type == 'dual_path':
                self.patch_seg_head = torch.nn.Sequential(
                    torch.nn.Linear(seg_input_dim, 256), torch.nn.LayerNorm(256), torch.nn.ReLU(),
                    torch.nn.Linear(256, 128), torch.nn.LayerNorm(128), torch.nn.ReLU(),
                    torch.nn.Linear(128, 1))
            else:
                self.patch_seg_head = torch.nn.Sequential(
                    torch.nn.Linear(feature_dim, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1))

        def _infer_seg_input_dim(self, feature_dim, ckpt_sd):
            if ckpt_sd and 'patch_seg_head.0.weight' in ckpt_sd:
                return ckpt_sd['patch_seg_head.0.weight'].shape[1]
            return feature_dim * 2 if self.seg_head_type == 'dual_path' else feature_dim

        def _check_voxel_decoder(self, ckpt_sd):
            if ckpt_sd is None:
                return False
            return any(k.startswith('voxel_decoder.') for k in ckpt_sd.keys())

        def forward(self, patches, coords, masks=None, mini_batch_size=8):
            if self.seg_use_multiscale:
                multi_features, features = self.feature_extractor.forward_multiscale(
                    patches, mini_batch_size=mini_batch_size)
            else:
                features = self.feature_extractor(patches, mini_batch_size=mini_batch_size)
                multi_features = None
            B, N, D = features.shape

            if self.codebook_after_mamba:
                sorted_features, sorted_coords, sort_perm = self.spatial_scanner(features, coords)
                mamba_out = self.mamba(sorted_features, mask=masks)
                inv_perm = sort_perm.argsort(dim=1)
                context_orig = torch.gather(mamba_out, 1, inv_perm.unsqueeze(-1).expand(B, N, D))
                quantized, codes, vq_loss = self.codebook(context_orig, None)
                logits, attention = self.mil_head(quantized, mask=masks)
            else:
                quantized, codes, vq_loss = self.codebook(features, None)
                sorted_quantized, sorted_coords, sort_perm = self.spatial_scanner(quantized, coords)
                context = self.mamba(sorted_quantized, mask=masks)
                logits, attention = self.mil_head(context, mask=masks)
                inv_perm = sort_perm.argsort(dim=1)
                context_orig = torch.gather(context, 1, inv_perm.unsqueeze(-1).expand(B, N, D))

            if self.seg_head_type == 'dual_path':
                context_detached = context_orig.detach()
                if self.seg_use_multiscale and multi_features is not None:
                    ms_cat = torch.cat(list(multi_features.values()), dim=-1)
                    seg_input = torch.cat([ms_cat, context_detached], dim=-1)
                else:
                    seg_input = torch.cat([features.detach(), context_detached], dim=-1)
            else:
                seg_input = context_orig.detach()

            if self.voxel_seg_enabled:
                vd = self.voxel_decoder
                _B, _N, _D = seg_input.shape
                x = vd.fc(seg_input)
                ch = vd.first_ch
                x = x.view(_B * _N, ch, *vd.init_spatial)
                x = vd.decoder(x)
                pd, ph, pw = vd.patch_size
                x = x[:, 0, :pd, :ph, :pw]
                voxel_seg_logits = x.view(_B, _N, pd, ph, pw)
                patch_seg_logits = voxel_seg_logits.mean(dim=(2, 3, 4)).unsqueeze(-1)
            else:
                patch_seg_logits = self.patch_seg_head(seg_input)

            return logits, attention, codes, patch_seg_logits

    return Volumetric3DMIL(config, ckpt_state_dict=ckpt_state_dict)


def build_v20_model_with_ckpt(config, ckpt_path, recompute_cancer_score=False):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model_state_dict']
    model = build_v20_model(config, ckpt_state_dict=state_dict)

    if 'codebook.cancer_score' in state_dict and not recompute_cancer_score:
        cs_tensor = state_dict['codebook.cancer_score']
        model.codebook.register_buffer('cancer_score', cs_tensor, persistent=True)

    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                filtered[k] = v
            else:
                skipped.append(k)
        else:
            filtered[k] = v
    for k in skipped:
        logging.warning(f"Skipping {k}: shape mismatch (ckpt={state_dict[k].shape} vs model={model_state[k].shape})")
    model.load_state_dict(filtered, strict=False)

    if recompute_cancer_score:
        if hasattr(model.codebook, 'cancer_score'):
            delattr(model.codebook, 'cancer_score')
        new_scores = model.codebook.get_cancer_scores()
        logging.info(f"[Quick Fix] Recomputed cancer_score from Phase 2 usage stats")
        logging.info(f"  New scores: min={new_scores.min():.4f} max={new_scores.max():.4f}")
        num_above_05 = (new_scores >= 0.5).sum().item()
        logging.info(f"  Codes with score >= 0.5: {num_above_05}")
        model.codebook.register_buffer('cancer_score', 
            model.codebook.get_cancer_scores().detach(), persistent=True)

    return model


def get_gt_mask_path(volume_path: str):
    base = os.path.basename(volume_path)
    name_noext = base.split('.nii')[0]
    label_dir = os.path.join(os.path.dirname(os.path.dirname(volume_path)), 'labels')
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


def load_gt_mask_hwz(gt_path: str):
    img = nib.load(gt_path)
    img = nib.as_closest_canonical(img)
    mask = img.get_fdata()
    if mask.ndim == 4:
        mask = mask[:, :, :, 0]
    tumor_label = _get_tumor_label(gt_path)
    if tumor_label is not None:
        return (mask == tumor_label).astype(np.uint8)
    return (mask > 0).astype(np.uint8)


def compute_patch_positive_labels(coords_zyx, gt_mask_hwz, patch_size):
    ph, pw, pd = patch_size[1], patch_size[2], patch_size[0]
    h, w, d = gt_mask_hwz.shape
    gt_patch_labels = np.zeros(len(coords_zyx), dtype=np.float32)
    for i, (z, y, x) in enumerate(coords_zyx):
        z_end = min(int(z) + pd, d)
        y_end = min(int(y) + ph, h)
        x_end = min(int(x) + pw, w)
        patch_gt = gt_mask_hwz[int(y):y_end, int(x):x_end, int(z):z_end]
        gt_patch_labels[i] = 1.0 if patch_gt.sum() > 0 else 0.0
    return gt_patch_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True,
                        choices=['gap', 'abmil', 'transmil', 'dsmil', 'v20'])
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--recompute_cancer_score', action='store_true',
                        help='Recompute cancer_score from usage stats instead of using stale Phase 1 buffer')
    parser.add_argument('--max_batches', type=int, default=0,
                        help='Max batches to evaluate (0=all, matches seg_val_max_batches in training)')
    parser.add_argument('--use_seg_head', action='store_true',
                        help='Use patch_seg_head output (sigmoid) instead of codebook cancer_scores for v20')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.method == 'v20':
        config_path = os.path.join(project_dir, 'configs/v20_mamba_first.yaml')
        ckpt_path = args.checkpoint or os.path.join(
            project_dir, 'volumetric_checkpoints/v20_best_seg_posdice0.328.pth')
    else:
        config_path = os.path.join(project_dir, 'configs/baseline_common.yaml')
        ckpt_path = args.checkpoint or os.path.join(
            project_dir, f'volumetric_checkpoints_baseline_{args.method}/best_model_{args.method}.pth')

    output_dir = args.output_dir or os.path.join(project_dir, 'eval_patch_level_results')
    os.makedirs(output_dir, exist_ok=True)

    config = Config.from_yaml(config_path)
    if args.method == 'v20':
        config.model['codebook_after_mamba'] = False
        config.model['seg_head']['use_multiscale'] = True
    patch_size = tuple(config.data['patch_size'])
    is_v20 = (args.method == 'v20')
    can_localize = (args.method != 'gap')

    if not can_localize:
        logging.info(f"GAP has no patch-level localization ability. Skipping.")
        with open(os.path.join(output_dir, 'gap_patch_eval.json'), 'w') as f:
            json.dump({'method': 'gap', 'note': 'GAP cannot produce patch-level scores'}, f)
        return

    logging.info(f"Evaluating {args.method} (patch-level) from {ckpt_path}")

    val_loader = create_volumetric_dataloader(
        json_path=config.data['test_json'],
        batch_size=config.data['batch_size'],
        patch_size=patch_size,
        stride=tuple(config.data['stride']),
        num_workers=config.data['num_workers'],
        shuffle=False,
        balanced_sampling=False,
        max_patches=config.data.get('max_patches', 64),
        use_ddp=False,
        min_hu=config.data['min_hu'],
        max_hu=config.data['max_hu']
    )

    if is_v20:
        model = build_v20_model_with_ckpt(config, ckpt_path,
                                           recompute_cancer_score=args.recompute_cancer_score)
    else:
        model = BaselineMILModel(config, aggregation_method=args.method)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()

    mini_bs = config.model['feature_extractor'].get('mini_batch_size', 8)

    all_patch_probs = []
    all_patch_gt = []
    all_patch_is_pos_bag = []

    per_volume_results = []
    total_patches = 0
    total_volumes_with_gt = 0
    skipped_no_gt = 0

    batch_count = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Patch-level eval"):
            if args.max_batches > 0 and batch_count >= args.max_batches:
                break
            batch_count += 1
            patches = batch['patches']
            coords = batch['coords'].to(device)
            labels = batch['labels'].to(device)
            masks = batch['masks'].to(device)

            use_amp = not args.use_seg_head
            with torch.amp.autocast('cuda', enabled=use_amp):
                model_out = model(patches, coords, masks, mini_batch_size=mini_bs)
                if is_v20:
                    logits, attention, third_out, patch_seg_logits = model_out
                else:
                    logits, attention, third_out = model_out
                    patch_seg_logits = None

            for b in range(labels.shape[0]):
                valid_n = int(masks[b].sum().item())
                if valid_n <= 0:
                    continue

                volume_path = batch['volume_paths'][b]
                vol_label = int(labels[b].item())
                gt_path = get_gt_mask_path(volume_path)
                gt_mask_hwz = load_gt_mask_hwz(gt_path) if gt_path else None

                if gt_mask_hwz is None:
                    skipped_no_gt += 1
                    continue

                total_volumes_with_gt += 1

                if is_v20:
                    if args.use_seg_head and patch_seg_logits is not None:
                        probs_b = torch.sigmoid(patch_seg_logits[b, :valid_n, 0]).cpu().numpy()
                    else:
                        cancer_scores = model.codebook.get_cancer_scores()
                        codes_b = third_out[b, :valid_n]
                        probs_b = cancer_scores[codes_b].float().cpu().numpy()
                else:
                    att_b = attention[b, :valid_n].cpu().numpy()
                    att_min, att_max = att_b.min(), att_b.max()
                    if att_max - att_min > 1e-8:
                        probs_b = (att_b - att_min) / (att_max - att_min)
                    else:
                        probs_b = np.zeros_like(att_b)

                coords_b = coords[b, :valid_n].detach().cpu().numpy()
                gt_labels_b = (compute_patch_positive_labels(coords_b, gt_mask_hwz, patch_size) > 0).astype(np.int8)

                all_patch_probs.append(probs_b)
                all_patch_gt.append(gt_labels_b)
                all_patch_is_pos_bag.append(vol_label == 1)
                total_patches += valid_n

                n_pos_patches = int(gt_labels_b.sum())
                n_neg_patches = valid_n - n_pos_patches

                per_volume_results.append({
                    'volume_path': volume_path,
                    'volume_name': os.path.basename(volume_path),
                    'volume_label': vol_label,
                    'n_patches': valid_n,
                    'n_pos_patches': n_pos_patches,
                    'n_neg_patches': n_neg_patches,
                    'patch_prob_mean': float(probs_b.mean()),
                    'patch_prob_max': float(probs_b.max()),
                    'pos_patch_prob_mean': float(probs_b[gt_labels_b == 1].mean()) if n_pos_patches > 0 else None,
                    'neg_patch_prob_mean': float(probs_b[gt_labels_b == 0].mean()) if n_neg_patches > 0 else None,
                })

    logging.info(f"Total volumes with GT: {total_volumes_with_gt}, skipped (no GT): {skipped_no_gt}")
    logging.info(f"Total patches evaluated: {total_patches}")

    all_probs = np.concatenate(all_patch_probs)
    all_gt = np.concatenate(all_patch_gt)
    all_is_pos_bag_expanded = []
    for probs_b, is_pos in zip(all_patch_probs, all_patch_is_pos_bag):
        all_is_pos_bag_expanded.extend([is_pos] * len(probs_b))
    all_is_pos_bag_expanded = np.array(all_is_pos_bag_expanded)

    pos_bag_mask = all_is_pos_bag_expanded
    neg_bag_mask = ~all_is_pos_bag_expanded

    def compute_metrics_at_thr(probs, gt, thr, bag_filter=None):
        if bag_filter is not None:
            probs = probs[bag_filter]
            gt = gt[bag_filter]
        preds = (probs >= thr).astype(np.int8)
        tp = int(((preds == 1) & (gt == 1)).sum())
        fp = int(((preds == 1) & (gt == 0)).sum())
        tn = int(((preds == 0) & (gt == 0)).sum())
        fn = int(((preds == 0) & (gt == 1)).sum())
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        return {
            'threshold': float(thr),
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'dice': dice,
            'total_patches': int(len(gt)),
            'total_pos_patches': int((gt == 1).sum()),
            'total_neg_patches': int((gt == 0).sum()),
        }

    thresholds = np.arange(0.05, 0.96, 0.01)
    best_dice = 0.0
    best_thr = 0.5
    for thr in thresholds:
        m = compute_metrics_at_thr(all_probs[pos_bag_mask], all_gt[pos_bag_mask], thr)
        if m['dice'] > best_dice:
            best_dice = m['dice']
            best_thr = float(thr)

    results_pos_only = compute_metrics_at_thr(all_probs[pos_bag_mask], all_gt[pos_bag_mask], best_thr)
    results_all = compute_metrics_at_thr(all_probs, all_gt, best_thr)
    results_neg_only = compute_metrics_at_thr(all_probs[neg_bag_mask], all_gt[neg_bag_mask], best_thr)

    results_at_05 = compute_metrics_at_thr(all_probs[pos_bag_mask], all_gt[pos_bag_mask], 0.5)
    results_all_05 = compute_metrics_at_thr(all_probs, all_gt, 0.5)

    neg_vol_fp_analysis = []
    for vr, probs_b, gt_b, is_pos in zip(per_volume_results, all_patch_probs, all_patch_gt, all_patch_is_pos_bag):
        if not is_pos:
            preds_b = (probs_b >= best_thr).astype(np.int8)
            fp_count = int(((preds_b == 1) & (gt_b == 0)).sum())
            neg_vol_fp_analysis.append({
                'volume_name': vr['volume_name'],
                'n_patches': vr['n_patches'],
                'fp_patches': fp_count,
                'fp_rate': fp_count / vr['n_patches'] if vr['n_patches'] > 0 else 0,
                'max_patch_prob': float(probs_b.max()),
                'mean_patch_prob': float(probs_b.mean()),
            })
    neg_vol_fp_analysis.sort(key=lambda x: x['fp_patches'], reverse=True)

    pos_vol_fn_analysis = []
    for vr, probs_b, gt_b, is_pos in zip(per_volume_results, all_patch_probs, all_patch_gt, all_patch_is_pos_bag):
        if is_pos and vr['n_pos_patches'] > 0:
            preds_b = (probs_b >= best_thr).astype(np.int8)
            fn_count = int(((preds_b == 0) & (gt_b == 1)).sum())
            tp_count = int(((preds_b == 1) & (gt_b == 1)).sum())
            fp_count = int(((preds_b == 1) & (gt_b == 0)).sum())
            pos_vol_fn_analysis.append({
                'volume_name': vr['volume_name'],
                'n_pos_patches': vr['n_pos_patches'],
                'n_neg_patches': vr['n_neg_patches'],
                'tp_patches': tp_count,
                'fp_patches': fp_count,
                'fn_patches': fn_count,
                'patch_sensitivity': tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0,
                'patch_precision': tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0,
            })
    pos_vol_fn_analysis.sort(key=lambda x: x['fp_patches'], reverse=True)

    output = {
        'method': args.method,
        'checkpoint': ckpt_path,
        'total_volumes_with_gt': total_volumes_with_gt,
        'total_patches': total_patches,
        'skipped_no_gt': skipped_no_gt,
        'best_threshold': best_thr,
        'pos_only_at_best_thr': results_pos_only,
        'all_volumes_at_best_thr': results_all,
        'neg_only_at_best_thr': results_neg_only,
        'pos_only_at_0.5': results_at_05,
        'all_volumes_at_0.5': results_all_05,
        'neg_vol_fp_analysis': neg_vol_fp_analysis[:20],
        'pos_vol_fp_analysis': pos_vol_fn_analysis[:20],
    }

    out_path = os.path.join(output_dir, f'{args.method}_patch_eval.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logging.info(f"\n=== {args.method.upper()} PATCH-LEVEL RESULTS ===")
    logging.info(f"Total volumes: {total_volumes_with_gt}, Total patches: {total_patches}")
    logging.info(f"Best threshold (PosDice): {best_thr:.2f}")
    logging.info(f"--- Positive volumes only (PosDice evaluation) ---")
    r = results_pos_only
    logging.info(f"  Dice={r['dice']:.4f} Sens={r['sensitivity']:.4f} Spec={r['specificity']:.4f} Prec={r['precision']:.4f}")
    logging.info(f"  TP={r['tp']} FP={r['fp']} TN={r['tn']} FN={r['fn']}")
    logging.info(f"  Pos patches={r['total_pos_patches']}, Neg patches={r['total_neg_patches']}")
    logging.info(f"--- All volumes ---")
    r = results_all
    logging.info(f"  Dice={r['dice']:.4f} Sens={r['sensitivity']:.4f} Spec={r['specificity']:.4f} Prec={r['precision']:.4f}")
    logging.info(f"  TP={r['tp']} FP={r['fp']} TN={r['tn']} FN={r['fn']}")
    logging.info(f"  Pos patches={r['total_pos_patches']}, Neg patches={r['total_neg_patches']}")
    logging.info(f"--- Negative volumes only (FP source) ---")
    r = results_neg_only
    logging.info(f"  FP patches={r['fp']} / {r['total_neg_patches']} neg patches = FP rate {r['fp']/(r['total_neg_patches']+1e-9):.4f}")
    if neg_vol_fp_analysis:
        logging.info(f"--- Top FP negative volumes (at thr={best_thr:.2f}) ---")
        for v in neg_vol_fp_analysis[:10]:
            logging.info(f"  {v['volume_name']}: {v['fp_patches']}/{v['n_patches']} FP patches, max_prob={v['max_patch_prob']:.4f}")
    logging.info(f"Results saved to {out_path}")


if __name__ == '__main__':
    main()
