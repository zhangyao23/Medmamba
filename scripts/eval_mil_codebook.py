import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
from tqdm import tqdm
from functools import lru_cache
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from src.training import Config
from src.data.volumetric_loader import create_volumetric_dataloader
from src.models.feature_extractor_3d import VolumetricFeatureExtractor
from src.models.spatial_scanner_3d import ZOrderSpatialScanner
from src.models.video_mamba import VideoMamba3D
from src.models.codebook import VectorQuantizer3D
from src.models.mil_head import AttentionMILHead


class FullSupMILCodebookModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        fe_cfg = config.model['feature_extractor']
        feature_dim = fe_cfg.get('feature_dim', 512)
        self.feature_extractor = VolumetricFeatureExtractor(
            arch=fe_cfg['arch'],
            spatial_dims=fe_cfg.get('spatial_dims', 3),
            n_input_channels=fe_cfg.get('n_input_channels', 1),
            pretrained=fe_cfg.get('pretrained', False),
        )

        cb_cfg = config.model['codebook']
        self.codebook = VectorQuantizer3D(
            num_embeddings=cb_cfg.get('num_embeddings', 64),
            embedding_dim=cb_cfg['embedding_dim'],
            commitment_cost=cb_cfg.get('commitment_cost', 0.25),
            use_ema=cb_cfg.get('use_ema', True),
            ema_decay=cb_cfg.get('ema_decay', 0.99),
        )

        self.spatial_scanner = ZOrderSpatialScanner()

        mamba_cfg = config.model.get('mamba', {})
        self.mamba = VideoMamba3D(
            d_model=mamba_cfg.get('d_model', feature_dim),
            d_state=mamba_cfg.get('d_state', 16),
            d_conv=mamba_cfg.get('d_conv', 4),
            expand=mamba_cfg.get('expand', 2),
            num_layers=mamba_cfg.get('num_layers', 4),
            bidirectional=mamba_cfg.get('bidirectional', True),
        )

        mil_cfg = config.model.get('mil', {})
        self.mil_head = AttentionMILHead(
            input_dim=feature_dim,
            hidden_dim=mil_cfg.get('hidden_dim', 256),
            num_classes=mil_cfg.get('num_classes', 2),
        )

        self.seg_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.code_classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, patches, coords, masks=None, mini_batch_size=8):
        features = self.feature_extractor(patches, mini_batch_size=mini_batch_size)
        quantized, code_indices, vq_loss = self.codebook(features)
        code_logits = self.code_classifier(quantized).squeeze(-1)
        sorted_features, sorted_coords, sort_perm = self.spatial_scanner(quantized, coords)
        context = self.mamba(sorted_features, mask=masks)
        inv_perm = sort_perm.argsort(dim=1)
        B, N, D = context.shape
        context_orig = torch.gather(
            context, 1, inv_perm.unsqueeze(-1).expand(B, N, D)
        )
        seg_logits = self.seg_head(context_orig).squeeze(-1)
        mil_logits, attention = self.mil_head(context_orig, mask=masks)
        return seg_logits, mil_logits, attention, vq_loss, code_logits


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


def load_gt_mask_hwz(gt_path: str):
    img = nib.load(gt_path)
    img = nib.as_closest_canonical(img)
    mask = img.get_fdata()
    if mask.ndim == 4:
        mask = mask[:, :, :, 0]
    tumor_label = _get_tumor_label(gt_path)
    if tumor_label is not None:
        return (mask == tumor_label).astype(np.float32)
    return (mask > 0).astype(np.float32)


@lru_cache(maxsize=4)
def _cached_load_gt_mask(volume_path: str):
    gt_path = get_gt_mask_path(volume_path)
    if gt_path is None:
        return None
    return load_gt_mask_hwz(gt_path)


def compute_patch_labels(coords_zyx, gt_mask_hwz, patch_size, threshold=0):
    pd, ph, pw = patch_size
    H, W, Z = gt_mask_hwz.shape
    labels = []
    for c in coords_zyx:
        z, y, x = int(c[0]), int(c[1]), int(c[2])
        z0, z1 = max(0, z), min(Z, z + pd)
        y0, y1 = max(0, y), min(H, y + ph)
        x0, x1 = max(0, x), min(W, x + pw)
        patch = gt_mask_hwz[y0:y1, x0:x1, z0:z1]
        labels.append(1 if patch.sum() > threshold else 0)
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    device = torch.device(f'cuda:{args.gpu}')
    config = Config.from_yaml(args.config)

    val_loader = create_volumetric_dataloader(
        json_path=config.data['test_json'],
        batch_size=config.data['batch_size'],
        patch_size=tuple(config.data['patch_size']),
        stride=tuple(config.data['stride']),
        num_workers=config.data['num_workers'],
        shuffle=False,
        balanced_sampling=False,
        max_patches=config.data.get('max_patches', 64),
        use_ddp=False,
        min_hu=config.data['min_hu'],
        max_hu=config.data['max_hu'],
        adaptive_norm=config.data.get('adaptive_norm', True),
    )

    model = FullSupMILCodebookModel(config).to(device)

    logging.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    logging.info(f"Checkpoint info: epoch={ckpt.get('epoch')}, "
                 f"best_pos_dice={ckpt.get('best_pos_dice')}, "
                 f"best_threshold={ckpt.get('best_threshold')}")

    patch_size = tuple(config.data['patch_size'])
    mini_batch_size = config.model['feature_extractor'].get('mini_batch_size', 8)
    use_amp = config.training.get('mixed_precision', False)
    voxel_thr = int(config.data.get('voxel_tolerance_threshold', 33))
    val_voxel_thresholds = config.data.get('val_voxel_thresholds', [1, 16, 33, 164, 328])

    model.eval()
    all_seg_probs = []
    all_seg_coords = []
    all_seg_gt_masks = []
    all_seg_is_pos = []
    all_vol_preds_seg = []
    all_vol_preds_mil = []
    all_vol_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            patches = batch['patches']
            coords_val = batch['coords'].to(device)
            labels_val = batch['labels'].to(device)
            masks_val = batch['masks'].to(device)

            with torch.amp.autocast('cuda', enabled=use_amp):
                seg_logits, mil_logits, attention, _, _ = model(
                    patches, coords_val, masks_val, mini_batch_size=mini_batch_size
                )

            seg_probs = torch.sigmoid(seg_logits)
            mil_probs = torch.softmax(mil_logits, dim=1)[:, 1]

            for b in range(labels_val.shape[0]):
                valid_n = int(masks_val[b].sum().item())
                if valid_n <= 0:
                    continue

                probs_b_pt = seg_probs[b, :valid_n]
                vol_pred_seg = float(probs_b_pt.max().item())
                vol_pred_mil = float(mil_probs[b].item())
                probs_b = probs_b_pt.cpu().numpy()
                all_vol_preds_seg.append(vol_pred_seg)
                all_vol_preds_mil.append(vol_pred_mil)
                all_vol_labels.append(int(labels_val[b].item()))

                volume_path = batch['volume_paths'][b]
                gt_mask_hwz = _cached_load_gt_mask(volume_path)
                if gt_mask_hwz is None:
                    continue

                coords_b = coords_val[b, :valid_n].cpu().numpy()
                all_seg_probs.append(probs_b)
                all_seg_coords.append(coords_b)
                all_seg_gt_masks.append(gt_mask_hwz)
                all_seg_is_pos.append(int(labels_val[b].item()) == 1)

    vol_labels_np = np.array(all_vol_labels)

    vol_preds_seg_np = np.array(all_vol_preds_seg)
    seg_auc = roc_auc_score(vol_labels_np, vol_preds_seg_np) if len(np.unique(vol_labels_np)) > 1 else 0.0
    seg_cls = (vol_preds_seg_np > 0.5).astype(int)
    logging.info(f"Vol Cls (seg max): AUC={seg_auc:.4f}, "
                 f"Acc={accuracy_score(vol_labels_np, seg_cls):.4f}, "
                 f"F1={f1_score(vol_labels_np, seg_cls):.4f}")

    vol_preds_mil_np = np.array(all_vol_preds_mil)
    mil_auc = roc_auc_score(vol_labels_np, vol_preds_mil_np) if len(np.unique(vol_labels_np)) > 1 else 0.0
    mil_cls = (vol_preds_mil_np > 0.5).astype(int)
    logging.info(f"Vol Cls (MIL):     AUC={mil_auc:.4f}, "
                 f"Acc={accuracy_score(vol_labels_np, mil_cls):.4f}, "
                 f"F1={f1_score(vol_labels_np, mil_cls):.4f}")

    if all_seg_probs:
        all_seg_gt = [
            (np.array(compute_patch_labels(c, m, patch_size, threshold=vt)) > 0).astype(np.int8)
            for c, m, vt in [(c, m, voxel_thr) for c, m in zip(all_seg_coords, all_seg_gt_masks)]
        ]

        all_probs_flat = np.concatenate(all_seg_probs)
        all_gt_flat = np.concatenate(all_seg_gt)

        pos_mask = np.array(all_seg_is_pos)
        pos_probs_list = [p for p, is_p in zip(all_seg_probs, pos_mask) if is_p]
        pos_gt_list = [g for g, is_p in zip(all_seg_gt, pos_mask) if is_p]
        pos_probs_flat = np.concatenate(pos_probs_list) if pos_probs_list else np.array([])
        pos_gt_flat = np.concatenate(pos_gt_list) if pos_gt_list else np.array([])

        from sklearn.metrics import roc_auc_score as auc_fn, average_precision_score

        logging.info("=" * 60)
        logging.info("Patch-level AUC (continuous probabilities)")
        logging.info("=" * 60)
        if len(np.unique(all_gt_flat)) > 1:
            patch_auc_all = auc_fn(all_gt_flat, all_probs_flat)
            patch_ap_all = average_precision_score(all_gt_flat, all_probs_flat)
            logging.info(f"All volumes:     AUC={patch_auc_all:.4f}, AP={patch_ap_all:.4f}")
        else:
            logging.info("All volumes:     skipped (single class)")
        if len(pos_gt_flat) > 0 and len(np.unique(pos_gt_flat)) > 1:
            patch_auc_pos = auc_fn(pos_gt_flat, pos_probs_flat)
            patch_ap_pos = average_precision_score(pos_gt_flat, pos_probs_flat)
            logging.info(f"Pos volumes only: AUC={patch_auc_pos:.4f}, AP={patch_ap_pos:.4f}")
        else:
            logging.info("Pos volumes only: skipped (single class)")

        logging.info("=" * 60)
        logging.info("Patch-level segmentation metrics (sweep thresholds)")
        logging.info("=" * 60)

        best_pos_dice = -1.0
        best_threshold = 0.5
        for pred_thr in np.arange(0.05, 0.96, 0.05):
            tp_pos, pred_pos, gt_pos = 0, 0, 0
            tp_all, pred_all, gt_all = 0, 0, 0
            for probs_b, gt_b, is_pos in zip(all_seg_probs, all_seg_gt, all_seg_is_pos):
                pred_b = (probs_b > pred_thr).astype(np.int8)
                tp = int((pred_b * gt_b).sum())
                tp_all += tp
                pred_all += int(pred_b.sum())
                gt_all += int(gt_b.sum())
                if is_pos:
                    tp_pos += tp
                    pred_pos += int(pred_b.sum())
                    gt_pos += int(gt_b.sum())

            dice = 2.0 * tp_all / (pred_all + gt_all + 1e-8)
            pos_dice = 2.0 * tp_pos / (pred_pos + gt_pos + 1e-8)
            precision = tp_pos / (pred_pos + 1e-8)
            recall = tp_pos / (gt_pos + 1e-8)
            f1 = 2.0 * precision * recall / (precision + recall + 1e-8)

            logging.info(
                f"pred_thr={pred_thr:.2f}: "
                f"PosDice={pos_dice:.4f}, F1={f1:.4f}, "
                f"Prec={precision:.4f}, Rec={recall:.4f}"
            )

            if pos_dice > best_pos_dice:
                best_pos_dice = pos_dice
                best_threshold = pred_thr

        logging.info("=" * 60)
        logging.info(
            f"Best: threshold={best_threshold:.2f}, PosDice={best_pos_dice:.4f}"
        )
        logging.info("=" * 60)


if __name__ == '__main__':
    main()
