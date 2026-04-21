import argparse
import logging
import os
import sys
from datetime import timedelta
from functools import lru_cache

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from src.training import Config, attach_log_file_handler, configure_runtime_paths, resolve_relative_path
from src.models.feature_extractor_3d import VolumetricFeatureExtractor
from src.models.spatial_scanner_3d import (
    ZOrderSpatialScanner,
    reorder_sequence,
    restore_sequence_order,
)
from src.models.video_mamba import VideoMamba3D
from src.models.codebook import VectorQuantizer3D
from src.models.mil_head import AttentionMILHead
from src.data.volumetric_loader import create_volumetric_dataloader


class FullSupMILCodebookModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        feature_dim = config.model['codebook']['embedding_dim']

        self.feature_extractor = VolumetricFeatureExtractor(
            arch=config.model['feature_extractor']['arch'],
            spatial_dims=config.model['feature_extractor']['spatial_dims'],
            n_input_channels=config.model['feature_extractor']['n_input_channels'],
            pretrained=config.model['feature_extractor']['pretrained'],
            frozen=config.model['feature_extractor']['frozen']
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
        sorted_masks = reorder_sequence(masks, sort_perm)
        context = self.mamba(sorted_features, mask=sorted_masks)
        context_orig = restore_sequence_order(context, sort_perm)
        seg_logits = self.seg_head(context_orig).squeeze(-1)
        mil_logits, attention = self.mil_head(context_orig, mask=masks)
        return seg_logits, mil_logits, attention, vq_loss, code_logits


def setup_ddp():
    dist.init_process_group(backend='nccl', timeout=timedelta(hours=2))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    import warnings
    warnings.filterwarnings('ignore', message='.*using the device under current context.*')
    return rank, world_size


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


@lru_cache(maxsize=4)
def _cached_load_gt_mask(volume_path: str):
    gt_path = get_gt_mask_path(volume_path)
    if gt_path is None:
        return None
    return load_gt_mask_hwz(gt_path)


def compute_patch_labels(coords_zyx, gt_mask_hwz, patch_size, threshold=0):
    ph, pw, pd = patch_size[1], patch_size[2], patch_size[0]
    h, w, d = gt_mask_hwz.shape
    labels = np.zeros(len(coords_zyx), dtype=np.float32)
    for i, (z, y, x) in enumerate(coords_zyx):
        z_end = min(int(z) + pd, d)
        y_end = min(int(y) + ph, h)
        x_end = min(int(x) + pw, w)
        patch_gt = gt_mask_hwz[int(y):y_end, int(x):x_end, int(z):z_end]
        labels[i] = 1.0 if patch_gt.sum() >= threshold else 0.0
    return labels


def resolve_lr_for_epoch(config, epoch_idx: int) -> float:
    current_epoch = epoch_idx + 1
    lr_schedule = config.training.get('lr_schedule', [])
    for stage in lr_schedule:
        start_epoch = int(stage.get('start_epoch', 1))
        end_epoch = int(stage.get('end_epoch', start_epoch))
        if start_epoch <= current_epoch <= end_epoch:
            return float(stage.get('lr', config.training['lr']))
    return float(config.training['lr'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to fullsup checkpoint for partial weight init')
    parser.add_argument('--output_root', type=str, default=None, help='Artifact root directory; defaults to a sibling mamba_artifacts directory')
    parser.add_argument('--run_name', type=str, default=None, help='Run name under the artifact root; defaults to the config filename stem')
    args = parser.parse_args()

    rank, world_size = setup_ddp()
    device = torch.device(f'cuda:{rank}')

    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    config = Config.from_yaml(args.config)
    runtime_paths = configure_runtime_paths(
        config,
        script_path=__file__,
        config_path=args.config,
        output_root=args.output_root,
        run_name=args.run_name,
    )
    if rank == 0:
        log_file_path = attach_log_file_handler(runtime_paths['log_dir'])
        logging.info(f"Training log file: {log_file_path}")

    scaler = torch.amp.GradScaler('cuda', enabled=config.training.get('mixed_precision', False))
    log_dir = config.logging['log_dir']
    save_dir = config.checkpoint['save_dir']

    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        logging.info("=" * 60)
        logging.info("FULLY SUPERVISED + MIL + CODEBOOK + MAMBA")
        logging.info("=" * 60)
        logging.info(f"Run name: {runtime_paths['run_name']}")
        logging.info(f"Artifact root: {runtime_paths['output_root']}")
        logging.info(f"Checkpoint dir: {runtime_paths['checkpoint_dir']}")
        logging.info(f"Log dir: {runtime_paths['log_dir']}")

    train_loader = create_volumetric_dataloader(
        json_path=config.data['train_json'],
        batch_size=config.data['batch_size'],
        patch_size=tuple(config.data['patch_size']),
        stride=tuple(config.data['stride']),
        num_workers=config.data['num_workers'],
        shuffle=True,
        balanced_sampling=True,
        max_patches=config.data.get('max_patches', 64),
        use_ddp=True,
        use_phase_aware=False,
        min_hu=config.data['min_hu'],
        max_hu=config.data['max_hu'],
        adaptive_norm=config.data.get('adaptive_norm', True),
        augment=True,
    )

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

    if args.pretrained and os.path.isfile(args.pretrained):
        if rank == 0:
            logging.info(f"Loading pretrained weights from {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device)
        ckpt_state = ckpt['model_state_dict']
        model_state = model.state_dict()
        loaded, skipped = [], []
        for k, v in ckpt_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                loaded.append(k)
            else:
                skipped.append(k)
        model.load_state_dict(model_state)
        if rank == 0:
            logging.info(f"Pretrained: loaded {len(loaded)} params, skipped {len(skipped)}")
            new_keys = [k for k in model_state if k not in ckpt_state]
            logging.info(f"New modules (random init): {len(new_keys)} params")
            prefixes = sorted(set(k.split('.')[0] for k in new_keys))
            logging.info(f"  Modules: {prefixes}")

    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training['lr'],
        weight_decay=config.training['weight_decay'],
    )

    patch_size = tuple(config.data['patch_size'])
    mini_batch_size = config.model['feature_extractor'].get('mini_batch_size', 8)
    use_amp = config.training.get('mixed_precision', False)
    val_freq = int(config.training.get('val_freq', 6))
    val_start_epoch = int(config.training.get('val_start_epoch', 6))
    epochs = int(config.training.get('epochs', 100))
    save_freq = int(config.checkpoint.get('save_freq', 6))

    seg_pos_weight_val = float(config.loss.get('pos_weight', 10.0))
    seg_pos_weight = torch.tensor([seg_pos_weight_val], device=device)
    lambda_cls = float(config.loss.get('lambda_cls', 1.0))
    lambda_vq = float(config.loss.get('lambda_vq', 0.1))
    lambda_code = float(config.loss.get('lambda_code', 0.5))
    cls_weights = config.loss.get('cls_class_weights', [2.0, 1.0])
    cls_weight_tensor = torch.tensor(cls_weights, device=device, dtype=torch.float32)
    label_smoothing = float(config.loss.get('label_smoothing', 0.0))

    voxel_thr = int(config.data.get('voxel_tolerance_threshold', 33))
    val_voxel_thresholds = config.data.get('val_voxel_thresholds', [1, 16, 33, 164, 328])

    best_seg_score = -1.0
    best_auc = 0.0
    start_epoch = 0

    resolved_resume = resolve_relative_path(args.resume, runtime_paths['checkpoint_dir'])
    if resolved_resume and os.path.isfile(resolved_resume):
        if rank == 0:
            logging.info(f"Resuming from {resolved_resume}")
        ckpt = torch.load(resolved_resume, map_location=device)
        missing, unexpected = model.module.load_state_dict(
            ckpt['model_state_dict'], strict=False
        )
        if rank == 0 and missing:
            logging.info(f"Resume missing keys: {missing}")
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except (ValueError, RuntimeError):
                if rank == 0:
                    logging.warning("Optimizer state mismatch, using fresh optimizer")
        start_epoch = ckpt.get('epoch', -1) + 1
        best_seg_score = ckpt.get('best_pos_dice', -1.0)
        best_auc = ckpt.get('best_auc', 0.0)
        if rank == 0:
            logging.info(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        need_val = ((epoch + 1) >= val_start_epoch and (epoch + 1) % val_freq == 0)
        do_val = (rank == 0 and need_val)

        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        current_lr = resolve_lr_for_epoch(config, epoch)
        for pg in optimizer.param_groups:
            pg['lr'] = current_lr

        model.train()
        total_loss = 0.0
        total_seg_loss = 0.0
        total_cls_loss = 0.0
        total_vq_loss = 0.0
        total_code_loss = 0.0
        num_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}") if rank == 0 else train_loader

        for batch_idx, batch in enumerate(pbar):
            patches = batch['patches']
            coords = batch['coords'].to(device)
            labels = batch['labels'].to(device)
            masks = batch['masks'].to(device)
            B = labels.shape[0]
            N = masks.shape[1]

            gt_patch_labels = torch.zeros(B, N, device=device)
            for b in range(B):
                volume_path = batch['volume_paths'][b]
                gt_mask_hwz = _cached_load_gt_mask(volume_path)
                if gt_mask_hwz is not None:
                    valid_n = int(masks[b].sum().item())
                    coords_b = coords[b, :valid_n].cpu().numpy()
                    pl = compute_patch_labels(coords_b, gt_mask_hwz, patch_size, threshold=voxel_thr)
                    gt_patch_labels[b, :valid_n] = torch.tensor(
                        np.array(pl, dtype=np.float32)).to(device)

            optimizer.zero_grad(set_to_none=True)
            try:
                with torch.amp.autocast('cuda', enabled=use_amp):
                    seg_logits, mil_logits, attention, vq_loss, code_logits = model(
                        patches, coords, masks, mini_batch_size=mini_batch_size
                    )

                    valid_mask = masks.bool()
                    logits_flat = seg_logits[valid_mask]
                    labels_flat = gt_patch_labels[valid_mask]
                    seg_loss = F.binary_cross_entropy_with_logits(
                        logits_flat, labels_flat, pos_weight=seg_pos_weight
                    )

                    code_logits_flat = code_logits[valid_mask]
                    code_cls_loss = F.binary_cross_entropy_with_logits(
                        code_logits_flat, labels_flat, pos_weight=seg_pos_weight
                    )

                    cls_target = labels.long()
                    cls_loss = F.cross_entropy(
                        mil_logits, cls_target,
                        weight=cls_weight_tensor,
                        label_smoothing=label_smoothing,
                    )

                    loss = (seg_loss + lambda_cls * cls_loss
                            + lambda_vq * vq_loss + lambda_code * code_cls_loss)

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                total_loss += loss.item()
                total_seg_loss += seg_loss.item()
                total_cls_loss += cls_loss.item()
                total_vq_loss += vq_loss.item()
                total_code_loss += code_cls_loss.item()
                num_batches += 1

                if rank == 0 and batch_idx % 10 == 0:
                    n_pos = int(labels_flat.sum().item())
                    n_total = len(labels_flat)
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'seg': f"{seg_loss.item():.4f}",
                        'cls': f"{cls_loss.item():.4f}",
                        'code': f"{code_cls_loss.item():.4f}",
                        'vq': f"{vq_loss.item():.4f}",
                        'pos': f"{n_pos}/{n_total}",
                    })

            except torch.cuda.OutOfMemoryError:
                if rank == 0:
                    logging.warning(f"OOM at batch {batch_idx}, skipping")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue

        if rank == 0 and num_batches > 0:
            logging.info(
                f"Train Loss: {total_loss/num_batches:.4f} "
                f"(seg={total_seg_loss/num_batches:.4f}, "
                f"cls={total_cls_loss/num_batches:.4f}, "
                f"code={total_code_loss/num_batches:.4f}, "
                f"vq={total_vq_loss/num_batches:.4f})"
            )

        if rank == 0 and ((epoch + 1) % save_freq == 0 or do_val):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_pos_dice': best_seg_score,
                'best_auc': best_auc,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

        if need_val:
            dist.barrier()

        if do_val:
            model.eval()
            all_seg_probs = []
            all_seg_coords = []
            all_seg_gt_masks = []
            all_seg_is_pos = []
            all_vol_preds_seg = []
            all_vol_preds_mil = []
            all_vol_labels = []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation", leave=False):
                    patches = batch['patches']
                    coords_val = batch['coords'].to(device)
                    labels_val = batch['labels'].to(device)
                    masks_val = batch['masks'].to(device)

                    with torch.amp.autocast('cuda', enabled=use_amp):
                        seg_logits, mil_logits, attention, _, _ = model.module(
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

            from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
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
                    (compute_patch_labels(c, m, patch_size, threshold=voxel_thr) > 0).astype(np.int8)
                    for c, m in zip(all_seg_coords, all_seg_gt_masks)
                ]

                best_pos_dice_this_val = -1.0
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

                    logging.info(
                        f"Seg pred_thr={pred_thr:.2f}: "
                        f"PosDice={pos_dice:.4f}, Dice={dice:.4f}, "
                        f"Prec={precision:.4f}, Rec={recall:.4f}"
                    )

                    if pos_dice > best_pos_dice_this_val:
                        best_pos_dice_this_val = pos_dice
                        best_threshold = pred_thr

                logging.info(
                    f"Best threshold={best_threshold:.2f}, "
                    f"PosDice={best_pos_dice_this_val:.4f}"
                )

                if best_pos_dice_this_val > best_seg_score:
                    best_seg_score = best_pos_dice_this_val
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'best_pos_dice': best_seg_score,
                        'best_threshold': best_threshold,
                    }, os.path.join(save_dir, 'best_seg.pth'))
                    logging.info(f"  -> New best PosDice: {best_seg_score:.4f} @ thr={best_threshold:.2f}")

                combined_auc = max(seg_auc, mil_auc)
                if combined_auc > best_auc:
                    best_auc = combined_auc
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'best_auc': best_auc,
                    }, os.path.join(save_dir, 'best_model.pth'))
                    logging.info(f"  -> New best AUC: {best_auc:.4f}")

        dist.barrier()

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
