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
from src.models.spatial_scanner_3d import ZOrderSpatialScanner
from src.models.video_mamba import VideoMamba3D
from src.data.volumetric_loader import create_volumetric_dataloader
from src.utils.metrics import evaluate_bag_level


class FullSupSegModel(nn.Module):
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
        mamba_cfg = config.model.get('mamba', {})
        self.spatial_scanner = ZOrderSpatialScanner()
        self.mamba = VideoMamba3D(
            d_model=mamba_cfg.get('d_model', feature_dim),
            d_state=mamba_cfg.get('d_state', 16),
            d_conv=mamba_cfg.get('d_conv', 4),
            expand=mamba_cfg.get('expand', 2),
            num_layers=mamba_cfg.get('num_layers', 4),
            bidirectional=mamba_cfg.get('bidirectional', True),
        )
        self.seg_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, patches, coords, masks=None, mini_batch_size=8):
        features = self.feature_extractor(patches, mini_batch_size=mini_batch_size)
        sorted_features, sorted_coords, sort_perm = self.spatial_scanner(features, coords)
        context = self.mamba(sorted_features, mask=masks)
        inv_perm = sort_perm.argsort(dim=1)
        B, N, D = context.shape
        context_orig = torch.gather(
            context, 1, inv_perm.unsqueeze(-1).expand(B, N, D)
        )
        seg_logits = self.seg_head(context_orig).squeeze(-1)
        return seg_logits


def setup_ddp():
    dist.init_process_group(backend='nccl', timeout=timedelta(hours=2))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    
    # Suppress the barrier device warning
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
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--epoch_limit', type=int, default=None, help='Stop after this epoch (exclusive), e.g. 24 to run only epoch 24')
    parser.add_argument('--validate_only', action='store_true', help='Load checkpoint, run one validation, then exit (for testing validation+barrier)')
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
        logging.info("FULLY SUPERVISED PATCH SEGMENTATION (Upper Bound)")
        logging.info("=" * 60)
        logging.info(f"Patch size: {config.data['patch_size']}")
        logging.info(f"Batch size per GPU: {config.data['batch_size']}")
        logging.info(f"Max patches: {config.data.get('max_patches', 64)}")
        logging.info(f"Voxel tolerance threshold: {config.data.get('voxel_tolerance_threshold', 0)}")
        logging.info(f"Loss pos_weight: {config.loss.get('pos_weight', 10.0)}")
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

    if rank == 0:
        logging.info("GT masks will be loaded on-demand per batch (LRU cache size=512)")

    model = FullSupSegModel(config).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

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
    end_epoch = args.epoch_limit if args.epoch_limit is not None else epochs
    best_seg_score = -1.0
    best_auc = 0.0
    pos_weight_val = float(config.loss.get('pos_weight', 10.0))
    pos_weight = torch.tensor([pos_weight_val], device=device)
    voxel_thr = int(config.data.get('voxel_tolerance_threshold', 1))
    val_pred_thresholds = [round(t * 0.05, 2) for t in range(1, 20)]

    start_epoch = 0
    resolved_resume = resolve_relative_path(args.resume, runtime_paths['checkpoint_dir'])
    if resolved_resume and os.path.isfile(resolved_resume):
        if rank == 0:
            logging.info(f"Resuming from checkpoint {resolved_resume}")
        checkpoint = torch.load(resolved_resume, map_location='cpu')
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        del checkpoint
        import gc; gc.collect()
        torch.cuda.empty_cache()
        if rank == 0:
            logging.info(f"Successfully resumed from epoch {start_epoch}")

    if args.validate_only:
        if not resolved_resume or start_epoch == 0:
            if rank == 0:
                logging.error("--validate_only requires --resume with a valid checkpoint")
            dist.destroy_process_group()
            return
        end_epoch = start_epoch
        start_epoch = start_epoch - 1

    for epoch in range(start_epoch, end_epoch):
        need_val = ((epoch + 1) >= val_start_epoch and (epoch + 1) % val_freq == 0)
        do_val = (rank == 0 and need_val)
        if args.validate_only:
            if rank == 0:
                logging.info(f"Validate only (epoch {epoch+1}), skipping training")
        if not args.validate_only and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        if not args.validate_only:
            current_lr = resolve_lr_for_epoch(config, epoch)
            for pg in optimizer.param_groups:
                pg['lr'] = current_lr

            model.train()
            total_loss = 0.0
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
                        pl_array = np.array(pl, dtype=np.float32)
                        gt_patch_labels[b, :valid_n] = torch.tensor(pl_array).to(device)

                optimizer.zero_grad(set_to_none=True)
                oom_flag = torch.zeros(1, device=device)
                try:
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        seg_logits = model(patches, coords, masks, mini_batch_size=mini_batch_size)

                        valid_mask = masks.bool()
                        logits_flat = seg_logits[valid_mask]
                        labels_flat = gt_patch_labels[valid_mask]

                        loss = F.binary_cross_entropy_with_logits(
                            logits_flat, labels_flat, pos_weight=pos_weight
                        )

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
                    num_batches += 1

                    if rank == 0 and batch_idx % 10 == 0:
                        n_pos = int(labels_flat.sum().item())
                        n_total = len(labels_flat)
                        pbar.set_postfix({
                            'loss': f"{loss.item():.4f}",
                            'pos': f"{n_pos}/{n_total}",
                        })

                except torch.cuda.OutOfMemoryError:
                    oom_flag.fill_(1.0)
                    for _v in ('seg_logits', 'loss', 'logits_flat', 'labels_flat', 'valid_mask'):
                        if _v in locals():
                            del locals()[_v]
                    import gc; gc.collect()
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()

                dist.all_reduce(oom_flag, op=dist.ReduceOp.MAX)
                if oom_flag.item() > 0:
                    if rank == 0:
                        logging.warning(f"OOM at batch {batch_idx}, all ranks skip")
                    model.zero_grad(set_to_none=True)
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    continue

            avg_loss = total_loss / max(num_batches, 1)
            if rank == 0:
                logging.info(f"Train Loss: {avg_loss:.4f}")

            if rank == 0 and ((epoch + 1) % 10 == 0 or do_val):
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

        if need_val:
            dist.barrier()

        if do_val:
            model.eval()
            all_seg_probs = []
            all_seg_coords = []
            all_seg_gt_masks = []
            all_seg_is_pos = []
            all_vol_preds = []
            all_vol_labels = []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation", leave=False):
                    patches = batch['patches']
                    coords_val = batch['coords'].to(device)
                    labels_val = batch['labels'].to(device)
                    masks_val = batch['masks'].to(device)

                    with torch.amp.autocast('cuda', enabled=use_amp):
                        seg_logits = model.module(patches, coords_val, masks_val, mini_batch_size=mini_batch_size)

                    seg_probs = torch.sigmoid(seg_logits)

                    for b in range(labels_val.shape[0]):
                        valid_n = int(masks_val[b].sum().item())
                        if valid_n <= 0:
                            continue

                        probs_b_pt = seg_probs[b, :valid_n]
                        vol_pred_score = float(probs_b_pt.max().item())
                        probs_b = probs_b_pt.cpu().numpy()
                        all_vol_preds.append(vol_pred_score)
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

            vol_preds_np = np.array(all_vol_preds)
            vol_labels_np = np.array(all_vol_labels)
            from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
            vol_auc = roc_auc_score(vol_labels_np, vol_preds_np) if len(np.unique(vol_labels_np)) > 1 else 0.0
            vol_cls = (vol_preds_np > 0.5).astype(int)
            vol_acc = accuracy_score(vol_labels_np, vol_cls)
            vol_f1 = f1_score(vol_labels_np, vol_cls)
            logging.info(f"Vol Cls (from seg max): AUC={vol_auc:.4f}, Acc={vol_acc:.4f}, F1={vol_f1:.4f}")

            if all_seg_probs:
                from sklearn.metrics import roc_auc_score as patch_auc_fn, average_precision_score

                all_seg_gt = [
                    (compute_patch_labels(c, m, patch_size, threshold=voxel_thr) > 0).astype(np.int8)
                    for c, m in zip(all_seg_coords, all_seg_gt_masks)
                ]

                all_probs_flat = np.concatenate(all_seg_probs)
                all_gt_flat = np.concatenate(all_seg_gt)
                if len(np.unique(all_gt_flat)) > 1:
                    patch_auc = patch_auc_fn(all_gt_flat, all_probs_flat)
                    patch_ap = average_precision_score(all_gt_flat, all_probs_flat)
                else:
                    patch_auc, patch_ap = 0.0, 0.0
                logging.info(f"Patch-level (continuous): AUC={patch_auc:.4f}, AP={patch_ap:.4f}")

                best_pos_dice_this_epoch = -1.0
                best_thr_this_epoch = 0.5
                for pred_thr in val_pred_thresholds:
                    tp_pos, pred_pos, gt_pos = 0, 0, 0
                    for probs_b, gt_b, is_pos in zip(all_seg_probs, all_seg_gt, all_seg_is_pos):
                        pred_b = (probs_b > pred_thr).astype(np.int8)
                        tp = int((pred_b * gt_b).sum())
                        if is_pos:
                            tp_pos += tp
                            pred_pos += int(pred_b.sum())
                            gt_pos += int(gt_b.sum())

                    pos_dice = 2.0 * tp_pos / (pred_pos + gt_pos + 1e-8)
                    precision = tp_pos / (pred_pos + 1e-8)
                    recall = tp_pos / (gt_pos + 1e-8)

                    logging.info(
                        f"  pred_thr={pred_thr:.2f}: "
                        f"PosDice={pos_dice:.4f}, "
                        f"Precision={precision:.4f}, Recall={recall:.4f}"
                    )

                    if pos_dice > best_pos_dice_this_epoch:
                        best_pos_dice_this_epoch = pos_dice
                        best_thr_this_epoch = pred_thr

                logging.info(
                    f"Best PosDice={best_pos_dice_this_epoch:.4f} @ thr={best_thr_this_epoch:.2f}"
                )

                if best_pos_dice_this_epoch > best_seg_score:
                    best_seg_score = best_pos_dice_this_epoch
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'best_pos_dice': best_pos_dice_this_epoch,
                        'best_threshold': best_thr_this_epoch,
                    }, os.path.join(save_dir, 'best_seg.pth'))
                    logging.info(f"  -> New best PosDice: {best_pos_dice_this_epoch:.4f}")

                if vol_auc > best_auc:
                    best_auc = vol_auc
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'best_auc': best_auc,
                    }, os.path.join(save_dir, 'best_model.pth'))
                    logging.info(f"  -> New best AUC: {vol_auc:.4f}")

            del all_seg_probs, all_seg_coords, all_seg_gt_masks, all_seg_is_pos
            del all_vol_preds, all_vol_labels
            import gc; gc.collect()

        dist.barrier()

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
