import argparse
import logging
import os
import sys
from datetime import timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from src.training import Config, attach_log_file_handler, configure_runtime_paths
from src.models.feature_extractor_3d import VolumetricFeatureExtractor
from src.models.baseline_heads import GAPHead, ABMILHead, TransMILHead, DSMILHead
from src.data.volumetric_loader import create_volumetric_dataloader
from src.utils.metrics import evaluate_bag_level


class BaselineMILModel(nn.Module):
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
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    def forward(self, patches, coords, masks=None, mini_batch_size=8):
        features = self.feature_extractor(patches, mini_batch_size=mini_batch_size)
        logits, attention = self.head(features, mask=masks)
        return logits, attention


def setup_ddp():
    dist.init_process_group(backend='nccl', timeout=timedelta(hours=2))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def cleanup_ddp():
    dist.destroy_process_group()


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


def train_one_epoch(model, dataloader, optimizer, epoch, device, rank, config):
    model.train()
    total_loss = 0.0
    num_batches = 0

    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    else:
        pbar = dataloader

    mini_batch_size = config.model['feature_extractor'].get('mini_batch_size', 8)
    use_amp = config.training.get('mixed_precision', False)
    scaler = config.training.get('_grad_scaler')
    cls_weights = config.loss.get('cls_class_weights', [2.0, 1.0])
    label_smoothing = float(config.loss.get('label_smoothing', 0.1))

    oom_skip_count = 0
    for batch_idx, batch in enumerate(pbar):
        patches = batch['patches']
        coords = batch['coords'].to(device)
        labels = batch['labels'].to(device)
        masks = batch['masks'].to(device)
        optimizer.zero_grad(set_to_none=True)

        try:
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits, attention = model(patches, coords, masks, mini_batch_size=mini_batch_size)

                cls_weights_tensor = torch.tensor(cls_weights, device=device, dtype=logits.dtype)
                loss = F.cross_entropy(
                    logits, labels,
                    weight=cls_weights_tensor,
                    label_smoothing=label_smoothing,
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
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        except torch.cuda.OutOfMemoryError:
            oom_skip_count += 1
            if rank == 0:
                logging.warning(f"OOM at batch {batch_idx}, skipping (total: {oom_skip_count})")
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            try:
                dummy = sum(p.sum() * 0.0 for p in model.parameters() if p.requires_grad)
                if use_amp:
                    scaler.scale(dummy).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    dummy.backward()
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            continue

        del logits, attention, patches, coords, labels, masks
        torch.cuda.empty_cache()

    if rank == 0 and oom_skip_count > 0:
        logging.warning(f"Epoch {epoch+1}: {oom_skip_count} batches skipped due to OOM")

    if num_batches == 0:
        return 0.0
    return total_loss / num_batches


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
    parser.add_argument('--aggregation_method', type=str, required=True,
                        choices=['gap', 'abmil', 'transmil', 'dsmil'])
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--mini_batch_size', type=int, default=None)
    parser.add_argument('--max_patches', type=int, default=None)
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--output_root', type=str, default=None, help='Artifact root directory; defaults to a sibling mamba_artifacts directory')
    parser.add_argument('--run_name', type=str, default=None, help='Run name under the artifact root; defaults to aggregation_method + config filename stem')
    args = parser.parse_args()

    rank, world_size = setup_ddp()
    device = torch.device(f'cuda:{rank}')

    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    config = Config.from_yaml(args.config)
    if args.batch_size is not None:
        config.data['batch_size'] = args.batch_size
    if args.mini_batch_size is not None:
        config.model['feature_extractor']['mini_batch_size'] = args.mini_batch_size
    if args.max_patches is not None:
        config.data['max_patches'] = args.max_patches
    if args.mixed_precision:
        config.training['mixed_precision'] = True
    if config.training.get('disable_amp', False):
        config.training['mixed_precision'] = False

    config.training['_grad_scaler'] = torch.amp.GradScaler(
        'cuda', enabled=config.training.get('mixed_precision', False)
    )

    agg = args.aggregation_method
    runtime_paths = configure_runtime_paths(
        config,
        script_path=__file__,
        config_path=args.config,
        output_root=args.output_root,
        run_name=args.run_name or f"{agg}_{os.path.splitext(os.path.basename(args.config))[0]}",
    )
    if rank == 0:
        log_file_path = attach_log_file_handler(runtime_paths['log_dir'])
        logging.info(f"Training log file: {log_file_path}")
    log_dir = config.logging['log_dir']
    save_dir = config.checkpoint['save_dir']
    validation_history_path = config.logging['validation_history_path']

    if rank == 0:
        logging.info(f"Aggregation method: {agg}")
        logging.info(f"Patch size: {config.data['patch_size']}, Stride: {config.data['stride']}")
        logging.info(f"Batch size per GPU: {config.data['batch_size']}")
        logging.info(f"Max patches: {config.data.get('max_patches', 256)}")
        logging.info(f"Mixed precision: {config.training.get('mixed_precision', False)}")
        logging.info(f"Log dir: {log_dir}")
        logging.info(f"Checkpoint dir: {save_dir}")
        logging.info(f"Run name: {runtime_paths['run_name']}")
        logging.info(f"Artifact root: {runtime_paths['output_root']}")

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
        use_phase_aware=True,
        is_phase1=False,
        min_hu=config.data['min_hu'],
        max_hu=config.data['max_hu'],
        augment=True
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
        max_hu=config.data['max_hu']
    )

    model = BaselineMILModel(config, aggregation_method=agg).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training['lr'],
        weight_decay=config.training['weight_decay']
    )

    fixed_eval_threshold = float(config.training.get('fixed_eval_threshold', 0.35))
    val_freq = int(config.training.get('val_freq', 6))
    seg_val_freq = int(config.training.get('seg_val_freq', val_freq))
    val_start_epoch = int(config.training.get('val_start_epoch', 30))
    seg_val_max_batches = int(config.training.get('seg_val_max_batches', 100))
    seg_thr_min = float(config.training.get('seg_thr_search_min', 0.10))
    seg_thr_max = float(config.training.get('seg_thr_search_max', 0.90))
    seg_thr_step = float(config.training.get('seg_thr_search_step', 0.02))
    best_auc = 0.0
    best_seg_score = -1.0
    patch_size = tuple(config.data['patch_size'])

    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        if not os.path.exists(validation_history_path):
            with open(validation_history_path, 'w', encoding='utf-8') as f:
                f.write("epoch\tauc\tacc\tf1\tf1_best\tbest_threshold\tacc_fixed\tf1_fixed\tspec_fixed\tpr_auc\tseg_patch_dice_pos\tlr\tseg_best_thr\tseg_dice_pos_at_05\n")

    total_epochs = config.training['epochs']
    for epoch in range(total_epochs):
        epoch_lr = resolve_lr_for_epoch(config, epoch)
        for group in optimizer.param_groups:
            group['lr'] = epoch_lr

        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'set_epoch'):
            train_loader.batch_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(model, train_loader, optimizer, epoch, device, rank, config)

        if rank == 0:
            logging.info(f"Epoch {epoch+1}/{total_epochs} | LR={epoch_lr:.8f} | Train Loss={train_loss:.4f}")

            do_validation = (epoch + 1) >= val_start_epoch and (
                val_freq <= 1 or (epoch + 1) % val_freq == 0 or (epoch + 1) == total_epochs
            )
            do_seg = do_validation and (
                seg_val_freq <= 1 or (epoch + 1) % seg_val_freq == 0 or (epoch + 1) == total_epochs
            )

            if do_validation:
                model.module.eval()
                all_preds = []
                all_labels = []
                all_seg_probs = []
                all_seg_gt = []
                all_seg_is_pos = []
                seg_batch_count = 0
                mini_batch_size = config.model['feature_extractor'].get('mini_batch_size', 8)
                can_localize = (agg != 'gap')

                with torch.no_grad():
                    for batch in tqdm(val_loader, desc="Validation", leave=False):
                        patches = batch['patches']
                        coords = batch['coords'].to(device)
                        labels = batch['labels'].to(device)
                        masks = batch['masks'].to(device)

                        with torch.amp.autocast('cuda', enabled=config.training.get('mixed_precision', False)):
                            logits, attention = model.module(
                                patches, coords, masks, mini_batch_size=mini_batch_size
                            )

                        all_preds.append(logits)
                        all_labels.append(labels)

                        if do_seg and can_localize and (seg_val_max_batches == 0 or seg_batch_count < seg_val_max_batches):
                            seg_batch_count += 1
                            for b in range(labels.shape[0]):
                                valid_n = int(masks[b].sum().item())
                                if valid_n <= 0:
                                    continue
                                volume_path = batch['volume_paths'][b]
                                gt_path = get_gt_mask_path(volume_path)
                                gt_mask_hwz = load_gt_mask_hwz(gt_path) if gt_path else None
                                if gt_mask_hwz is None:
                                    continue
                                att_b = attention[b, :valid_n].cpu().numpy()
                                att_min = att_b.min()
                                att_max = att_b.max()
                                if att_max - att_min > 1e-8:
                                    probs_b = (att_b - att_min) / (att_max - att_min)
                                else:
                                    probs_b = np.zeros_like(att_b)
                                coords_b = coords[b, :valid_n].detach().cpu().numpy()
                                gt_labels_b = (compute_patch_positive_labels(
                                    coords_b, gt_mask_hwz, patch_size
                                ) > 0).astype(np.int8)
                                all_seg_probs.append(probs_b)
                                all_seg_gt.append(gt_labels_b)
                                all_seg_is_pos.append(int(labels[b].item()) == 1)

                all_preds = torch.cat(all_preds, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
                metrics = evaluate_bag_level(all_preds, all_labels, fixed_threshold=fixed_eval_threshold)

                seg_best_thr = 0.5
                seg_patch_dice_pos = -1.0
                seg_dice_pos_at_05 = -1.0
                if do_seg and can_localize and len(all_seg_probs) > 0:
                    def _seg_dice_at_thr(probs_list, gt_list, bag_filter, thr):
                        tp = fp = fn = 0
                        for p, g, keep in zip(probs_list, gt_list, bag_filter):
                            if not keep:
                                continue
                            pred = p >= thr
                            gv = g.astype(bool)
                            tp += int((pred & gv).sum())
                            fp += int((pred & ~gv).sum())
                            fn += int((~pred & gv).sum())
                        return (2.0 * tp) / (2.0 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

                    thresholds = np.arange(seg_thr_min, seg_thr_max + 1e-9, seg_thr_step)
                    best_d = 0.0
                    for thr in thresholds:
                        d = _seg_dice_at_thr(all_seg_probs, all_seg_gt, all_seg_is_pos, thr)
                        if d > best_d:
                            best_d = d
                            seg_best_thr = float(thr)
                    seg_patch_dice_pos = best_d
                    seg_dice_pos_at_05 = _seg_dice_at_thr(
                        all_seg_probs, all_seg_gt, all_seg_is_pos, 0.5
                    )

                logging.info(
                    f"Val: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}, "
                    f"F1_best={metrics['f1_best']:.4f}@th={metrics['best_threshold']:.2f}, "
                    f"PR-AUC={metrics['pr_auc']:.4f}"
                )
                logging.info(
                    f"Val Fixed@{fixed_eval_threshold:.2f}: "
                    f"F1={metrics['f1_fixed']:.4f}, Spec={metrics['specificity_fixed']:.4f}"
                )
                if do_seg and can_localize:
                    logging.info(
                        f"Val Seg (attention): PosDice={seg_patch_dice_pos:.4f}@thr={seg_best_thr:.2f}, "
                        f"PosDice@0.5={seg_dice_pos_at_05:.4f}"
                    )

                with open(validation_history_path, 'a', encoding='utf-8') as f:
                    f.write(
                        f"{epoch+1}\t{metrics['auc']:.6f}\t{metrics['accuracy']:.6f}\t"
                        f"{metrics['f1']:.6f}\t{metrics['f1_best']:.6f}\t"
                        f"{metrics['best_threshold']:.4f}\t{metrics['accuracy_fixed']:.6f}\t"
                        f"{metrics['f1_fixed']:.6f}\t{metrics['specificity_fixed']:.6f}\t"
                        f"{metrics['pr_auc']:.6f}\t{seg_patch_dice_pos:.6f}\t"
                        f"{epoch_lr:.8f}\t{seg_best_thr:.4f}\t{seg_dice_pos_at_05:.6f}\n"
                    )

                if metrics['auc'] > best_auc:
                    best_auc = metrics['auc']
                    ckpt_path = os.path.join(save_dir, f'best_model_{agg}.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'metrics': metrics,
                        'aggregation_method': agg,
                    }, ckpt_path)
                    logging.info(f"New best AUC={best_auc:.4f}, saved to {ckpt_path}")

                if seg_patch_dice_pos > best_seg_score and metrics['auc'] >= 0.70:
                    best_seg_score = seg_patch_dice_pos
                    ckpt_path = os.path.join(save_dir, f'best_seg_{agg}.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'metrics': {**metrics, 'seg_patch_dice_pos': seg_patch_dice_pos, 'seg_best_thr': seg_best_thr},
                        'aggregation_method': agg,
                    }, ckpt_path)
                    logging.info(f"New best PosDice={best_seg_score:.4f}, saved to {ckpt_path}")

        dist.barrier(device_ids=[rank])

    if rank == 0:
        logging.info(f"Training complete. Best AUC={best_auc:.4f}, Best PosDice={best_seg_score:.4f}")

    cleanup_ddp()


if __name__ == '__main__':
    main()
