import torch
import numpy as np
import logging
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from typing import Dict


def evaluate_bag_level(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    fixed_threshold: float = 0.5
) -> Dict[str, float]:
    probs = torch.softmax(predictions, dim=1)[:, 1]
    preds = predictions.argmax(1)
    
    labels_np = labels.cpu().numpy()
    probs_np = probs.detach().cpu().numpy()
    preds_np = preds.cpu().numpy()
    fixed_preds = (probs_np >= float(fixed_threshold)).astype(np.int64)
    tn = float(((fixed_preds == 0) & (labels_np == 0)).sum())
    fp = float(((fixed_preds == 1) & (labels_np == 0)).sum())
    specificity_fixed = tn / (tn + fp + 1e-12)
    unique_labels = np.unique(labels_np)
    single_class_eval = len(unique_labels) < 2

    if single_class_eval:
        logging.warning(
            "single-class split, auc/pr_auc skipped (labels=%s)",
            unique_labels.tolist(),
        )

    best_f1 = 0.0
    best_threshold = 0.5
    best_acc = 0.0
    for t in np.linspace(0.05, 0.95, 19):
        preds_t = (probs_np >= t).astype(np.int64)
        f1_t = f1_score(labels_np, preds_t, zero_division=0)
        if f1_t > best_f1:
            best_f1 = float(f1_t)
            best_threshold = float(t)
            best_acc = float(accuracy_score(labels_np, preds_t))
    
    metrics = {
        'auc': 0.0 if single_class_eval else float(roc_auc_score(labels_np, probs_np)),
        'pr_auc': 0.0 if single_class_eval else float(average_precision_score(labels_np, probs_np)),
        'accuracy': float(accuracy_score(labels_np, preds_np)),
        'precision': float(precision_score(labels_np, preds_np, zero_division=0)),
        'recall': float(recall_score(labels_np, preds_np, zero_division=0)),
        'f1': float(f1_score(labels_np, preds_np, zero_division=0)),
        'accuracy_fixed': float(accuracy_score(labels_np, fixed_preds)),
        'precision_fixed': float(precision_score(labels_np, fixed_preds, zero_division=0)),
        'recall_fixed': float(recall_score(labels_np, fixed_preds, zero_division=0)),
        'f1_fixed': float(f1_score(labels_np, fixed_preds, zero_division=0)),
        'specificity_fixed': float(specificity_fixed),
        'fixed_threshold': float(fixed_threshold),
        'f1_best': best_f1,
        'best_threshold': best_threshold,
        'accuracy_best': best_acc,
        'single_class_eval': bool(single_class_eval),
    }
    
    return metrics


def evaluate_slice_level(
    attention_weights: torch.Tensor,
    gt_masks: torch.Tensor
) -> Dict[str, float]:
    gt_slice_labels = (gt_masks.sum(dim=(1, 2)) > 0).float()
    
    att_flat = attention_weights.flatten().cpu().numpy()
    gt_flat = gt_slice_labels.flatten().cpu().numpy()
    
    if len(np.unique(gt_flat)) > 1:
        slice_auc = float(roc_auc_score(gt_flat, att_flat))
    else:
        slice_auc = 0.0
    
    return {'slice_auc': slice_auc}


def evaluate_segmentation_binary(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor
) -> Dict[str, float]:
    pred = pred_mask.float().view(-1)
    gt = gt_mask.float().view(-1)
    
    intersection = (pred * gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    
    dice = (2 * intersection) / (pred_sum + gt_sum + 1e-6)
    iou = intersection / (pred_sum + gt_sum - intersection + 1e-6)
    
    return {
        'dice': float(dice),
        'iou': float(iou)
    }
