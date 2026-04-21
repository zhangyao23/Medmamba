import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


def temporal_consistency_loss(
    attention_weights: torch.Tensor,
    threshold: float = 0.5
) -> torch.Tensor:
    pred = (attention_weights > threshold).float()
    
    pad_pred = F.pad(pred, (1, 1), mode='replicate')
    prev = pad_pred[:, :-2]
    next = pad_pred[:, 2:]
    
    isolated = pred * (1 - prev) * (1 - next)
    
    return isolated.sum() / (pred.sum() + 1e-6)


def codebook_entropy_loss(
    code_indices: torch.Tensor,
    num_embeddings: int
) -> torch.Tensor:
    code_indices_flat = code_indices.reshape(-1)
    
    counts = torch.bincount(code_indices_flat, minlength=num_embeddings).float()
    probs = counts / (counts.sum() + 1e-10)
    
    probs = probs + 1e-10
    entropy = -(probs * torch.log(probs)).sum()
    
    max_entropy = torch.log(torch.tensor(num_embeddings, dtype=torch.float32, device=code_indices.device))
    
    normalized_entropy = entropy / max_entropy
    
    entropy_loss = 1.0 - normalized_entropy
    
    return entropy_loss


def code_guided_attention_loss(
    attention: torch.Tensor,
    codes: torch.Tensor,
    labels: torch.Tensor,
    healthy_code_mask: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    cancer_weight: float = 1.0,
    healthy_weight: float = 0.01,
) -> torch.Tensor:
    pos_mask = (labels == 1)
    if not pos_mask.any() or healthy_code_mask is None or not healthy_code_mask.any():
        return torch.tensor(0.0, device=attention.device)

    loss = torch.tensor(0.0, device=attention.device)
    count = 0

    for b in range(labels.shape[0]):
        if labels[b].item() != 1:
            continue

        att_b = attention[b]
        codes_b = codes[b]

        if mask is not None:
            valid = mask[b].bool()
            att_b = att_b[valid]
            codes_b = codes_b[valid]

        if att_b.numel() == 0:
            continue

        is_cancer = ~healthy_code_mask[codes_b]
        target = torch.where(
            is_cancer,
            torch.tensor(cancer_weight, device=attention.device),
            torch.tensor(healthy_weight, device=attention.device)
        )
        target = target / (target.sum() + 1e-8)

        att_log = torch.log(att_b.clamp(min=1e-8))
        target_log = torch.log(target.clamp(min=1e-8))
        kl = (target * (target_log - att_log)).sum()
        loss = loss + kl.clamp(min=0, max=10)
        count += 1

    if count > 0:
        loss = loss / count
    return loss


def gt_guided_code_loss(
    z_e: torch.Tensor,
    gt_patch_labels: torch.Tensor,
    codebook_embedding: torch.Tensor,
    healthy_code_mask: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Push pre-quantization encoder outputs toward the appropriate code type
    based on patch-level GT segmentation labels.

    GT-positive patches (label=1) → minimize distance to nearest cancer code.
    GT-negative patches (label=0) → minimize distance to nearest healthy code.
    Patches with label=-1 are ignored.

    This loss has real gradients back to the encoder because it operates on
    z_e (continuous), not on discrete code indices.
    """
    cancer_code_mask = ~healthy_code_mask
    if not cancer_code_mask.any() or not healthy_code_mask.any():
        return torch.tensor(0.0, device=z_e.device)

    B, N, D = z_e.shape
    z_flat = z_e.reshape(-1, D)
    gt_flat = gt_patch_labels.reshape(-1)

    valid = gt_flat >= 0
    if mask is not None:
        valid = valid & mask.reshape(-1).bool()
    if not valid.any():
        return torch.tensor(0.0, device=z_e.device)

    z_valid = z_flat[valid]
    gt_valid = (gt_flat[valid] > 0).float()

    cancer_embs  = codebook_embedding[cancer_code_mask].detach()
    healthy_embs = codebook_embedding[healthy_code_mask].detach()

    dist_to_cancer  = torch.cdist(z_valid, cancer_embs).min(dim=1)[0]
    dist_to_healthy = torch.cdist(z_valid, healthy_embs).min(dim=1)[0]

    loss = (gt_valid * dist_to_cancer + (1.0 - gt_valid) * dist_to_healthy).mean()
    return loss


def attention_sparsity_loss(
    attention: torch.Tensor,
    labels: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    pos_mask = (labels == 1)
    if not pos_mask.any():
        return torch.tensor(0.0, device=attention.device)

    loss = torch.tensor(0.0, device=attention.device)
    count = 0

    for b in range(labels.shape[0]):
        if labels[b].item() != 1:
            continue
        att_b = attention[b]
        if mask is not None:
            valid = mask[b].bool()
            att_b = att_b[valid]
        n = att_b.numel()
        if n < 2:
            continue
        att_b = att_b.clamp(min=1e-8)
        entropy = -(att_b * torch.log(att_b)).sum()
        max_entropy = torch.log(torch.tensor(float(n), device=attention.device))
        loss = loss + entropy / max_entropy
        count += 1

    if count > 0:
        loss = loss / count
    return loss


def attention_uniformity_loss(
    attention: torch.Tensor,
    labels: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    neg_mask = (labels == 0)
    if not neg_mask.any():
        return torch.tensor(0.0, device=attention.device)

    loss = torch.tensor(0.0, device=attention.device)
    count = 0

    for b in range(labels.shape[0]):
        if labels[b].item() != 0:
            continue
        att_b = attention[b]
        if mask is not None:
            valid = mask[b].bool()
            att_b = att_b[valid]
        n = att_b.numel()
        if n < 2:
            continue
        att_b = att_b.clamp(min=1e-8)
        entropy = -(att_b * torch.log(att_b)).sum()
        max_entropy = torch.log(torch.tensor(float(n), device=attention.device))
        loss = loss + (1.0 - entropy / max_entropy)
        count += 1

    if count > 0:
        loss = loss / count
    return loss


def cancer_logit_enrichment_reg(
    cancer_logit: torch.Tensor,
    normal_patch_count: torch.Tensor,
    cancer_patch_count: torch.Tensor,
) -> torch.Tensor:
    total = normal_patch_count + cancer_patch_count + 1e-8
    enrichment = (cancer_patch_count / total).detach()
    return F.mse_loss(torch.sigmoid(cancer_logit), enrichment)


def recon_teacher_loss(
    mil_seg_logits: torch.Tensor,
    recon_pred: torch.Tensor,
    recon_target: torch.Tensor,
    labels: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    top_k_ratio: float = 0.2,
) -> torch.Tensor:
    B, N = mil_seg_logits.shape
    loss_sum = torch.tensor(0.0, device=mil_seg_logits.device)
    count = 0

    for b in range(B):
        if mask is not None:
            valid = mask[b].bool()
        else:
            valid = torch.ones(N, dtype=torch.bool, device=mil_seg_logits.device)
        n_valid = int(valid.sum().item())
        if n_valid < 2:
            continue

        logits_b = mil_seg_logits[b][valid]

        rp = recon_pred[b, :N][valid].float()
        rt = recon_target[b, :N][valid].float()
        recon_err = (rp - rt).abs().mean(dim=tuple(range(1, rp.dim())))

        if labels[b].item() == 1:
            err_min = recon_err.min()
            err_max = recon_err.max()
            if err_max - err_min > 1e-6:
                teacher = (recon_err - err_min) / (err_max - err_min)
            else:
                teacher = torch.zeros_like(recon_err)
        else:
            teacher = torch.zeros(n_valid, device=mil_seg_logits.device)

        pred_prob = torch.sigmoid(logits_b)
        loss_sum = loss_sum + F.mse_loss(pred_prob, teacher)
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=mil_seg_logits.device)
    return loss_sum / count


def feature_distillation_loss(
    z_e: torch.Tensor,
    z_e_frozen: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if mask is not None:
        mask_f = mask.unsqueeze(-1).float()
        diff = (z_e - z_e_frozen.detach()) ** 2
        return (diff * mask_f).sum() / mask_f.sum().clamp(min=1) / z_e.shape[-1]
    return F.mse_loss(z_e, z_e_frozen.detach())


def mil_seg_loss(mil_seg_logits, labels, masks=None, smooth_r=5.0):
    B, N = mil_seg_logits.shape
    loss_sum = torch.tensor(0.0, device=mil_seg_logits.device)
    count = 0

    for b in range(B):
        if masks is not None:
            valid = masks[b].bool()
        else:
            valid = torch.ones(N, dtype=torch.bool, device=mil_seg_logits.device)
        n_valid = valid.sum()
        if n_valid == 0:
            continue

        logits_b = mil_seg_logits[b][valid]

        if labels[b].item() == 1:
            bag_logit = (1.0 / smooth_r) * torch.logsumexp(smooth_r * logits_b, dim=0)
        else:
            bag_logit = logits_b.mean()

        loss_sum = loss_sum + F.binary_cross_entropy_with_logits(
            bag_logit, labels[b].float())
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=mil_seg_logits.device)
    return loss_sum / count


def patch_seg_bce_loss(patch_seg_logits, gt_patch_labels, mask=None, pos_weight=5.0):
    logits_flat = patch_seg_logits.reshape(-1)
    gt_flat = gt_patch_labels.reshape(-1).float()
    valid = gt_flat >= 0
    if mask is not None:
        valid = valid & mask.reshape(-1).bool()
    if not valid.any():
        return torch.tensor(0.0, device=patch_seg_logits.device)
    pw = torch.tensor([pos_weight], device=logits_flat.device)
    return F.binary_cross_entropy_with_logits(
        logits_flat[valid], gt_flat[valid], pos_weight=pw
    )


def patch_seg_dice_bce_loss(patch_seg_logits, gt_patch_labels, mask=None,
                            pos_weight=20.0, dice_weight=1.0, bce_weight=1.0):
    bce = patch_seg_bce_loss(patch_seg_logits, gt_patch_labels, mask, pos_weight)

    logits_flat = patch_seg_logits.reshape(-1)
    gt_flat = gt_patch_labels.reshape(-1).float()
    valid = gt_flat >= 0
    if mask is not None:
        valid = valid & mask.reshape(-1).bool()
    if not valid.any():
        return bce_weight * bce

    probs = torch.sigmoid(logits_flat[valid])
    targets = gt_flat[valid]
    intersection = (probs * targets).sum()
    dice = 1.0 - (2.0 * intersection + 1e-6) / (probs.sum() + targets.sum() + 1e-6)

    return bce_weight * bce + dice_weight * dice


def patch_seg_focal_dice_bce_loss(patch_seg_logits, gt_patch_labels, mask=None,
                                   pos_weight=20.0, dice_weight=1.0, bce_weight=1.0,
                                   focal_alpha=0.25, focal_gamma=2.0):
    logits_flat = patch_seg_logits.reshape(-1)
    gt_flat = gt_patch_labels.reshape(-1).float()
    valid = gt_flat >= 0
    if mask is not None:
        valid = valid & mask.reshape(-1).bool()
    if not valid.any():
        return torch.tensor(0.0, device=patch_seg_logits.device)

    logits_v = logits_flat[valid]
    targets_v = gt_flat[valid]

    bce_per_elem = F.binary_cross_entropy_with_logits(logits_v, targets_v, reduction='none')
    pt = torch.exp(-bce_per_elem)
    pw = torch.where(targets_v > 0.5, pos_weight, 1.0)
    focal_bce = (focal_alpha * (1.0 - pt) ** focal_gamma * bce_per_elem * pw).mean()

    probs = torch.sigmoid(logits_v)
    intersection = (probs * targets_v).sum()
    dice = 1.0 - (2.0 * intersection + 1e-6) / (probs.sum() + targets_v.sum() + 1e-6)

    return bce_weight * focal_bce + dice_weight * dice


def voxel_seg_dice_loss(voxel_logits, gt_voxel_masks, patch_mask=None):
    B, N = voxel_logits.shape[:2]
    spatial = voxel_logits.shape[2:]
    logits_flat = voxel_logits.reshape(B * N, -1)
    gt_flat = gt_voxel_masks.reshape(B * N, -1).float()

    if patch_mask is not None:
        valid_patches = patch_mask.reshape(B * N).bool()
    else:
        valid_patches = torch.ones(B * N, dtype=torch.bool, device=voxel_logits.device)

    if not valid_patches.any():
        return torch.tensor(0.0, device=voxel_logits.device)

    logits_v = logits_flat[valid_patches]
    gt_v = gt_flat[valid_patches]

    probs = torch.sigmoid(logits_v)
    intersection = (probs * gt_v).sum(dim=1)
    dice_per_patch = 1.0 - (2.0 * intersection + 1e-6) / (probs.sum(dim=1) + gt_v.sum(dim=1) + 1e-6)

    return dice_per_patch.mean()


def voxel_seg_dice_bce_loss(voxel_logits, gt_voxel_masks, patch_mask=None,
                            dice_weight=1.0, bce_weight=0.5):
    B, N = voxel_logits.shape[:2]
    logits_flat = voxel_logits.reshape(B * N, -1)
    gt_flat = gt_voxel_masks.reshape(B * N, -1).float()

    if patch_mask is not None:
        valid_patches = patch_mask.reshape(B * N).bool()
    else:
        valid_patches = torch.ones(B * N, dtype=torch.bool, device=voxel_logits.device)

    if not valid_patches.any():
        return torch.tensor(0.0, device=voxel_logits.device)

    logits_v = logits_flat[valid_patches]
    gt_v = gt_flat[valid_patches]

    probs = torch.sigmoid(logits_v)
    intersection = (probs * gt_v).sum(dim=1)
    dice = 1.0 - (2.0 * intersection + 1e-6) / (probs.sum(dim=1) + gt_v.sum(dim=1) + 1e-6)
    dice_loss = dice.mean()

    bce_loss = F.binary_cross_entropy_with_logits(logits_v, gt_v, reduction='mean')

    return dice_weight * dice_loss + bce_weight * bce_loss


def voxel_seg_focal_dice_bce_loss(voxel_logits, gt_voxel_masks, patch_mask=None,
                                    dice_weight=1.0, bce_weight=1.0,
                                    focal_alpha=0.25, focal_gamma=2.0, pos_weight=20.0):
    B, N = voxel_logits.shape[:2]
    logits_flat = voxel_logits.reshape(B * N, -1)
    gt_flat = gt_voxel_masks.reshape(B * N, -1).float()

    if patch_mask is not None:
        valid_patches = patch_mask.reshape(B * N).bool()
    else:
        valid_patches = torch.ones(B * N, dtype=torch.bool, device=voxel_logits.device)

    if not valid_patches.any():
        return torch.tensor(0.0, device=voxel_logits.device)

    logits_v = logits_flat[valid_patches]
    gt_v = gt_flat[valid_patches]

    # Focal BCE
    bce_per_elem = F.binary_cross_entropy_with_logits(logits_v, gt_v, reduction='none')
    pt = torch.exp(-bce_per_elem)
    # Apply pos_weight to positive samples in focal loss
    pw = torch.where(gt_v > 0.5, pos_weight, 1.0)
    focal_bce = (focal_alpha * (1.0 - pt) ** focal_gamma * bce_per_elem * pw).mean()

    # Dice
    probs = torch.sigmoid(logits_v)
    intersection = (probs * gt_v).sum(dim=1)
    dice = 1.0 - (2.0 * intersection + 1e-6) / (probs.sum(dim=1) + gt_v.sum(dim=1) + 1e-6)
    dice_loss = dice.mean()

    return dice_weight * dice_loss + bce_weight * focal_bce


def voxel_seg_deep_supervision_loss(
    main_logits, aux_outputs, gt_voxel_masks, patch_mask=None,
    ds_weights=None, loss_type='dice_bce',
):
    if ds_weights is None:
        ds_weights = [0.25, 0.25]

    if loss_type == 'focal_dice_bce':
        loss_fn = voxel_seg_focal_dice_bce_loss
    elif loss_type == 'dice_bce':
        loss_fn = voxel_seg_dice_bce_loss
    else:
        loss_fn = voxel_seg_dice_loss

    main_loss = loss_fn(main_logits, gt_voxel_masks, patch_mask=patch_mask)

    if aux_outputs is None:
        return main_loss

    ds_loss = torch.tensor(0.0, device=main_logits.device)
    for idx, (key, weight) in enumerate(zip(['ds_8', 'ds_16'], ds_weights)):
        if key not in aux_outputs:
            continue
        aux_logits = aux_outputs[key]
        B, N = aux_logits.shape[:2]
        aux_spatial = aux_logits.shape[2:]
        gt_down = F.adaptive_max_pool3d(
            gt_voxel_masks.view(B * N, 1, *gt_voxel_masks.shape[2:]).float(),
            output_size=aux_spatial[1:],
        ).view(B, N, *aux_spatial[1:])
        ds_loss = ds_loss + weight * loss_fn(aux_logits, gt_down, patch_mask=patch_mask)

    return main_loss + ds_loss


def segmentation_prior_loss(
    codes: torch.Tensor,
    labels: torch.Tensor,
    healthy_code_mask: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    pos_ratio_min: float = 0.03,
    pos_ratio_max: float = 0.20,
    sep_margin: float = 0.08,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if codes is None or healthy_code_mask is None or not healthy_code_mask.any():
        zero = torch.tensor(0.0, device=labels.device)
        return zero, zero, zero

    # cancer ratio per bag: fraction of patches assigned to cancer codes
    is_cancer_code = (~healthy_code_mask[codes]).float()
    if mask is not None:
        valid = mask.float()
        denom = valid.sum(dim=1).clamp(min=1.0)
        bag_ratio = (is_cancer_code * valid).sum(dim=1) / denom
    else:
        bag_ratio = is_cancer_code.mean(dim=1)

    neg_mask = (labels == 0)
    pos_mask = (labels == 1)

    if neg_mask.any():
        neg_loss = bag_ratio[neg_mask].mean()
    else:
        neg_loss = torch.tensor(0.0, device=labels.device)

    if pos_mask.any():
        pos_ratio = bag_ratio[pos_mask]
        pos_low = torch.relu(pos_ratio_min - pos_ratio)
        pos_high = torch.relu(pos_ratio - pos_ratio_max)
        pos_loss = (pos_low + pos_high).mean()
    else:
        pos_loss = torch.tensor(0.0, device=labels.device)

    if neg_mask.any() and pos_mask.any():
        gap = bag_ratio[pos_mask].mean() - bag_ratio[neg_mask].mean()
        sep_loss = torch.relu(sep_margin - gap)
    else:
        sep_loss = torch.tensor(0.0, device=labels.device)

    return neg_loss, pos_loss, sep_loss


def spatial_smoothness_loss(patch_seg_logits, coords, mask=None, patch_size=(32, 32, 32)):
    B, N = patch_seg_logits.shape[:2]
    total_loss = torch.tensor(0.0, device=patch_seg_logits.device)
    count = 0
    pd, ph, pw = patch_size

    for b in range(B):
        if mask is not None:
            valid_n = int(mask[b].sum().item())
        else:
            valid_n = N
        if valid_n < 2:
            continue

        logits_b = patch_seg_logits[b, :valid_n, 0]
        coords_b = coords[b, :valid_n].float()

        diffs = coords_b.unsqueeze(0) - coords_b.unsqueeze(1)
        diffs[:, :, 0] = diffs[:, :, 0] / pd
        diffs[:, :, 1] = diffs[:, :, 1] / ph
        diffs[:, :, 2] = diffs[:, :, 2] / pw
        dist_sq = (diffs ** 2).sum(dim=-1)

        neighbor_mask = (dist_sq > 0) & (dist_sq <= 3.01)

        if neighbor_mask.any():
            logit_diff = (logits_b.unsqueeze(0) - logits_b.unsqueeze(1)) ** 2
            total_loss = total_loss + logit_diff[neighbor_mask].mean()
            count += 1

    if count > 0:
        total_loss = total_loss / count
    return total_loss


class CombinedLoss(nn.Module):
    def __init__(
        self,
        lambda_cls: float = 1.0,
        lambda_nu: float = 0.5,
        lambda_vq: float = 0.25,
        lambda_temporal: float = 0.1,
        lambda_entropy: float = 0.1,
        lambda_recon: float = 0.0,
        recon_in_mil_weight: float = 0.0,
        recon_type: str = 'l1',
        single_class_cls_weight: float = 0.0,
        cls_class_weights: Optional[Tuple[float, float]] = None,
        use_focal_cls: bool = False,
        focal_gamma: float = 2.0,
        focal_alpha_pos: float = 0.5,
        label_smoothing: float = 0.0,
        nu_margin: float = 0.5,
        num_embeddings: int = 100,
        lambda_code_att: float = 0.0,
        code_att_cancer_weight: float = 1.0,
        code_att_healthy_weight: float = 0.01,
        lambda_seg_neg: float = 0.0,
        lambda_seg_pos: float = 0.0,
        lambda_seg_sep: float = 0.0,
        seg_pos_ratio_min: float = 0.03,
        seg_pos_ratio_max: float = 0.20,
        seg_sep_margin: float = 0.08,
        lambda_gt_code: float = 0.0,
        lambda_patch_seg: float = 0.0,
        patch_seg_pos_weight: float = 5.0,
        seg_loss_type: str = 'bce',
        seg_dice_weight: float = 1.0,
        seg_bce_weight: float = 1.0,
        lambda_voxel_seg: float = 0.0,
        lambda_spatial_smooth: float = 0.0,
        lambda_code_cls: float = 0.0,
        lambda_embed_sep: float = 0.0,
        lambda_pseudo_routing: float = 0.0,
        lambda_mil_seg: float = 0.0,
        mil_seg_smooth_r: float = 5.0,
        lambda_att_sparsity: float = 0.0,
        lambda_att_uniformity: float = 0.0,
        lambda_cancer_logit_reg: float = 0.0,
        lambda_recon_teacher: float = 0.0,
        recon_teacher_top_k: float = 0.2,
        lambda_feat_distill: float = 0.0,
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_nu = lambda_nu
        self.lambda_vq = lambda_vq
        self.lambda_temporal = lambda_temporal
        self.lambda_entropy = lambda_entropy
        self.lambda_recon = lambda_recon
        self.recon_in_mil_weight = recon_in_mil_weight
        self.recon_type = recon_type
        self.single_class_cls_weight = single_class_cls_weight
        self.cls_class_weights = cls_class_weights
        self.use_focal_cls = use_focal_cls
        self.focal_gamma = focal_gamma
        self.focal_alpha_pos = focal_alpha_pos
        self.label_smoothing = label_smoothing
        self.nu_margin = nu_margin
        self.num_embeddings = num_embeddings
        self.lambda_code_att = lambda_code_att
        self.code_att_cancer_weight = code_att_cancer_weight
        self.code_att_healthy_weight = code_att_healthy_weight
        self.lambda_seg_neg = lambda_seg_neg
        self.lambda_seg_pos = lambda_seg_pos
        self.lambda_seg_sep = lambda_seg_sep
        self.seg_pos_ratio_min = seg_pos_ratio_min
        self.seg_pos_ratio_max = seg_pos_ratio_max
        self.seg_sep_margin = seg_sep_margin
        self.lambda_gt_code = lambda_gt_code
        self.lambda_patch_seg = lambda_patch_seg
        self.patch_seg_pos_weight = patch_seg_pos_weight
        self.seg_loss_type = seg_loss_type
        self.seg_dice_weight = seg_dice_weight
        self.seg_bce_weight = seg_bce_weight
        self.lambda_voxel_seg = lambda_voxel_seg
        self.lambda_spatial_smooth = lambda_spatial_smooth
        self.lambda_code_cls = lambda_code_cls
        self.lambda_embed_sep = lambda_embed_sep
        self.lambda_pseudo_routing = lambda_pseudo_routing
        self.lambda_mil_seg = lambda_mil_seg
        self.mil_seg_smooth_r = mil_seg_smooth_r
        self.lambda_att_sparsity = lambda_att_sparsity
        self.lambda_att_uniformity = lambda_att_uniformity
        self.lambda_cancer_logit_reg = lambda_cancer_logit_reg
        self.lambda_recon_teacher = lambda_recon_teacher
        self.recon_teacher_top_k = recon_teacher_top_k
        self.lambda_feat_distill = lambda_feat_distill
        self.asymmetric_recon = True
        self.phase2_disable_recon = False
    
    def forward(
        self,
        logits: torch.Tensor,
        attention: torch.Tensor,
        labels: torch.Tensor,
        vq_loss: torch.Tensor,
        codes: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        recon_pred: Optional[torch.Tensor] = None,
        recon_target: Optional[torch.Tensor] = None,
        mode: str = 'mil_train',
        healthy_code_mask: Optional[torch.Tensor] = None,
        z_e: Optional[torch.Tensor] = None,
        gt_patch_labels: Optional[torch.Tensor] = None,
        codebook_embedding: Optional[torch.Tensor] = None,
        patch_seg_logits: Optional[torch.Tensor] = None,
        voxel_seg_logits: Optional[torch.Tensor] = None,
        gt_voxel_masks: Optional[torch.Tensor] = None,
        voxel_aux_outputs: Optional[Dict] = None,
        sorted_coords: Optional[torch.Tensor] = None,
        patch_size: Optional[Tuple[int, int, int]] = None,
        code_cls_logit: Optional[torch.Tensor] = None,
        embed_sep_loss: Optional[torch.Tensor] = None,
        pseudo_routing_loss: Optional[torch.Tensor] = None,
        mil_seg_logits: Optional[torch.Tensor] = None,
        cancer_logit: Optional[torch.Tensor] = None,
        normal_patch_count: Optional[torch.Tensor] = None,
        cancer_patch_count: Optional[torch.Tensor] = None,
        z_e_frozen: Optional[torch.Tensor] = None,
        distill_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        cls_weights_tensor = None
        if self.cls_class_weights is not None:
            cls_weights_tensor = torch.tensor(
                self.cls_class_weights,
                device=logits.device,
                dtype=logits.dtype
            )
        if self.use_focal_cls:
            ce = F.cross_entropy(logits, labels, reduction='none', weight=cls_weights_tensor)
            pt = torch.exp(-ce)
            alpha_t = torch.where(
                labels == 1,
                torch.tensor(self.focal_alpha_pos, device=labels.device, dtype=logits.dtype),
                torch.tensor(1.0 - self.focal_alpha_pos, device=labels.device, dtype=logits.dtype)
            )
            raw_cls_loss = (alpha_t * ((1.0 - pt) ** self.focal_gamma) * ce).mean()
        else:
            raw_cls_loss = F.cross_entropy(
                logits, labels, weight=cls_weights_tensor,
                label_smoothing=self.label_smoothing
            )
        if labels.unique().numel() < 2:
            cls_loss = raw_cls_loss * self.single_class_cls_weight
        else:
            cls_loss = raw_cls_loss
        
        if torch.isnan(cls_loss) or torch.isinf(cls_loss):
            cls_loss = (logits * 0).sum()
        
        neg_mask = (labels == 0)
        pos_mask = (labels == 1)
        
        if neg_mask.any() and pos_mask.any():
            if mask is not None:
                neg_att = attention[neg_mask].clone()
                neg_att = neg_att.masked_fill(mask[neg_mask] == 0, -1e4)
            else:
                neg_att = attention[neg_mask]
            hard_neg_score = torch.clamp(neg_att.max(dim=1)[0].mean(), -10, 10)
            
            if mask is not None:
                pos_att = attention[pos_mask].clone()
                pos_att = pos_att.masked_fill(mask[pos_mask] == 0, -1e4)
            else:
                pos_att = attention[pos_mask]
            pos_score = torch.clamp(pos_att.max(dim=1)[0].mean(), -10, 10)
            
            nu_loss = torch.clamp(
                hard_neg_score - pos_score + self.nu_margin,
                min=0,
                max=10
            )
        else:
            nu_loss = torch.tensor(0.0, device=logits.device)
        
        temporal_loss = temporal_consistency_loss(attention)
        temporal_loss = torch.clamp(temporal_loss, 0, 10)
        
        if torch.isnan(vq_loss) or torch.isinf(vq_loss):
            vq_loss = (logits * 0).sum()
        
        if codes is not None and self.lambda_entropy > 0:
            entropy_loss = codebook_entropy_loss(codes, self.num_embeddings)
        else:
            entropy_loss = torch.tensor(0.0, device=logits.device)
        
        if recon_pred is not None and recon_target is not None and self.lambda_recon > 0 and not (self.phase2_disable_recon and mode == 'mil_train_phase2'):
            if self.asymmetric_recon and mode == 'mil_train_phase2':
                neg_mask_recon = (labels == 0)
                if neg_mask_recon.any():
                    rp = recon_pred[neg_mask_recon]
                    rt = recon_target[neg_mask_recon]
                    if mask is not None:
                        m = mask[neg_mask_recon].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                        rp = rp * m
                        rt = rt * m
                        n_elem = m.sum() * rp.shape[2] * rp.shape[3] * rp.shape[4] * rp.shape[5]
                        if self.recon_type == 'mse':
                            recon_loss = ((rp - rt) ** 2).sum() / n_elem.clamp(min=1)
                        else:
                            recon_loss = (rp - rt).abs().sum() / n_elem.clamp(min=1)
                    else:
                        if self.recon_type == 'mse':
                            recon_loss = F.mse_loss(rp, rt)
                        else:
                            recon_loss = F.l1_loss(rp, rt)
                else:
                    recon_loss = torch.tensor(0.0, device=logits.device)
            else:
                if self.recon_type == 'mse':
                    recon_loss = F.mse_loss(recon_pred, recon_target)
                else:
                    recon_loss = F.l1_loss(recon_pred, recon_target)
            recon_loss = torch.clamp(recon_loss, 0, 10)
        else:
            recon_loss = torch.tensor(0.0, device=logits.device)

        if (codes is not None and healthy_code_mask is not None
                and self.lambda_code_att > 0 and healthy_code_mask.any()):
            code_att_loss = code_guided_attention_loss(
                attention=attention,
                codes=codes,
                labels=labels,
                healthy_code_mask=healthy_code_mask,
                mask=mask,
                cancer_weight=self.code_att_cancer_weight,
                healthy_weight=self.code_att_healthy_weight,
            )
        else:
            code_att_loss = torch.tensor(0.0, device=logits.device)

        use_phase2 = mode == 'mil_train_phase2'

        if (use_phase2 and self.lambda_gt_code > 0
                and z_e is not None and gt_patch_labels is not None
                and codebook_embedding is not None
                and healthy_code_mask is not None
                and (~healthy_code_mask).any() and healthy_code_mask.any()):
            gt_code_loss = gt_guided_code_loss(
                z_e=z_e,
                gt_patch_labels=gt_patch_labels,
                codebook_embedding=codebook_embedding,
                healthy_code_mask=healthy_code_mask,
                mask=mask,
            )
        else:
            gt_code_loss = torch.tensor(0.0, device=logits.device)

        if (self.lambda_patch_seg > 0
                and patch_seg_logits is not None
                and gt_patch_labels is not None):
            if self.seg_loss_type == 'focal_dice_bce':
                ps_loss = patch_seg_focal_dice_bce_loss(
                    patch_seg_logits, gt_patch_labels,
                    mask=mask, pos_weight=self.patch_seg_pos_weight,
                    dice_weight=self.seg_dice_weight, bce_weight=self.seg_bce_weight,
                )
            elif self.seg_loss_type == 'dice_bce':
                ps_loss = patch_seg_dice_bce_loss(
                    patch_seg_logits, gt_patch_labels,
                    mask=mask, pos_weight=self.patch_seg_pos_weight,
                    dice_weight=self.seg_dice_weight, bce_weight=self.seg_bce_weight,
                )
            else:
                ps_loss = patch_seg_bce_loss(
                    patch_seg_logits, gt_patch_labels,
                    mask=mask, pos_weight=self.patch_seg_pos_weight,
                )
        else:
            ps_loss = torch.tensor(0.0, device=logits.device)

        if (self.lambda_voxel_seg > 0
                and voxel_seg_logits is not None
                and gt_voxel_masks is not None):
            voxel_loss_type = getattr(self, 'voxel_loss_type', 'dice')
            use_ds = getattr(self, 'voxel_deep_supervision', False)
            ds_weights = getattr(self, 'voxel_ds_weights', [0.25, 0.25])
            pos_patch_mask = None
            if gt_patch_labels is not None and mask is not None:
                pos_patch_mask = (gt_patch_labels > 0).float() * mask
            elif gt_patch_labels is not None:
                pos_patch_mask = (gt_patch_labels > 0).float()
            else:
                pos_patch_mask = mask
            if pos_patch_mask is not None and pos_patch_mask.sum() == 0:
                vs_loss = torch.tensor(0.0, device=logits.device)
            elif use_ds and voxel_aux_outputs is not None:
                vs_loss = voxel_seg_deep_supervision_loss(
                    voxel_seg_logits, voxel_aux_outputs, gt_voxel_masks,
                    patch_mask=pos_patch_mask, ds_weights=ds_weights,
                    loss_type=voxel_loss_type,
                )
            elif voxel_loss_type == 'focal_dice_bce':
                vs_loss = voxel_seg_focal_dice_bce_loss(
                    voxel_seg_logits, gt_voxel_masks, patch_mask=pos_patch_mask,
                    dice_weight=self.seg_dice_weight, bce_weight=self.seg_bce_weight,
                    pos_weight=self.patch_seg_pos_weight
                )
            elif voxel_loss_type == 'dice_bce':
                vs_loss = voxel_seg_dice_bce_loss(voxel_seg_logits, gt_voxel_masks, patch_mask=pos_patch_mask)
            else:
                vs_loss = voxel_seg_dice_loss(voxel_seg_logits, gt_voxel_masks, patch_mask=pos_patch_mask)
        else:
            vs_loss = torch.tensor(0.0, device=logits.device)

        use_seg_prior = use_phase2
        if use_seg_prior and (
            self.lambda_seg_neg > 0 or self.lambda_seg_pos > 0 or self.lambda_seg_sep > 0
        ):
            seg_neg_loss, seg_pos_loss, seg_sep_loss = segmentation_prior_loss(
                codes=codes,
                labels=labels,
                healthy_code_mask=healthy_code_mask,
                mask=mask,
                pos_ratio_min=self.seg_pos_ratio_min,
                pos_ratio_max=self.seg_pos_ratio_max,
                sep_margin=self.seg_sep_margin,
            )
        else:
            seg_neg_loss = torch.tensor(0.0, device=logits.device)
            seg_pos_loss = torch.tensor(0.0, device=logits.device)
            seg_sep_loss = torch.tensor(0.0, device=logits.device)

        if (use_phase2 and self.lambda_spatial_smooth > 0
                and patch_seg_logits is not None
                and sorted_coords is not None):
            _ps = patch_size if patch_size is not None else (32, 32, 32)
            smooth_loss = spatial_smoothness_loss(
                patch_seg_logits, sorted_coords, mask=mask, patch_size=_ps
            )
        else:
            smooth_loss = torch.tensor(0.0, device=logits.device)

        code_cls_loss = torch.tensor(0.0, device=logits.device)
        if self.lambda_code_cls > 0 and code_cls_logit is not None and mode != 'pretrain_healthy_vqvae':
            code_cls_loss = F.binary_cross_entropy_with_logits(code_cls_logit, labels.float())

        if embed_sep_loss is None:
            embed_sep_loss = torch.tensor(0.0, device=logits.device)

        if pseudo_routing_loss is None:
            pseudo_routing_loss = torch.tensor(0.0, device=logits.device)

        if self.lambda_mil_seg > 0 and mil_seg_logits is not None and mode != 'pretrain_healthy_vqvae':
            ms_loss = mil_seg_loss(mil_seg_logits, labels, mask, smooth_r=self.mil_seg_smooth_r)
        else:
            ms_loss = torch.tensor(0.0, device=logits.device)

        if self.lambda_att_sparsity > 0 and mode == 'mil_train_phase2':
            att_sparse_loss = attention_sparsity_loss(attention, labels, mask)
        else:
            att_sparse_loss = torch.tensor(0.0, device=logits.device)

        if self.lambda_att_uniformity > 0 and mode == 'mil_train_phase2':
            att_uniform_loss = attention_uniformity_loss(attention, labels, mask)
        else:
            att_uniform_loss = torch.tensor(0.0, device=logits.device)

        if (self.lambda_cancer_logit_reg > 0 and mode == 'mil_train_phase2'
                and cancer_logit is not None
                and normal_patch_count is not None
                and cancer_patch_count is not None):
            cl_reg_loss = cancer_logit_enrichment_reg(
                cancer_logit, normal_patch_count, cancer_patch_count
            )
        else:
            cl_reg_loss = torch.tensor(0.0, device=logits.device)

        if (self.lambda_recon_teacher > 0 and mode == 'mil_train_phase2'
                and mil_seg_logits is not None
                and recon_pred is not None and recon_target is not None):
            rt_loss = recon_teacher_loss(
                mil_seg_logits, recon_pred, recon_target, labels, mask,
                top_k_ratio=self.recon_teacher_top_k,
            )
        else:
            rt_loss = torch.tensor(0.0, device=logits.device)

        if (self.lambda_feat_distill > 0 and mode == 'mil_train_phase2'
                and distill_feat is not None and z_e_frozen is not None):
            fd_loss = feature_distillation_loss(distill_feat, z_e_frozen, mask)
        else:
            fd_loss = torch.tensor(0.0, device=logits.device)

        if mode == 'pretrain_healthy_vqvae':
            total_loss = (
                self.lambda_vq * torch.clamp(vq_loss, 0, 10) +
                self.lambda_entropy * entropy_loss +
                self.lambda_recon * recon_loss
            )
        else:
            total_loss = (
                self.lambda_cls * cls_loss +
                self.lambda_nu * nu_loss +
                self.lambda_vq * torch.clamp(vq_loss, 0, 10) +
                self.lambda_temporal * temporal_loss +
                self.lambda_entropy * entropy_loss +
                self.lambda_recon * self.recon_in_mil_weight * recon_loss +
                self.lambda_code_att * code_att_loss +
                self.lambda_gt_code * gt_code_loss +
                self.lambda_patch_seg * ps_loss +
                self.lambda_voxel_seg * vs_loss +
                self.lambda_seg_neg * seg_neg_loss +
                self.lambda_seg_pos * seg_pos_loss +
                self.lambda_seg_sep * seg_sep_loss +
                self.lambda_spatial_smooth * smooth_loss +
                self.lambda_code_cls * code_cls_loss +
                self.lambda_embed_sep * embed_sep_loss +
                self.lambda_pseudo_routing * pseudo_routing_loss +
                self.lambda_mil_seg * ms_loss +
                self.lambda_att_sparsity * att_sparse_loss +
                self.lambda_att_uniformity * att_uniform_loss +
                self.lambda_cancer_logit_reg * cl_reg_loss +
                self.lambda_recon_teacher * rt_loss +
                self.lambda_feat_distill * fd_loss
            )
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = (logits * 0).sum()
        
        loss_dict = {
            'total': total_loss.item(),
            'cls': cls_loss.item(),
            'nu': nu_loss.item() if isinstance(nu_loss, torch.Tensor) else nu_loss,
            'vq': vq_loss.item() if isinstance(vq_loss, torch.Tensor) else vq_loss,
            'temporal': temporal_loss.item(),
            'entropy': entropy_loss.item() if isinstance(entropy_loss, torch.Tensor) else entropy_loss,
            'recon': recon_loss.item() if isinstance(recon_loss, torch.Tensor) else recon_loss,
            'code_att': code_att_loss.item() if isinstance(code_att_loss, torch.Tensor) else code_att_loss,
            'seg_neg': seg_neg_loss.item() if isinstance(seg_neg_loss, torch.Tensor) else seg_neg_loss,
            'seg_pos': seg_pos_loss.item() if isinstance(seg_pos_loss, torch.Tensor) else seg_pos_loss,
            'seg_sep': seg_sep_loss.item() if isinstance(seg_sep_loss, torch.Tensor) else seg_sep_loss,
            'gt_code': gt_code_loss.item() if isinstance(gt_code_loss, torch.Tensor) else gt_code_loss,
            'patch_seg': ps_loss.item() if isinstance(ps_loss, torch.Tensor) else ps_loss,
            'voxel_seg': vs_loss.item() if isinstance(vs_loss, torch.Tensor) else vs_loss,
            'spatial_smooth': smooth_loss.item() if isinstance(smooth_loss, torch.Tensor) else smooth_loss,
            'code_cls': code_cls_loss.item() if isinstance(code_cls_loss, torch.Tensor) else code_cls_loss,
            'embed_sep': embed_sep_loss.item() if isinstance(embed_sep_loss, torch.Tensor) else embed_sep_loss,
            'pseudo_routing': pseudo_routing_loss.item() if isinstance(pseudo_routing_loss, torch.Tensor) else pseudo_routing_loss,
            'mil_seg': ms_loss.item() if isinstance(ms_loss, torch.Tensor) else ms_loss,
            'att_sparse': att_sparse_loss.item() if isinstance(att_sparse_loss, torch.Tensor) else att_sparse_loss,
            'att_uniform': att_uniform_loss.item() if isinstance(att_uniform_loss, torch.Tensor) else att_uniform_loss,
            'cl_reg': cl_reg_loss.item() if isinstance(cl_reg_loss, torch.Tensor) else cl_reg_loss,
            'rt_loss': rt_loss.item() if isinstance(rt_loss, torch.Tensor) else rt_loss,
            'feat_distill': fd_loss.item() if isinstance(fd_loss, torch.Tensor) else fd_loss,
        }
        
        return total_loss, loss_dict
