import argparse
import logging
import os
import sys
from datetime import timedelta
from contextlib import nullcontext

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
from src.training.losses import CombinedLoss
from src.models.feature_extractor_3d import VolumetricFeatureExtractor
from src.models.vector_quantizer_3d import PartitionedVectorQuantizer
from src.models.spatial_scanner_3d import ZOrderSpatialScanner
from src.models.hilbert_scanner import HilbertCurveSpatialScanner
from src.models.video_mamba import VideoMamba3D
from src.models.mil_head import AttentionMILHead
from src.models.self_correction import SelfCorrectionModule
from src.models.decoder_3d import VolumetricDecoder3D
from src.data.volumetric_loader import create_volumetric_dataloader
from src.utils.metrics import evaluate_bag_level


class Volumetric3DMIL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.codebook_after_mamba = config.model.get('codebook_after_mamba', False)
        
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
            explore_prob=config.model['codebook'].get('explore_prob', 0.0),
            gate_std_factor=config.model['codebook'].get('gate_std_factor', 1.5),
            min_cancer_fraction=config.model['codebook'].get('min_cancer_fraction', 0.1),
            lambda_code_nu=config.model['codebook'].get('lambda_code_nu', 0.5)
        )
        self.soft_cancer_temperature = config.training.get('soft_cancer_temperature', 1.0)
        self.embed_sep_margin = float(config.loss.get('embed_sep_margin', 2.0))
        self.mil_seg_proj = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.frozen_feature_extractor = None
        self.pseudo_top_k_ratio = float(config.training.get('pseudo_top_k_ratio', 0.15))
        self.pseudo_routing_margin = float(config.training.get('pseudo_routing_margin', 0.5))
        
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

        seg_head_config = config.model.get('seg_head', {})
        self.seg_head_type = seg_head_config.get('type', 'simple')
        self.seg_use_multiscale = seg_head_config.get('use_multiscale', False)

        if self.seg_use_multiscale:
            ms_dims = self.feature_extractor.get_multiscale_dims()
            seg_feat_dim = sum(ms_dims.values())
        else:
            seg_feat_dim = feature_dim

        if self.seg_head_type == 'dual_path':
            seg_input_dim = seg_feat_dim + feature_dim
            self.patch_seg_head = nn.Sequential(
                nn.Linear(seg_input_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        else:
            self.patch_seg_head = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

        voxel_seg_config = config.model.get('voxel_seg', {})
        self.voxel_seg_enabled = voxel_seg_config.get('enabled', False)
        self.voxel_use_skip = False
        self.use_spatial_decoder = voxel_seg_config.get('decoder_type', 'mlp') == 'spatial_unet'
        if self.voxel_seg_enabled:
            if self.use_spatial_decoder:
                from src.models.patch_voxel_decoder import SpatialUNetDecoder
                self.spatial_decoder = SpatialUNetDecoder(
                    patch_size=tuple(config.data['patch_size']),
                )
            else:
                from src.models.patch_voxel_decoder import PatchVoxelDecoder
                voxel_input_dim = seg_feat_dim + feature_dim if self.seg_head_type == 'dual_path' else feature_dim
                self.voxel_decoder = PatchVoxelDecoder(
                    input_dim=voxel_input_dim,
                    patch_size=tuple(config.data['patch_size']),
                    hidden_dims=voxel_seg_config.get('hidden_dims', [256, 128, 64]),
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
        if self.voxel_seg_enabled and self.use_spatial_decoder and self.seg_use_multiscale:
            multi_features, features, voxel_seg_logits = self._forward_with_spatial_decoder(
                patches, mini_batch_size)
        elif self.seg_use_multiscale:
            multi_features, features = self.feature_extractor.forward_multiscale(
                patches, mini_batch_size=mini_batch_size)
            voxel_seg_logits = None
        else:
            features = self.feature_extractor(patches, mini_batch_size=mini_batch_size)
            multi_features = None
            voxel_seg_logits = None

        if self.codebook_after_mamba:
            sorted_features, sorted_coords, sort_perm = self.spatial_scanner(features, coords)
            mamba_out = self.mamba(sorted_features, mask=masks)

            inv_perm = sort_perm.argsort(dim=1)
            B, N, D = mamba_out.shape
            context_orig = torch.gather(
                mamba_out, 1,
                inv_perm.unsqueeze(-1).expand(B, N, D)
            )

            z_e = context_orig
            quantized, codes, vq_loss = self.codebook(context_orig, labels)
            logits, attention = self.mil_head(quantized, mask=masks)

            correction_rate = torch.tensor(0.0, device=logits.device)
            if self.use_self_correction and self.training:
                corrected_codes, corrected_attention, correction_rate = self.self_correction(
                    codes=codes, coords=coords, attention=attention,
                    codebook_embeddings=self.codebook.embedding.weight, mask=masks)
                codes = corrected_codes
                attention = corrected_attention

            recon_patches = None
            if self.decoder_enabled:
                recon_patches = self.decoder(quantized)

            if self.seg_head_type == 'dual_path':
                context_detached = context_orig.detach()
                if self.seg_use_multiscale and multi_features is not None:
                    ms_cat = torch.cat(list(multi_features.values()), dim=-1)
                    seg_input = torch.cat([ms_cat, context_detached], dim=-1)
                else:
                    seg_input = torch.cat([features.detach(), context_detached], dim=-1)
            else:
                seg_input = context_orig.detach()
        else:
            z_e = features
            quantized, codes, vq_loss = self.codebook(features, labels)

            sorted_quantized, sorted_coords, sort_perm = self.spatial_scanner(quantized, coords)
            context = self.mamba(sorted_quantized, mask=masks)

            logits, attention = self.mil_head(context, mask=masks)

            correction_rate = torch.tensor(0.0, device=logits.device)
            if self.use_self_correction and self.training:
                corrected_codes, corrected_attention, correction_rate = self.self_correction(
                    codes=codes, coords=coords, attention=attention,
                    codebook_embeddings=self.codebook.embedding.weight, mask=masks)
                codes = corrected_codes
                attention = corrected_attention

            recon_patches = None
            if self.decoder_enabled:
                recon_patches = self.decoder(quantized)

            inv_perm = sort_perm.argsort(dim=1)
            B, N, D = context.shape
            context_orig = torch.gather(
                context, 1,
                inv_perm.unsqueeze(-1).expand(B, N, D)
            )

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
            if not self.use_spatial_decoder:
                voxel_seg_logits = self.voxel_decoder(seg_input)
            patch_seg_logits = voxel_seg_logits.mean(dim=(2, 3, 4)).unsqueeze(-1)
        else:
            patch_seg_logits = self.patch_seg_head(seg_input)

        patch_cancer_logits = self.codebook.compute_soft_cancer_logits(
            z_e, self.soft_cancer_temperature)
        if masks is not None:
            att_weights = attention.detach()
            weighted_logits = patch_cancer_logits * att_weights * masks.float()
            code_cls_logit = weighted_logits.sum(dim=1)
        else:
            code_cls_logit = None

        embed_sep_loss = self.codebook.compute_embedding_separation_loss(
            margin=self.embed_sep_margin)

        if self.training and labels is not None:
            pseudo_routing_loss = self.codebook.compute_pseudo_routing_loss(
                z_e, labels, masks, attention,
                top_k_ratio=self.pseudo_top_k_ratio,
                margin=self.pseudo_routing_margin)
        else:
            pseudo_routing_loss = torch.tensor(0.0, device=logits.device)

        mil_seg_logits = self.mil_seg_proj(z_e).squeeze(-1)

        z_e_frozen = None
        if self.training and self.frozen_feature_extractor is not None:
            with torch.no_grad():
                z_e_frozen = self.frozen_feature_extractor(patches, mini_batch_size=mini_batch_size)

        return logits, attention, vq_loss, codes, sorted_coords, correction_rate, recon_patches, context_orig if self.training else None, patch_seg_logits, voxel_seg_logits, code_cls_logit, embed_sep_loss, pseudo_routing_loss, patch_cancer_logits, mil_seg_logits, z_e_frozen

    def _forward_with_spatial_decoder(self, patches, mini_batch_size):
        B, N, C, D_p, H_p, W_p = patches.shape
        enc = self.feature_extractor.encoder
        encoder_device = next(enc.parameters()).device
        frozen = self.feature_extractor.frozen

        if self.training and frozen:
            enc.eval()

        all_l2, all_l3, all_l4 = [], [], []
        all_voxel = []

        for b in range(B):
            sample_patches = patches[b]
            b_l2, b_l3, b_l4, b_vox = [], [], [], []

            for i in range(0, N, mini_batch_size):
                end_idx = min(i + mini_batch_size, N)
                mb = sample_patches[i:end_idx].to(encoder_device, non_blocking=True)

                with torch.set_grad_enabled(not frozen):
                    h = enc.act(enc.bn1(enc.conv1(mb)))
                    h = enc.maxpool(h)
                    h1 = enc.layer1(h)
                    h2 = enc.layer2(h1)
                    h3 = enc.layer3(h2)
                    h4 = enc.layer4(h3)

                    f2 = F.adaptive_avg_pool3d(h2, 1).view(h2.size(0), -1)
                    f3 = F.adaptive_avg_pool3d(h3, 1).view(h3.size(0), -1)
                    f4 = F.adaptive_avg_pool3d(h4, 1).view(h4.size(0), -1)

                vox = self.spatial_decoder(h1, h2, h3, h4)

                b_l2.append(f2)
                b_l3.append(f3)
                b_l4.append(f4)
                b_vox.append(vox)

                del mb, h, h1, h2, h3, h4

            all_l2.append(torch.cat(b_l2, 0))
            all_l3.append(torch.cat(b_l3, 0))
            all_l4.append(torch.cat(b_l4, 0))
            all_voxel.append(torch.cat(b_vox, 0))

        multi_features = {
            'layer2': torch.stack(all_l2, 0),
            'layer3': torch.stack(all_l3, 0),
            'layer4': torch.stack(all_l4, 0),
        }
        features = multi_features['layer4']
        voxel_seg_logits = torch.stack(all_voxel, 0)

        return multi_features, features, voxel_seg_logits


def setup_ddp():
    dist.init_process_group(backend='nccl', timeout=timedelta(hours=2))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def cleanup_ddp():
    dist.destroy_process_group()


def train_one_epoch(model, dataloader, optimizer, loss_fn, epoch, device, rank, config,
                    train_mode='mil_train'):
    model.train()
    total_loss = 0.0
    num_batches = 0

    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    else:
        pbar = dataloader

    mini_batch_size = config.model['feature_extractor'].get('mini_batch_size', 8)
    patch_size = tuple(config.data['patch_size'])

    use_amp = config.training.get('mixed_precision', False)
    scaler = config.training.get('_grad_scaler')

    oom_skip_count = 0
    for batch_idx, batch in enumerate(pbar):
        patches = batch['patches']
        coords = batch['coords'].to(device)
        labels = batch['labels'].to(device)
        masks = batch['masks'].to(device)
        optimizer.zero_grad(set_to_none=True)

        try:
            gt_patch_labels_tensor = None
            gt_voxel_masks_tensor = None
            need_gt = train_mode == 'mil_train_phase2' and (
                loss_fn.lambda_gt_code > 0 or loss_fn.lambda_patch_seg > 0 or loss_fn.lambda_voxel_seg > 0
            )
            if need_gt:
                B, N = coords.shape[:2]
                coords_cpu = coords.detach().cpu()
                masks_cpu  = masks.detach().cpu()
                gt_labels_np = np.full((B, N), -1.0, dtype=np.float32)
                model_module_tmp = model.module if hasattr(model, 'module') else model
                need_voxel_gt = loss_fn.lambda_voxel_seg > 0 and model_module_tmp.voxel_seg_enabled
                if need_voxel_gt:
                    ph, pw, pd = patch_size[1], patch_size[2], patch_size[0]
                    voxel_gt_np = np.zeros((B, N, pd, ph, pw), dtype=np.float32)
                volume_paths = batch.get('volume_paths', None)
                if volume_paths is not None:
                    for b in range(B):
                        vpath = volume_paths[b]
                        gp = get_gt_mask_path(vpath)
                        gt_mask_hwz = load_gt_mask_hwz(gp) if gp else None
                        if gt_mask_hwz is None:
                            continue
                        valid_n = int(masks_cpu[b].sum().item())
                        if valid_n <= 0:
                            continue
                        coords_b = coords_cpu[b, :valid_n].numpy()
                        patch_labels = compute_patch_positive_labels(
                            coords_b, gt_mask_hwz, patch_size
                        )
                        gt_labels_np[b, :valid_n] = patch_labels
                        if need_voxel_gt:
                            h_vol, w_vol, d_vol = gt_mask_hwz.shape
                            for i, (z, y, x) in enumerate(coords_b):
                                z_s, y_s, x_s = int(z), int(y), int(x)
                                z_e = min(z_s + pd, d_vol)
                                y_e = min(y_s + ph, h_vol)
                                x_e = min(x_s + pw, w_vol)
                                crop = gt_mask_hwz[y_s:y_e, x_s:x_e, z_s:z_e]
                                voxel_gt_np[b, i, :crop.shape[2], :crop.shape[0], :crop.shape[1]] = crop.transpose(2, 0, 1)
                gt_patch_labels_tensor = torch.from_numpy(
                    gt_labels_np
                ).to(device, non_blocking=True)
                if need_voxel_gt:
                    gt_voxel_masks_tensor = torch.from_numpy(voxel_gt_np).to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits, attention, vq_loss, codes, sorted_coords, correction_rate, recon_patches, z_e, patch_seg_logits, voxel_seg_logits, code_cls_logit, embed_sep_loss, pseudo_routing_loss, _, mil_seg_logits, z_e_frozen = model(
                    patches, coords, labels, masks, mini_batch_size=mini_batch_size
                )

                recon_target = patches.to(device, non_blocking=True) if recon_patches is not None else None
                model_module = model.module if hasattr(model, 'module') else model

                loss, loss_dict = loss_fn(
                    logits=logits,
                    labels=labels,
                    attention=attention,
                    vq_loss=vq_loss,
                    mask=masks,
                    codes=codes,
                    recon_pred=recon_patches,
                    recon_target=recon_target,
                    mode=train_mode,
                    healthy_code_mask=model_module.codebook.healthy_code_mask.detach(),
                    z_e=z_e,
                    gt_patch_labels=gt_patch_labels_tensor,
                    codebook_embedding=model_module.codebook.embedding.weight,
                    patch_seg_logits=patch_seg_logits,
                    voxel_seg_logits=voxel_seg_logits,
                    gt_voxel_masks=gt_voxel_masks_tensor,
                    code_cls_logit=code_cls_logit,
                    embed_sep_loss=embed_sep_loss,
                    pseudo_routing_loss=pseudo_routing_loss,
                    mil_seg_logits=mil_seg_logits,
                    cancer_logit=model_module.codebook.cancer_logit,
                    normal_patch_count=model_module.codebook.normal_patch_count,
                    cancer_patch_count=model_module.codebook.cancer_patch_count,
                    z_e_frozen=z_e_frozen,
                )

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if 'clip_grad_norm' in config.training:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.training['clip_grad_norm']
                )

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if rank == 0 and batch_idx % config.logging['print_freq'] == 0:
                seg_prior_sum = loss_dict['seg_neg'] + loss_dict['seg_pos'] + loss_dict['seg_sep']
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'cls': f"{loss_dict['cls']:.4f}",
                    'vq': f"{loss_dict['vq']:.6f}",
                    'rec': f"{loss_dict['recon']:.4f}",
                    'ccls': f"{loss_dict.get('code_cls', 0):.4f}",
                    'esep': f"{loss_dict.get('embed_sep', 0):.4f}",
                    'mseg': f"{loss_dict.get('mil_seg', 0):.4f}",
                    'sp': f"{seg_prior_sum:.4f}",
                    'rt': f"{loss_dict.get('rt_loss', 0):.4f}",
                    'fd': f"{loss_dict.get('feat_distill', 0):.4f}",
                })

        except torch.cuda.OutOfMemoryError:
            oom_skip_count += 1
            if rank == 0:
                logging.warning(f"OOM at batch {batch_idx}, skipping (total skipped: {oom_skip_count})")
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
                if rank == 0:
                    logging.warning(f"OOM during dummy backward too, skipping batch (other ranks may hang)")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            continue

        del logits, attention, vq_loss, codes, sorted_coords, patches, coords, labels, masks, recon_patches, z_e, patch_seg_logits, voxel_seg_logits, gt_voxel_masks_tensor, mil_seg_logits, z_e_frozen
        torch.cuda.empty_cache()
    
    if rank == 0 and oom_skip_count > 0:
        logging.warning(f"Epoch {epoch+1}: {oom_skip_count} batches skipped due to OOM")

    if num_batches == 0:
        return 0.0
    return total_loss / num_batches


def set_trainable_for_pretrain(model_module, pretrain_enabled):
    for p in model_module.mamba.parameters():
        p.requires_grad = not pretrain_enabled
    for p in model_module.mil_head.parameters():
        p.requires_grad = not pretrain_enabled
    if hasattr(model_module, 'self_correction'):
        for p in model_module.self_correction.parameters():
            p.requires_grad = not pretrain_enabled
    if hasattr(model_module, 'mil_seg_proj'):
        for p in model_module.mil_seg_proj.parameters():
            p.requires_grad = not pretrain_enabled


def rebuild_optimizer(model, config):
    import copy
    model_module = model.module if hasattr(model, 'module') else model
    cancer_logit_params = [model_module.codebook.cancer_logit]
    mil_seg_params = list(model_module.mil_seg_proj.parameters())
    mil_seg_ids = {id(p) for p in mil_seg_params}
    cancer_logit_ids = {id(p) for p in cancer_logit_params}
    encoder_params = [p for p in model_module.feature_extractor.parameters() if p.requires_grad]
    encoder_ids = {id(p) for p in encoder_params}
    other_params = [p for n, p in model.named_parameters()
                    if p.requires_grad
                    and id(p) not in cancer_logit_ids
                    and id(p) not in mil_seg_ids
                    and id(p) not in encoder_ids]
    cancer_logit_lr = config.training.get('cancer_logit_lr', 0.01)
    mil_seg_lr = config.training.get('mil_seg_lr', 0.001)
    encoder_lr = config.training.get('encoder_lr', config.training['lr'])
    param_groups = [
        {'params': other_params, 'lr': config.training['lr'],
         'weight_decay': config.training['weight_decay']},
        {'params': cancer_logit_params, 'lr': cancer_logit_lr,
         'weight_decay': 0.0},
        {'params': mil_seg_params, 'lr': mil_seg_lr,
         'weight_decay': config.training['weight_decay']},
    ]
    if encoder_params:
        param_groups.append(
            {'params': encoder_params, 'lr': encoder_lr,
             'weight_decay': config.training['weight_decay']})
    return torch.optim.AdamW(param_groups)


def sync_codebook_state(model_module):
    dist.broadcast(model_module.codebook.embedding.weight.data, src=0)
    if model_module.codebook.use_ema:
        dist.broadcast(model_module.codebook.ema_w, src=0)
        dist.broadcast(model_module.codebook.ema_cluster_size, src=0)
    dist.broadcast(model_module.codebook.healthy_code_mask, src=0)
    dist.broadcast(model_module.codebook.frozen_code_mask, src=0)
    dist.broadcast(model_module.codebook.healthy_distance_threshold, src=0)
    state_tensor = torch.tensor(
        [
            int(model_module.codebook.phase1_complete),
            int(model_module.codebook.use_dynamic_partition),
            int(model_module.codebook.healthy_frozen),
        ],
        device=model_module.codebook.embedding.weight.device,
        dtype=torch.int32
    )
    dist.broadcast(state_tensor, src=0)
    model_module.codebook.phase1_complete = bool(state_tensor[0].item())
    model_module.codebook.use_dynamic_partition = bool(state_tensor[1].item())
    model_module.codebook.healthy_frozen = bool(state_tensor[2].item())


def sync_codebook_usage_stats(model_module):
    dist.all_reduce(model_module.codebook.normal_code_usage, op=dist.ReduceOp.SUM)
    dist.all_reduce(model_module.codebook.cancer_code_usage, op=dist.ReduceOp.SUM)
    if hasattr(model_module.codebook, 'normal_patch_count'):
        dist.all_reduce(model_module.codebook.normal_patch_count, op=dist.ReduceOp.SUM)
    if hasattr(model_module.codebook, 'cancer_patch_count'):
        dist.all_reduce(model_module.codebook.cancer_patch_count, op=dist.ReduceOp.SUM)
    if hasattr(model_module.codebook, '_healthy_dist_count'):
        total_count = model_module.codebook._healthy_dist_count.clone()
        dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
        if total_count.item() > 0:
            weighted_mean = model_module.codebook._healthy_dist_ema_mean * model_module.codebook._healthy_dist_count
            dist.all_reduce(weighted_mean, op=dist.ReduceOp.SUM)
            weighted_var = model_module.codebook._healthy_dist_ema_var * model_module.codebook._healthy_dist_count
            dist.all_reduce(weighted_var, op=dist.ReduceOp.SUM)
            model_module.codebook._healthy_dist_ema_mean = weighted_mean / total_count
            model_module.codebook._healthy_dist_ema_var = weighted_var / total_count
            model_module.codebook._healthy_dist_count = total_count


def resolve_lr_for_epoch(config, epoch_idx: int) -> float:
    current_epoch = epoch_idx + 1
    lr_schedule = config.training.get('lr_schedule', [])
    for stage in lr_schedule:
        start_epoch = int(stage.get('start_epoch', 1))
        end_epoch = int(stage.get('end_epoch', start_epoch))
        if start_epoch <= current_epoch <= end_epoch:
            return float(stage.get('lr', config.training['lr']))
    return float(config.training['lr'])


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


def compute_patch_positive_labels(coords_zyx: np.ndarray, gt_mask_hwz: np.ndarray, patch_size) -> np.ndarray:
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
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--mini_batch_size', type=int, default=None)
    parser.add_argument('--max_patches', type=int, default=None)
    parser.add_argument('--patch_size', type=int, default=None, help='Override patch size (same for D,H,W); e.g. 16 or 8 for 16^3 or 8^3')
    parser.add_argument('--stride', type=int, default=None, help='Override stride (same for D,H,W); default same as patch_size')
    parser.add_argument('--weak_supervision', action='store_true', help='Zero GT mask loss (v20-style); lambda_gt_code, lambda_patch_seg, lambda_voxel_seg=0')
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint path to resume; use --no_resume to train from scratch and ignore config')
    parser.add_argument('--no_resume', action='store_true', help='Do not resume; train from scratch (overrides config checkpoint.resume)')
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
        logging.info(f"Using device: {device}")
        logging.info(f"World size: {world_size}")
        logging.info(f"Rank: {rank}")
    
    config = Config.from_yaml(args.config)
    if args.batch_size is not None:
        config.data['batch_size'] = args.batch_size
    if args.mini_batch_size is not None:
        config.model['feature_extractor']['mini_batch_size'] = args.mini_batch_size
    if args.max_patches is not None:
        config.data['max_patches'] = args.max_patches
    if args.patch_size is not None:
        config.data['patch_size'] = [args.patch_size, args.patch_size, args.patch_size]
        config.data['stride'] = [args.stride or args.patch_size] * 3
    elif args.stride is not None:
        config.data['stride'] = [args.stride, args.stride, args.stride]
    if args.weak_supervision:
        config.loss['lambda_gt_code'] = 0.0
        config.loss['lambda_patch_seg'] = 0.0
        config.loss['lambda_voxel_seg'] = 0.0
    if args.mixed_precision:
        config.training['mixed_precision'] = True
    
    if config.training.get('disable_amp', False):
        config.training['mixed_precision'] = False

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
    
    config.training['_grad_scaler'] = torch.amp.GradScaler('cuda', enabled=config.training.get('mixed_precision', False))
    use_two_phase = config.training.get('two_phase_training', False)
    phase1_epochs = config.training.get('phase1_epochs', 30)
    phase1_normal_ratio = config.training.get('phase1_normal_ratio', 0.8)
    healthy_code_threshold = config.training.get('healthy_code_threshold', 0.7)
    phase1_mode = config.training.get('phase1_mode', 'standard')
    phase1_warmup_epochs = config.training.get('phase1_warmup_epochs', 0)
    use_strict_warmup = use_two_phase and phase1_mode == 'strict_negative_warmup' and phase1_warmup_epochs > 0
    phase2_freeze_healthy_dynamic = config.training.get('phase2_freeze_healthy_dynamic', True)
    pretrain_epochs = config.training.get('pretrain_healthy_vqvae_epochs', 0)
    pretrain_healthy_only = config.training.get('pretrain_healthy_only', True)
    freeze_healthy_after_pretrain = config.training.get('freeze_healthy_codes_after_pretrain', True)
    min_codes_before_freeze = config.training.get('min_codes_before_freeze', 6)
    revive_enabled = config.model.get('codebook', {}).get('revive_dead_codes', True)
    revive_min_epoch_usage = config.model.get('codebook', {}).get('revive_min_epoch_usage', 1.0)
    revive_noise_std = config.model.get('codebook', {}).get('revive_noise_std', 0.01)
    revive_until_epoch = config.model.get('codebook', {}).get('revive_until_epoch', 40)
    fixed_eval_threshold = float(config.training.get('fixed_eval_threshold', 0.5))
    best_model_min_f1_fixed = float(config.training.get('best_model_min_f1_fixed', 0.0))
    best_model_min_specificity = float(config.training.get('best_model_min_specificity', 0.0))
    phase2_cls_ramp_epochs = int(config.training.get('phase2_cls_ramp_epochs', 0))
    phase2_cls_start_factor = float(config.training.get('phase2_cls_start_factor', 1.0))
    phase2_encoder_freeze_epochs = int(config.training.get('phase2_encoder_freeze_epochs', 0))
    phase2_data_transition_epochs = int(config.training.get('phase2_data_transition_epochs', 0))
    phase2_target_normal_ratio = float(config.training.get('phase2_target_normal_ratio', 0.5))
    phase2_seg_prior_warmup_epochs = int(config.training.get('phase2_seg_prior_warmup_epochs', 8))
    phase2_seg_prior_start_factor = float(config.training.get('phase2_seg_prior_start_factor', 0.2))
    seg_training_start_epoch = int(config.training.get('seg_training_start_epoch', 0))
    base_lambda_patch_seg = float(config.loss.get('lambda_patch_seg', 0.0))
    base_lambda_voxel_seg = float(config.loss.get('lambda_voxel_seg', 0.0))
    use_codebook_seg_val = (base_lambda_patch_seg == 0 and base_lambda_voxel_seg == 0)
    checkpoint_save_freq = int(config.checkpoint.get('save_freq', 10))
    val_freq = int(config.training.get('val_freq', 1))
    seg_val_freq = int(config.training.get('seg_val_freq', val_freq))
    seg_val_max_batches = int(config.training.get('seg_val_max_batches', 0))  # 0 = no limit
    val_start_epoch = int(config.training.get('val_start_epoch', 1))
    validation_history_path = config.logging.get(
        'validation_history_path',
        os.path.join(config.logging.get('log_dir', '.'), 'validation_history.txt')
    )
    best_seg_min_auc = float(config.training.get('best_seg_min_auc', 0.0))
    
    if rank == 0:
        logging.info(f"Run name: {runtime_paths['run_name']}")
        logging.info(f"Artifact root: {runtime_paths['output_root']}")
        logging.info(f"Checkpoint dir: {runtime_paths['checkpoint_dir']}")
        logging.info(f"Log dir: {runtime_paths['log_dir']}")
        logging.info(f"Results dir: {runtime_paths['results_dir']}")
        logging.info(f"Patch size: {config.data['patch_size']}, Stride: {config.data['stride']}")
        logging.info(f"Codebook position: {'AFTER mamba (mamba-first)' if config.model.get('codebook_after_mamba', False) else 'BEFORE mamba (default)'}")
        logging.info(f"Batch size per GPU: {config.data['batch_size']}")
        logging.info(f"Feature extractor mini-batch size: {config.model['feature_extractor'].get('mini_batch_size', 8)}")
        logging.info(f"Max patches per volume: {config.data.get('max_patches', 256)}")
        logging.info(f"Weak supervision (no GT mask): lambda_gt_code={config.loss.get('lambda_gt_code', 0)}, lambda_patch_seg={config.loss.get('lambda_patch_seg', 0)}, lambda_voxel_seg={config.loss.get('lambda_voxel_seg', 0)}")
        if use_codebook_seg_val:
            logging.info("Seg validation: using codebook cancer-code assignment (no seg head supervision)")
        logging.info(f"Mixed precision: {config.training.get('mixed_precision', False)}")
        logging.info(f"Decoder enabled: {config.model.get('decoder', {}).get('enabled', False)}")
        logging.info(f"Phase 0 healthy VQ-VAE epochs: {pretrain_epochs}")
        
        if use_two_phase:
            logging.info("="*60)
            logging.info("TWO-PHASE TRAINING ENABLED")
            if use_strict_warmup:
                logging.info(f"Stage A: Epochs 1-{phase1_warmup_epochs} (Negative-only warmup)")
                logging.info(f"Stage B: Epochs {phase1_warmup_epochs+1}-{phase1_epochs} (Codebook Cleaning)")
                logging.info(f"  - Normal ratio: {phase1_normal_ratio:.1%}")
            else:
                logging.info(f"Phase 1: Epochs 1-{phase1_epochs} (Codebook Cleaning)")
                logging.info(f"  - Normal ratio: {phase1_normal_ratio:.1%}")
            logging.info(f"Phase 2: Epochs {phase1_epochs+1}-{config.training['epochs']} (Dynamic Partition)")
            logging.info(f"  - Healthy code threshold: {healthy_code_threshold:.1%}")
            logging.info("="*60)
    
    train_loader = create_volumetric_dataloader(
        json_path=config.data['train_json'],
        batch_size=config.data['batch_size'],
        patch_size=tuple(config.data['patch_size']),
        stride=tuple(config.data['stride']),
        num_workers=config.data['num_workers'],
        shuffle=True,
        balanced_sampling=not use_two_phase and config.data.get('balanced_sampling', True),
        positive_ratio=config.data.get('positive_ratio', 0.2),
        adaptive_norm=config.data.get('adaptive_norm', True),
        max_patches=config.data.get('max_patches', 256),
        use_ddp=True,
        use_phase_aware=use_two_phase,
        is_phase1=True,
        phase1_normal_ratio=phase1_normal_ratio,
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
        max_patches=config.data.get('max_patches', 256),
        use_ddp=False,
        min_hu=config.data['min_hu'],
        max_hu=config.data['max_hu'],
        adaptive_norm=config.data.get('adaptive_norm', True),
    )

    pretrain_loader = None
    if pretrain_epochs > 0:
        pretrain_loader = create_volumetric_dataloader(
            json_path=config.data['train_json'],
            batch_size=config.data['batch_size'],
            patch_size=tuple(config.data['patch_size']),
            stride=tuple(config.data['stride']),
            num_workers=config.data['num_workers'],
            shuffle=True,
            balanced_sampling=False,
            positive_ratio=config.data.get('positive_ratio', 0.2),
            max_patches=config.data.get('max_patches', 256),
            use_ddp=True,
            use_phase_aware=False,
            healthy_only=pretrain_healthy_only,
            min_hu=config.data['min_hu'],
            max_hu=config.data['max_hu'],
            adaptive_norm=config.data.get('adaptive_norm', True),
        )
    
    model = Volumetric3DMIL(config).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    cancer_logit_params = [model.module.codebook.cancer_logit]
    mil_seg_params = list(model.module.mil_seg_proj.parameters())
    mil_seg_ids = {id(p) for p in mil_seg_params}
    cancer_logit_ids = {id(p) for p in cancer_logit_params}
    other_params = [p for n, p in model.named_parameters()
                    if p.requires_grad
                    and id(p) not in cancer_logit_ids
                    and id(p) not in mil_seg_ids]
    cancer_logit_lr = config.training.get('cancer_logit_lr', 0.01)
    mil_seg_lr = config.training.get('mil_seg_lr', 0.001)
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': config.training['lr'],
         'weight_decay': config.training['weight_decay']},
        {'params': cancer_logit_params, 'lr': cancer_logit_lr,
         'weight_decay': 0.0},
        {'params': mil_seg_params, 'lr': mil_seg_lr,
         'weight_decay': config.training['weight_decay']},
    ])
    
    loss_fn = CombinedLoss(
        lambda_cls=config.loss['lambda_cls'],
        lambda_nu=config.loss['lambda_nu'],
        lambda_vq=config.loss['lambda_vq'],
        lambda_temporal=config.loss['lambda_temporal'],
        lambda_entropy=config.loss.get('lambda_entropy', 0.5),
        lambda_recon=config.loss.get('lambda_recon', 0.0),
        recon_in_mil_weight=config.loss.get('recon_in_mil_weight', 0.0),
        recon_type=config.loss.get('recon_type', 'l1'),
        single_class_cls_weight=config.loss.get('single_class_cls_weight', 0.0),
        cls_class_weights=tuple(config.loss['cls_class_weights']) if config.loss.get('cls_class_weights') else None,
        use_focal_cls=config.loss.get('use_focal_cls', False),
        focal_gamma=float(config.loss.get('focal_gamma', 2.0)),
        focal_alpha_pos=float(config.loss.get('focal_alpha_pos', 0.5)),
        label_smoothing=float(config.loss.get('label_smoothing', 0.0)),
        nu_margin=config.loss['nu_margin'],
        num_embeddings=config.loss.get('num_embeddings', 100),
        lambda_code_att=float(config.loss.get('lambda_code_att', 0.0)),
        code_att_cancer_weight=float(config.loss.get('code_att_cancer_weight', 1.0)),
        code_att_healthy_weight=float(config.loss.get('code_att_healthy_weight', 0.01)),
        lambda_seg_neg=float(config.loss.get('lambda_seg_neg', 0.0)),
        lambda_seg_pos=float(config.loss.get('lambda_seg_pos', 0.0)),
        lambda_seg_sep=float(config.loss.get('lambda_seg_sep', 0.0)),
        seg_pos_ratio_min=float(config.loss.get('seg_pos_ratio_min', 0.03)),
        seg_pos_ratio_max=float(config.loss.get('seg_pos_ratio_max', 0.20)),
        seg_sep_margin=float(config.loss.get('seg_sep_margin', 0.08)),
        lambda_gt_code=float(config.loss.get('lambda_gt_code', 0.0)),
        lambda_patch_seg=float(config.loss.get('lambda_patch_seg', 0.0)),
        patch_seg_pos_weight=float(config.loss.get('patch_seg_pos_weight', 5.0)),
        seg_loss_type=config.loss.get('seg_loss_type', 'bce'),
        seg_dice_weight=float(config.loss.get('seg_dice_weight', 1.0)),
        seg_bce_weight=float(config.loss.get('seg_bce_weight', 1.0)),
        lambda_voxel_seg=float(config.loss.get('lambda_voxel_seg', 0.0)),
        lambda_code_cls=float(config.loss.get('lambda_code_cls', 0.0)),
        lambda_embed_sep=float(config.loss.get('lambda_embed_sep', 0.0)),
        lambda_pseudo_routing=float(config.loss.get('lambda_pseudo_routing', 0.0)),
        lambda_mil_seg=float(config.loss.get('lambda_mil_seg', 0.0)),
        mil_seg_smooth_r=float(config.loss.get('mil_seg_smooth_r', 5.0)),
        lambda_att_sparsity=float(config.loss.get('lambda_att_sparsity', 0.0)),
        lambda_att_uniformity=float(config.loss.get('lambda_att_uniformity', 0.0)),
        lambda_cancer_logit_reg=float(config.loss.get('lambda_cancer_logit_reg', 0.0)),
        lambda_recon_teacher=float(config.loss.get('lambda_recon_teacher', 0.0)),
        recon_teacher_top_k=float(config.loss.get('recon_teacher_top_k', 0.2)),
        lambda_feat_distill=float(config.loss.get('lambda_feat_distill', 0.0)),
    )
    if config.training.get('phase2_freeze_recon_pipeline', False):
        loss_fn.phase2_disable_recon = True
    voxel_seg_config = config.model.get('voxel_seg', {})
    loss_fn.voxel_loss_type = voxel_seg_config.get('voxel_loss_type', 'dice')
    loss_fn.voxel_deep_supervision = voxel_seg_config.get('deep_supervision', False)
    loss_fn.voxel_ds_weights = voxel_seg_config.get('ds_weights', [0.25, 0.25])
    base_lambda_gt_code = float(config.loss.get('lambda_gt_code', 0.0))
    base_lambda_cls = float(config.loss['lambda_cls'])
    base_lambda_seg_neg = float(config.loss.get('lambda_seg_neg', 0.0))
    base_lambda_seg_pos = float(config.loss.get('lambda_seg_pos', 0.0))
    base_lambda_seg_sep = float(config.loss.get('lambda_seg_sep', 0.0))
    
    best_auc = 0.0
    best_seg_score = -1.0
    start_epoch = 0
    
    if rank == 0:
        os.makedirs(config.checkpoint['save_dir'], exist_ok=True)
        os.makedirs(os.path.dirname(validation_history_path), exist_ok=True)
        if not os.path.exists(validation_history_path):
            with open(validation_history_path, 'w', encoding='utf-8') as f:
                f.write("epoch\tauc\tacc\tf1\tf1_best\tbest_threshold\tacc_fixed\tf1_fixed\tspec_fixed\tpr_auc\tseg_patch_dice\tseg_patch_dice_pos\tlr\tseg_best_thr\tseg_dice_pos_at_05\tvoxel_dice\n")

    if getattr(args, 'no_resume', False):
        resume_path = None
    elif args.resume is not None and str(args.resume).strip():
        resume_path = resolve_relative_path(args.resume.strip(), runtime_paths['checkpoint_dir'])
    else:
        resume_path = config.checkpoint.get('resume')
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        ckpt_state = checkpoint['model_state_dict']
        model_state = model.module.state_dict()
        filtered_state = {}
        skipped_keys = []
        for k, v in ckpt_state.items():
            if k in model_state and model_state[k].shape != v.shape:
                skipped_keys.append(k)
            else:
                filtered_state[k] = v
        if rank == 0 and skipped_keys:
            logging.info(f"Checkpoint keys skipped (shape mismatch, will init randomly): {skipped_keys}")
        missing, unexpected = model.module.load_state_dict(
            filtered_state, strict=False
        )
        if rank == 0 and missing:
            logging.info(f"Checkpoint missing keys (defaults used): {missing}")
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except (ValueError, RuntimeError):
                if rank == 0:
                    logging.warning("Optimizer state mismatch, skipping optimizer resume (fresh optimizer will be used)")
            cancer_logit_lr = config.training.get('cancer_logit_lr', 0.01)
            mil_seg_lr = config.training.get('mil_seg_lr', 0.001)
            for i, group in enumerate(optimizer.param_groups):
                if i == 0:
                    group['lr'] = config.training['lr']
                    group['weight_decay'] = config.training['weight_decay']
                elif i == 1:
                    group['lr'] = cancer_logit_lr
                    group['weight_decay'] = 0.0
                elif i == 2:
                    group['lr'] = mil_seg_lr
                    group['weight_decay'] = config.training['weight_decay']
        start_epoch = int(checkpoint.get('epoch', -1)) + 1
        ckpt_metrics = checkpoint.get('metrics', {})
        best_auc = float(ckpt_metrics.get('auc', best_auc))
        if (
            model.module.codebook.phase1_complete and
            model.module.codebook.use_dynamic_partition and
            model.module.codebook.healthy_frozen
        ):
            model.module.codebook.unfreeze_healthy_codes()
        dist.barrier(device_ids=[rank])
        sync_codebook_state(model.module)
        phase2_freeze_recon_pipeline = config.training.get('phase2_freeze_recon_pipeline', False)
        if phase2_freeze_recon_pipeline and start_epoch >= phase1_epochs:
            import copy
            phase2_unfreeze_encoder = config.training.get('phase2_unfreeze_encoder', False)
            if phase2_unfreeze_encoder:
                frozen_enc = copy.deepcopy(model.module.feature_extractor)
                for p in frozen_enc.parameters():
                    p.requires_grad = False
                frozen_enc.eval()
                model.module.frozen_feature_extractor = frozen_enc
                for p in model.module.feature_extractor.parameters():
                    p.requires_grad = True
            else:
                for p in model.module.feature_extractor.parameters():
                    p.requires_grad = False
            for p in model.module.decoder.parameters():
                p.requires_grad = False
            model.module.codebook.ema_frozen = True
            optimizer = rebuild_optimizer(model, config)
            if rank == 0:
                if phase2_unfreeze_encoder:
                    enc_lr = config.training.get('encoder_lr', config.training['lr'])
                    logging.info(f"Resume: ENCODER UNFROZEN with distillation (lr={enc_lr})")
                    logging.info("Resume: DECODER + CODEBOOK EMA FROZEN")
                else:
                    logging.info("Resume: RECON PIPELINE FROZEN (encoder + codebook EMA + decoder)")
                logging.info("Resume: Optimizer rebuilt")
        elif not phase2_freeze_recon_pipeline:
            phase2_freeze_encoder = config.training.get('phase2_freeze_encoder', False)
            if phase2_freeze_encoder and start_epoch >= phase1_epochs:
                for p in model.module.feature_extractor.parameters():
                    p.requires_grad = False
                optimizer = rebuild_optimizer(model, config)
                if rank == 0:
                    logging.info("Resume: ENCODER FROZEN for Phase 2 (feature_extractor grad disabled)")
                    logging.info("Resume: Optimizer rebuilt (frozen params excluded)")
        if rank == 0:
            logging.info(f"Resumed from: {resume_path}")
            logging.info(f"Resume epoch index: {start_epoch}")
            logging.info(f"Resume best AUC: {best_auc:.4f}")
            logging.info(f"Resume optimizer override: lr={config.training['lr']}, weight_decay={config.training['weight_decay']}")
            logging.info(f"Resume healthy frozen: {model.module.codebook.healthy_frozen}")

    if pretrain_loader is not None and start_epoch == 0:
        if rank == 0:
            logging.info("=" * 60)
            logging.info("PHASE 0: HEALTHY-ONLY VQ-VAE PRETRAIN")
            logging.info("=" * 60)
        set_trainable_for_pretrain(model.module, pretrain_enabled=True)
        optimizer = rebuild_optimizer(model, config)
        for pre_epoch in range(pretrain_epochs):
            if hasattr(pretrain_loader.sampler, 'set_epoch'):
                pretrain_loader.sampler.set_epoch(pre_epoch)
            if hasattr(pretrain_loader, 'batch_sampler') and hasattr(pretrain_loader.batch_sampler, 'set_epoch'):
                pretrain_loader.batch_sampler.set_epoch(pre_epoch)
            model.module.codebook.reset_epoch_statistics()
            pretrain_loss = train_one_epoch(
                model, pretrain_loader, optimizer, loss_fn, pre_epoch, device, rank, config, train_mode='pretrain_healthy_vqvae'
            )
            revived_codes = 0
            if revive_enabled:
                if rank == 0:
                    revived_codes = model.module.codebook.revive_dead_codes(
                        min_epoch_usage=revive_min_epoch_usage,
                        noise_std=revive_noise_std,
                        use_epoch_stats=True
                    )
                dist.barrier(device_ids=[rank])
                sync_codebook_state(model.module)
            if rank == 0:
                pre_stats = model.module.codebook.get_code_statistics(epoch_only=True)
                logging.info(
                    f"[Phase 0] Epoch {pre_epoch+1}/{pretrain_epochs} Loss={pretrain_loss:.4f} "
                    f"Codes={pre_stats['codes_used']}/{pre_stats['num_embeddings']} Revived={revived_codes}"
                )
        if freeze_healthy_after_pretrain:
            final_pretrain_stats = model.module.codebook.get_code_statistics(epoch_only=True)
            if final_pretrain_stats['codes_used'] >= min_codes_before_freeze:
                model.module.codebook.freeze_healthy_codes(mask_source='initial')
            elif rank == 0:
                logging.info(
                    f"Skip healthy freeze: codes_used={final_pretrain_stats['codes_used']} "
                    f"< min_codes_before_freeze={min_codes_before_freeze}"
                )
        set_trainable_for_pretrain(model.module, pretrain_enabled=False)
        optimizer = rebuild_optimizer(model, config)
        if rank == 0:
            logging.info(f"Healthy code frozen: {model.module.codebook.healthy_frozen}")
            logging.info("=" * 60)
    elif pretrain_loader is not None and start_epoch > 0 and rank == 0:
        logging.info("Skip Phase 0 pretrain because training resumed from checkpoint.")
    
    

    for epoch in range(start_epoch, config.training['epochs']):

        epoch_lr = resolve_lr_for_epoch(config, epoch)
        cancer_logit_lr = config.training.get('cancer_logit_lr', 0.01)
        mil_seg_lr = config.training.get('mil_seg_lr', 0.001)
        for i, group in enumerate(optimizer.param_groups):
            if i == 0:
                group['lr'] = epoch_lr
            elif i == 1:
                group['lr'] = cancer_logit_lr
            elif i == 2:
                group['lr'] = mil_seg_lr
        if use_two_phase and epoch >= phase1_epochs:
            if phase2_seg_prior_warmup_epochs > 0:
                warmup_progress = min(
                    1.0,
                    max(0.0, (epoch - phase1_epochs + 1) / float(phase2_seg_prior_warmup_epochs))
                )
            else:
                warmup_progress = 1.0
            seg_scale = phase2_seg_prior_start_factor + (1.0 - phase2_seg_prior_start_factor) * warmup_progress
        else:
            seg_scale = 0.0
        loss_fn.lambda_seg_neg = base_lambda_seg_neg * seg_scale
        loss_fn.lambda_seg_pos = base_lambda_seg_pos * seg_scale
        loss_fn.lambda_seg_sep = base_lambda_seg_sep * seg_scale
        loss_fn.lambda_gt_code = base_lambda_gt_code * seg_scale
        if seg_training_start_epoch > 0 and (epoch + 1) < seg_training_start_epoch:
            loss_fn.lambda_patch_seg = 0.0
            loss_fn.lambda_voxel_seg = 0.0
        else:
            loss_fn.lambda_patch_seg = base_lambda_patch_seg
            loss_fn.lambda_voxel_seg = base_lambda_voxel_seg
        if rank == 0:
            logging.info("="*50)
            logging.info(f"Epoch {epoch+1}/{config.training['epochs']}")
            logging.info(f"Learning rate: {epoch_lr:.8f}")
            logging.info(f"lambda_cls: {loss_fn.lambda_cls:.4f}")
            logging.info(
                f"seg_prior_scale={seg_scale:.3f}, "
                f"lambda_seg_neg={loss_fn.lambda_seg_neg:.4f}, "
                f"lambda_seg_pos={loss_fn.lambda_seg_pos:.4f}, "
                f"lambda_seg_sep={loss_fn.lambda_seg_sep:.4f}, "
                f"lambda_gt_code={loss_fn.lambda_gt_code:.4f}"
            )
            logging.info("="*50)
        if use_two_phase and hasattr(train_loader, 'batch_sampler'):
            if use_strict_warmup and epoch < phase1_warmup_epochs:
                train_loader.batch_sampler.set_phase('warmup_negative_only')
                if rank == 0:
                    logging.info(f"[Stage A] Negative-only warmup ({epoch+1}/{phase1_warmup_epochs})")
            elif epoch < phase1_epochs:
                train_loader.batch_sampler.set_phase('phase1_mixed')
                if rank == 0:
                    if use_strict_warmup:
                        logging.info(f"[Stage B] Mixed cleaning ({epoch+1-phase1_warmup_epochs}/{phase1_epochs-phase1_warmup_epochs})")
                    else:
                        logging.info("[Phase 1] Mixed cleaning")
            else:
                train_loader.batch_sampler.set_phase('phase2_dynamic')
                if phase2_data_transition_epochs > 0:
                    epochs_in_phase2 = epoch - phase1_epochs
                    if epochs_in_phase2 < phase2_data_transition_epochs:
                        t = epochs_in_phase2 / float(phase2_data_transition_epochs)
                        cur_normal_ratio = phase1_normal_ratio + t * (phase2_target_normal_ratio - phase1_normal_ratio)
                    else:
                        cur_normal_ratio = phase2_target_normal_ratio
                    train_loader.batch_sampler.set_phase2_normal_ratio(cur_normal_ratio)
                    if rank == 0:
                        logging.info(f"Phase 2 data normal ratio: {cur_normal_ratio:.2%}")
        
        if use_two_phase and epoch == phase1_epochs and (not model.module.codebook.phase1_complete):
            dist.barrier(device_ids=[rank])
            if rank == 0:
                logging.info("\n" + "="*60)
                logging.info(">>> SWITCHING TO PHASE 2 <<<")
                logging.info("="*60)

            sync_codebook_usage_stats(model.module)
            if rank == 0:
                model.module.codebook.complete_phase1(threshold=healthy_code_threshold)
                model.module.codebook.enable_dynamic_partition()
                model.module.codebook.unfreeze_healthy_codes()
            if hasattr(train_loader, 'batch_sampler'):
                train_loader.batch_sampler.set_phase('phase2_dynamic')

            phase2_freeze_recon_pipeline = config.training.get('phase2_freeze_recon_pipeline', False)
            if phase2_freeze_recon_pipeline:
                import copy
                phase2_unfreeze_encoder = config.training.get('phase2_unfreeze_encoder', False)
                if phase2_unfreeze_encoder:
                    frozen_enc = copy.deepcopy(model.module.feature_extractor)
                    for p in frozen_enc.parameters():
                        p.requires_grad = False
                    frozen_enc.eval()
                    model.module.frozen_feature_extractor = frozen_enc
                    for p in model.module.feature_extractor.parameters():
                        p.requires_grad = True
                else:
                    for p in model.module.feature_extractor.parameters():
                        p.requires_grad = False
                for p in model.module.decoder.parameters():
                    p.requires_grad = False
                model.module.codebook.ema_frozen = True
                optimizer = rebuild_optimizer(model, config)
                if rank == 0:
                    if phase2_unfreeze_encoder:
                        enc_lr = config.training.get('encoder_lr', config.training['lr'])
                        logging.info(f"Phase 2: ENCODER UNFROZEN with distillation (lr={enc_lr})")
                        logging.info("Phase 2: DECODER + CODEBOOK EMA FROZEN")
                    else:
                        logging.info("Phase 2: RECON PIPELINE FROZEN (encoder + codebook EMA + decoder)")
                    logging.info("Phase 2: Optimizer rebuilt")
            else:
                phase2_freeze_encoder = config.training.get('phase2_freeze_encoder', False)
                if phase2_freeze_encoder:
                    for p in model.module.feature_extractor.parameters():
                        p.requires_grad = False
                    optimizer = rebuild_optimizer(model, config)
                    if rank == 0:
                        logging.info("Phase 2: ENCODER FROZEN (feature_extractor grad disabled)")
                        logging.info("Phase 2: Optimizer rebuilt (frozen params excluded)")

            if rank == 0:
                decoder_snap_path = os.path.join(
                    config.checkpoint['save_dir'], 'decoder_phase1_snapshot.pth')
                torch.save(model.module.decoder.state_dict(), decoder_snap_path)
                logging.info(f"Phase 1 decoder snapshot saved to {decoder_snap_path}")
                logging.info("="*60 + "\n")
            dist.barrier(device_ids=[rank])
            sync_codebook_state(model.module)
            dist.barrier(device_ids=[rank])
        
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'set_epoch'):
            train_loader.batch_sampler.set_epoch(epoch)
        
        model.module.codebook.reset_epoch_statistics()
        
        train_mode = 'mil_train_phase2' if (use_two_phase and epoch >= phase1_epochs) else 'mil_train'
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, epoch, device, rank, config,
            train_mode=train_mode
        )
        revived_codes = 0
        if revive_enabled and epoch < revive_until_epoch:
            if rank == 0:
                revived_codes = model.module.codebook.revive_dead_codes(
                    min_epoch_usage=revive_min_epoch_usage,
                    noise_std=revive_noise_std,
                    use_epoch_stats=True
                )
            dist.barrier(device_ids=[rank])
            sync_codebook_state(model.module)
        
        if rank == 0:
            logging.info(f"Train Loss: {train_loss:.4f}")
            if revive_enabled and epoch < revive_until_epoch:
                logging.info(f"Dead code revived (epoch): {revived_codes}")
            
            epoch_stats = model.module.codebook.get_code_statistics(epoch_only=True)
            cumulative_stats = model.module.codebook.get_code_statistics(epoch_only=False)
            
            if use_two_phase and epoch < phase1_epochs:
                logging.info(f"[Phase 1] Codebook (this epoch): {epoch_stats['codes_used']}/{epoch_stats['num_embeddings']} codes used")
                logging.info(f"[Phase 1] Normal usage: {model.module.codebook.normal_code_usage.sum().item():.0f}, Cancer usage: {model.module.codebook.cancer_code_usage.sum().item():.0f}")
            else:
                logging.info(
                    f"Codebook (Epoch): healthy={epoch_stats['healthy_codes_used']}/{epoch_stats['total_healthy_codes']}, "
                    f"cancer={epoch_stats['cancer_codes_used']}/{epoch_stats['total_cancer_codes']}"
                )
                logging.info(
                    f"Codebook (Cumulative): healthy={cumulative_stats['healthy_codes_used']}/{cumulative_stats['total_healthy_codes']}, "
                    f"cancer={cumulative_stats['cancer_codes_used']}/{cumulative_stats['total_cancer_codes']}"
                )
            
            do_validation = (epoch + 1) >= val_start_epoch and (
                (val_freq <= 1) or ((epoch + 1) % val_freq == 0) or (epoch + 1 == config.training['epochs'])
            )
            do_seg_validation = do_validation and (
                (seg_val_freq <= 1) or ((epoch + 1) % seg_val_freq == 0) or (epoch + 1 == config.training['epochs'])
            )
            if not do_validation:
                logging.info(f"Skipping validation (val_freq={val_freq}, epoch={epoch+1})")
            if do_validation:
                seg_max = seg_val_max_batches if (do_seg_validation and seg_val_max_batches > 0) else 0
                if do_seg_validation and seg_max > 0:
                    seg_label = f" + Seg(first {seg_max} batches)"
                elif do_seg_validation:
                    seg_label = " + Seg"
                else:
                    seg_label = " (cls only)"
                logging.info(f"Running validation{seg_label}...")
                model.module.eval()
                all_preds = []
                all_labels = []
                patch_size = tuple(config.data['patch_size'])
                all_seg_probs = []
                all_seg_probs_milseg = []
                all_seg_probs_cancer = []
                all_seg_probs_combined = []
                all_seg_gt = []
                all_seg_is_pos_bag = []
                seg_batch_count = 0
                val_voxel_tp = val_voxel_fp = val_voxel_fn = 0
                val_voxel_enabled = model.module.voxel_seg_enabled

                mini_batch_size = config.model['feature_extractor'].get('mini_batch_size', 8)

                use_tta = config.training.get('tta_enabled', False)

                with torch.no_grad():
                    for batch in tqdm(val_loader, desc="Validation", leave=False):
                        patches = batch['patches']
                        coords = batch['coords'].to(device)
                        labels = batch['labels'].to(device)
                        masks = batch['masks'].to(device)

                        with torch.amp.autocast('cuda', enabled=config.training.get('mixed_precision', False)):
                            logits, attention_val, _, codes, _, _, recon_patches_val, _, patch_seg_logits, voxel_seg_logits_val, _, _, _, soft_cancer_logits, mil_seg_logits_val, _ = model.module(
                                patches, coords, None, masks, mini_batch_size=mini_batch_size
                            )

                            if use_tta:
                                tta_logits_sum = logits.clone()
                                tta_seg_sum = patch_seg_logits.clone() if patch_seg_logits is not None else None
                                for flip_dims in [[3], [4], [5], [3,4], [3,5], [4,5], [3,4,5]]:
                                    p_flip = torch.flip(patches, dims=flip_dims) if isinstance(patches, torch.Tensor) else patches
                                    l_f, _, _, _, _, _, _, _, seg_f, _, _, _, _, _, _, _ = model.module(
                                        p_flip, coords, None, masks, mini_batch_size=mini_batch_size
                                    )
                                    tta_logits_sum = tta_logits_sum + l_f
                                    if tta_seg_sum is not None and seg_f is not None:
                                        tta_seg_sum = tta_seg_sum + seg_f
                                logits = tta_logits_sum / 8.0
                                if tta_seg_sum is not None:
                                    patch_seg_logits = tta_seg_sum / 8.0

                        all_preds.append(logits)
                        all_labels.append(labels)

                        do_seg_this_batch = (
                            do_seg_validation and
                            (seg_max == 0 or seg_batch_count < seg_max)
                        )
                        if do_seg_this_batch:
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

                                att_b = attention_val[b, :valid_n].float().cpu().numpy()
                                att_min, att_max = att_b.min(), att_b.max()
                                att_norm = (att_b - att_min) / (att_max - att_min + 1e-8)

                                if recon_patches_val is not None:
                                    orig_b = patches[b, :valid_n].float().to(device)
                                    recon_b = recon_patches_val[b, :valid_n].float()
                                    recon_err = (orig_b - recon_b).abs().mean(dim=(1, 2, 3, 4)).cpu().numpy()
                                    re_min, re_max = recon_err.min(), recon_err.max()
                                    recon_norm = (recon_err - re_min) / (re_max - re_min + 1e-8)
                                else:
                                    recon_norm = np.zeros_like(att_norm)

                                coords_b = coords[b, :valid_n].detach().cpu().numpy()
                                gt_labels_b = (compute_patch_positive_labels(coords_b, gt_mask_hwz, patch_size) > 0).astype(np.int8)

                                all_seg_probs.append(recon_norm)
                                all_seg_gt.append(gt_labels_b)
                                all_seg_is_pos_bag.append(int(labels[b].item()) == 1)

                                if mil_seg_logits_val is not None:
                                    ms_b = torch.sigmoid(mil_seg_logits_val[b, :valid_n]).float().cpu().numpy()
                                    all_seg_probs_milseg.append(ms_b)
                                else:
                                    all_seg_probs_milseg.append(np.zeros_like(recon_norm))

                                if soft_cancer_logits is not None:
                                    sc_b = torch.sigmoid(soft_cancer_logits[b, :valid_n]).float().cpu().numpy()
                                    sc_min, sc_max = sc_b.min(), sc_b.max()
                                    sc_norm = (sc_b - sc_min) / (sc_max - sc_min + 1e-8)
                                    all_seg_probs_cancer.append(sc_norm)
                                else:
                                    all_seg_probs_cancer.append(np.zeros_like(recon_norm))

                                ms_val = all_seg_probs_milseg[-1]
                                comb = 0.4 * recon_norm + 0.3 * ms_val + 0.3 * all_seg_probs_cancer[-1]
                                all_seg_probs_combined.append(comb)

                                if not use_codebook_seg_val and val_voxel_enabled and voxel_seg_logits_val is not None and int(labels[b].item()) == 1:
                                    ph, pw, pd = patch_size[1], patch_size[2], patch_size[0]
                                    h_vol, w_vol, d_vol = gt_mask_hwz.shape
                                    vox_pred_b = torch.sigmoid(voxel_seg_logits_val[b, :valid_n]).cpu().numpy()
                                    for pi in range(valid_n):
                                        z_s, y_s, x_s = int(coords_b[pi, 0]), int(coords_b[pi, 1]), int(coords_b[pi, 2])
                                        z_e = min(z_s + pd, d_vol)
                                        y_e = min(y_s + ph, h_vol)
                                        x_e = min(x_s + pw, w_vol)
                                        gt_crop = gt_mask_hwz[y_s:y_e, x_s:x_e, z_s:z_e]
                                        pred_crop = vox_pred_b[pi, :gt_crop.shape[2], :gt_crop.shape[0], :gt_crop.shape[1]]
                                        pred_bin = (pred_crop >= 0.5)
                                        gt_bin = gt_crop.transpose(2, 0, 1).astype(bool)
                                        val_voxel_tp += int((pred_bin & gt_bin).sum())
                                        val_voxel_fp += int((pred_bin & ~gt_bin).sum())
                                        val_voxel_fn += int((~pred_bin & gt_bin).sum())
                
                all_preds = torch.cat(all_preds, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
                
                metrics = evaluate_bag_level(all_preds, all_labels, fixed_threshold=fixed_eval_threshold)

                seg_best_thr = 0.5
                seg_patch_dice_pos_at_05 = -1.0
                if do_seg_validation and len(all_seg_probs) > 0:
                    seg_thr_min = float(config.training.get('seg_thr_search_min', 0.10))
                    seg_thr_max = float(config.training.get('seg_thr_search_max', 0.90))
                    seg_thr_step = float(config.training.get('seg_thr_search_step', 0.02))

                    def _seg_dice_at_thr(probs_list, gt_list, bag_filter, thr):
                        tp = fp = fn = 0
                        for p, g, keep in zip(probs_list, gt_list, bag_filter):
                            if not keep:
                                continue
                            valid = g >= 0
                            pv = p[valid]
                            gv = g[valid].astype(bool)
                            pred = pv >= thr
                            tp += int((pred & gv).sum())
                            fp += int((pred & ~gv).sum())
                            fn += int((~pred & gv).sum())
                        return (2.0 * tp) / (2.0 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

                    all_true = [True] * len(all_seg_probs)
                    thresholds = np.arange(seg_thr_min, seg_thr_max + 1e-9, seg_thr_step)
                    best_dice_pos = 0.0
                    for thr in thresholds:
                        d = _seg_dice_at_thr(all_seg_probs, all_seg_gt, all_seg_is_pos_bag, thr)
                        if d > best_dice_pos:
                            best_dice_pos = d
                            seg_best_thr = float(thr)

                    seg_patch_dice_pos = best_dice_pos
                    seg_patch_dice_pos_at_05 = _seg_dice_at_thr(all_seg_probs, all_seg_gt, all_seg_is_pos_bag, 0.5)
                    seg_patch_dice = _seg_dice_at_thr(all_seg_probs, all_seg_gt, all_true, seg_best_thr)

                    def _best_dice(probs_list):
                        bd = 0.0
                        for thr in thresholds:
                            d = _seg_dice_at_thr(probs_list, all_seg_gt, all_seg_is_pos_bag, thr)
                            if d > bd:
                                bd = d
                        return bd

                    milseg_posdice = _best_dice(all_seg_probs_milseg) if len(all_seg_probs_milseg) == len(all_seg_gt) else -1.0
                    cancer_posdice = _best_dice(all_seg_probs_cancer) if len(all_seg_probs_cancer) == len(all_seg_gt) else -1.0
                    combined_posdice = _best_dice(all_seg_probs_combined) if len(all_seg_probs_combined) == len(all_seg_gt) else -1.0
                else:
                    seg_patch_dice = -1.0
                    seg_patch_dice_pos = -1.0
                    milseg_posdice = -1.0
                    cancer_posdice = -1.0
                    combined_posdice = -1.0
                if val_voxel_enabled and (2 * val_voxel_tp + val_voxel_fp + val_voxel_fn) > 0:
                    voxel_dice = (2.0 * val_voxel_tp) / (2.0 * val_voxel_tp + val_voxel_fp + val_voxel_fn)
                else:
                    voxel_dice = -1.0

                metrics['seg_patch_dice'] = float(seg_patch_dice)
                metrics['seg_patch_dice_pos'] = float(seg_patch_dice_pos)
                metrics['seg_best_thr'] = float(seg_best_thr)
                metrics['seg_dice_pos_at_05'] = float(seg_patch_dice_pos_at_05)
                metrics['voxel_dice'] = float(voxel_dice)
                metrics['milseg_posdice'] = float(milseg_posdice)
                metrics['cancer_posdice'] = float(cancer_posdice)
                metrics['combined_posdice'] = float(combined_posdice)

                logging.info(
                    f"Val Metrics: AUC={metrics['auc']:.4f}, Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, "
                    f"F1_best={metrics['f1_best']:.4f}@th={metrics['best_threshold']:.2f}"
                )
                logging.info(
                    f"Val Fixed@th={metrics['fixed_threshold']:.2f}: Acc={metrics['accuracy_fixed']:.4f}, "
                    f"F1={metrics['f1_fixed']:.4f}, Spec={metrics['specificity_fixed']:.4f}, PR-AUC={metrics['pr_auc']:.4f}"
                )
                if do_seg_validation:
                    logging.info(
                        f"Val Seg (recon): PosDice={metrics['seg_patch_dice_pos']:.4f}@thr={metrics['seg_best_thr']:.2f}, "
                        f"PosDice@0.5={metrics['seg_dice_pos_at_05']:.4f}, "
                        f"Dice={metrics['seg_patch_dice']:.4f}"
                    )
                    logging.info(
                        f"Val Seg (milseg): PosDice={metrics['milseg_posdice']:.4f}, "
                        f"(cancer): PosDice={metrics['cancer_posdice']:.4f}, "
                        f"(combined): PosDice={metrics['combined_posdice']:.4f}"
                    )
                    if voxel_dice >= 0:
                        logging.info(f"Val Seg (voxel): VoxelDice={voxel_dice:.4f}")
                with open(validation_history_path, 'a', encoding='utf-8') as f:
                    f.write(
                        f"{epoch+1}\t{metrics['auc']:.6f}\t{metrics['accuracy']:.6f}\t{metrics['f1']:.6f}\t"
                        f"{metrics['f1_best']:.6f}\t{metrics['best_threshold']:.4f}\t{metrics['accuracy_fixed']:.6f}\t"
                        f"{metrics['f1_fixed']:.6f}\t{metrics['specificity_fixed']:.6f}\t{metrics['pr_auc']:.6f}\t"
                        f"{metrics['seg_patch_dice']:.6f}\t{metrics['seg_patch_dice_pos']:.6f}\t{epoch_lr:.8f}\t"
                        f"{metrics['seg_best_thr']:.4f}\t{metrics['seg_dice_pos_at_05']:.6f}\t{metrics['voxel_dice']:.6f}\n"
                    )
                
                codebook_stats = model.module.codebook.get_code_statistics()
                logging.info(
                    f"Codebook: healthy={codebook_stats['healthy_codes_used']}/{codebook_stats['total_healthy_codes']}, "
                    f"cancer={codebook_stats['cancer_codes_used']}/{codebook_stats['total_cancer_codes']}, "
                    f"healthy_frozen={model.module.codebook.healthy_frozen}"
                )
                if do_seg_validation:
                    cs = model.module.codebook.get_cancer_scores()
                    hm = model.module.codebook.healthy_code_mask
                    h_scores = cs[hm] if hm.any() else cs
                    c_scores = cs[~hm] if (~hm).any() else cs
                    logging.info(
                        f"CancerScores: healthy_mean={h_scores.mean():.4f}, cancer_mean={c_scores.mean():.4f}, "
                        f"gap={c_scores.mean() - h_scores.mean():.4f}, max={cs.max():.4f}"
                    )
                
                if (
                    metrics['auc'] > best_auc and
                    metrics['f1_fixed'] >= best_model_min_f1_fixed and
                    metrics['specificity_fixed'] >= best_model_min_specificity
                ):
                    best_auc = metrics['auc']
                    save_path = os.path.join(config.checkpoint['save_dir'], 'best_model.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'metrics': metrics,
                    }, save_path)
                    logging.info(f"Best model saved with AUC: {best_auc:.4f}")

                if do_seg_validation and metrics['auc'] >= best_seg_min_auc:
                    best_candidate = max(
                        metrics['seg_patch_dice_pos'],
                        metrics.get('milseg_posdice', -1.0),
                        metrics.get('combined_posdice', -1.0),
                    )
                    if best_candidate > best_seg_score:
                        best_seg_score = best_candidate
                        save_path = os.path.join(config.checkpoint['save_dir'], 'best_seg_model.pth')
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'metrics': metrics,
                        }, save_path)
                        logging.info(
                            f"Best segmentation model saved with PosPatchDice: {best_seg_score:.4f} "
                            f"(AUC={metrics['auc']:.4f})"
                        )
                
                if checkpoint_save_freq > 0 and (epoch + 1) % checkpoint_save_freq == 0:
                    save_path = os.path.join(config.checkpoint['save_dir'], f'checkpoint_epoch_{epoch+1}.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'metrics': metrics,
                    }, save_path)
                    logging.info(f"Checkpoint saved at epoch {epoch+1}")
                
                logging.info("-"*50)

        dist.barrier(device_ids=[rank])
    
    cleanup_ddp()


if __name__ == '__main__':
    main()
