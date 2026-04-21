import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PartitionedVectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int = 100,
        embedding_dim: int = 512,
        healthy_ratio: float = 0.8,
        commitment_cost: float = 0.25,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        epsilon: float = 1e-5,
        usage_balance_alpha: float = 0.0,
        explore_prob: float = 0.0,
        gate_std_factor: float = 1.5,
        min_cancer_fraction: float = 0.1,
        lambda_code_nu: float = 0.5
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.use_ema = use_ema
        self.usage_balance_alpha = usage_balance_alpha
        self.explore_prob = explore_prob
        self.gate_std_factor = gate_std_factor
        self.min_cancer_fraction = min_cancer_fraction
        self.lambda_code_nu = lambda_code_nu
        
        self.num_healthy_codes = int(num_embeddings * healthy_ratio)
        self.num_cancer_codes = num_embeddings - self.num_healthy_codes
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
        self.register_buffer(
            'code_usage_count',
            torch.zeros(num_embeddings)
        )
        self.register_buffer(
            'epoch_code_usage',
            torch.zeros(num_embeddings)
        )
        self.register_buffer(
            'normal_code_usage',
            torch.zeros(num_embeddings)
        )
        self.register_buffer(
            'cancer_code_usage',
            torch.zeros(num_embeddings)
        )
        self.register_buffer(
            'normal_patch_count',
            torch.zeros(num_embeddings)
        )
        self.register_buffer(
            'cancer_patch_count',
            torch.zeros(num_embeddings)
        )
        self.register_buffer(
            'healthy_code_mask',
            torch.zeros(num_embeddings, dtype=torch.bool)
        )
        
        self.register_buffer('_healthy_dist_ema_mean', torch.tensor(0.0))
        self.register_buffer('_healthy_dist_ema_var', torch.tensor(1.0))
        self.register_buffer('_healthy_dist_count', torch.tensor(0.0))
        self.register_buffer('healthy_distance_threshold', torch.tensor(float('inf')))

        self.cancer_logit = nn.Parameter(torch.zeros(num_embeddings))
        
        self.use_dynamic_partition = False
        self.phase1_complete = False
        self.healthy_frozen = False
        self.ema_frozen = False
        self.register_buffer(
            'frozen_code_mask',
            torch.zeros(num_embeddings, dtype=torch.bool)
        )
        
        if use_ema:
            self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
            self.register_buffer('ema_w', self.embedding.weight.data.clone())
            self.ema_decay = ema_decay
            self.epsilon = epsilon

        self.embedding.weight.register_hook(self._embedding_grad_hook)
        
        print(f"Partitioned Codebook: {num_embeddings} codes")
        print(f"  - Healthy codes: [0:{self.num_healthy_codes}]")
        print(f"  - Cancer codes:  [{self.num_healthy_codes}:{num_embeddings}]")
    
    def forward(
        self,
        z: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, D = z.shape
        z_flat = z.reshape(-1, D)
        has_labels = labels is not None
        valid_flat = mask.reshape(-1).bool() if mask is not None else None
        
        encoding_indices = torch.zeros(B * N, dtype=torch.long, device=z.device)
        
        for b in range(B):
            label = labels[b].item() if has_labels else None
            start_idx = b * N
            end_idx = (b + 1) * N
            
            z_sample = z_flat[start_idx:end_idx]
            valid_b = mask[b].bool() if mask is not None else None
            if valid_b is not None:
                if not valid_b.any():
                    continue
                z_assign = z_sample[valid_b]
            else:
                z_assign = z_sample
            
            if not self.phase1_complete:
                distances = torch.cdist(z_assign, self.embedding.weight)
                if self.training and self.usage_balance_alpha > 0:
                    usage = self.epoch_code_usage.to(z_assign.device)
                    usage_norm = usage / (usage.max() + 1e-6)
                    distances = distances + self.usage_balance_alpha * usage_norm.unsqueeze(0)
                indices = distances.argmin(dim=1)

                if self.training and has_labels and label == 0:
                    assigned_features = self.embedding.weight[indices]
                    patch_dists = (z_assign - assigned_features).norm(dim=1)
                    batch_mean = patch_dists.mean().detach()
                    batch_var = patch_dists.var(unbiased=False).detach()
                    n = float(patch_dists.numel())
                    old_count = self._healthy_dist_count.item()
                    new_count = old_count + n
                    if new_count > 0:
                        delta = batch_mean - self._healthy_dist_ema_mean
                        self._healthy_dist_ema_mean += delta * n / new_count
                        self._healthy_dist_ema_var = (
                            self._healthy_dist_ema_var * old_count + batch_var * n
                        ) / new_count
                        self._healthy_dist_count.fill_(new_count)

            elif self.use_dynamic_partition:
                distances = torch.cdist(z_assign, self.embedding.weight)
                if self.training and self.usage_balance_alpha > 0:
                    usage = self.epoch_code_usage.to(z_assign.device)
                    usage_norm = usage / (usage.max() + 1e-6)
                    distances = distances + self.usage_balance_alpha * usage_norm.unsqueeze(0)
                indices = distances.argmin(dim=1)
            else:
                if has_labels and label == 0:
                    codebook_subset = self.embedding.weight[:self.num_healthy_codes]
                    distances = torch.cdist(z_assign, codebook_subset)
                    if self.training and self.usage_balance_alpha > 0:
                        usage = self.epoch_code_usage[:self.num_healthy_codes].to(z_assign.device)
                        usage_norm = usage / (usage.max() + 1e-6)
                        distances = distances + self.usage_balance_alpha * usage_norm.unsqueeze(0)
                    indices = distances.argmin(dim=1)
                else:
                    distances = torch.cdist(z_assign, self.embedding.weight)
                    if self.training and self.usage_balance_alpha > 0:
                        usage = self.epoch_code_usage.to(z_assign.device)
                        usage_norm = usage / (usage.max() + 1e-6)
                        distances = distances + self.usage_balance_alpha * usage_norm.unsqueeze(0)
                    indices = distances.argmin(dim=1)

            if self.training and self.explore_prob > 0:
                explore_mask = torch.rand(indices.shape, device=indices.device) < self.explore_prob
                if explore_mask.any():
                    if self.use_dynamic_partition and self.phase1_complete and has_labels and label == 0 and self.healthy_code_mask.any():
                        healthy_indices = torch.where(self.healthy_code_mask)[0]
                        if healthy_indices.numel() > 0:
                            rand_pos = torch.randint(0, healthy_indices.numel(), (indices.numel(),), device=indices.device)
                            random_indices = healthy_indices[rand_pos]
                            indices = torch.where(explore_mask, random_indices, indices)
                    elif self.phase1_complete and has_labels and label == 0:
                        random_indices = torch.randint(0, self.num_healthy_codes, (indices.numel(),), device=indices.device)
                        indices = torch.where(explore_mask, random_indices, indices)
                    else:
                        random_indices = torch.randint(0, self.num_embeddings, (indices.numel(),), device=indices.device)
                        indices = torch.where(explore_mask, random_indices, indices)
            
            if valid_b is not None:
                sample_indices = encoding_indices[start_idx:end_idx]
                sample_indices[valid_b] = indices
                encoding_indices[start_idx:end_idx] = sample_indices
            else:
                encoding_indices[start_idx:end_idx] = indices
        
        quantized = self.embedding(encoding_indices)
        
        if self.training:
            if valid_flat is None:
                z_for_stats = z_flat
                quantized_for_loss = quantized
                encoding_indices_for_stats = encoding_indices
            elif valid_flat.any():
                z_for_stats = z_flat[valid_flat]
                quantized_for_loss = quantized[valid_flat]
                encoding_indices_for_stats = encoding_indices[valid_flat]
            else:
                z_for_stats = z_flat.new_zeros((0, D))
                quantized_for_loss = quantized.new_zeros((0, D))
                encoding_indices_for_stats = encoding_indices.new_zeros((0,), dtype=encoding_indices.dtype)

            if self.use_ema:
                if not self.ema_frozen and encoding_indices_for_stats.numel() > 0:
                    self._ema_update(z_for_stats, encoding_indices_for_stats)
                if quantized_for_loss.numel() > 0:
                    vq_loss = self.commitment_cost * F.mse_loss(
                        quantized_for_loss.detach(), z_for_stats
                    )
                else:
                    vq_loss = torch.tensor(0.0, device=z.device)
            else:
                if quantized_for_loss.numel() > 0:
                    e_latent_loss = F.mse_loss(quantized_for_loss.detach(), z_for_stats)
                    q_latent_loss = F.mse_loss(quantized_for_loss, z_for_stats.detach())
                    vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss
                else:
                    vq_loss = torch.tensor(0.0, device=z.device)
            
            if encoding_indices_for_stats.numel() > 0:
                unique_indices = encoding_indices_for_stats.unique().detach()
                self.code_usage_count[unique_indices] += 1
                self.epoch_code_usage[unique_indices] += 1
            
            if has_labels:
                for b in range(B):
                    label = labels[b].item()
                    start_idx = b * N
                    end_idx = (b + 1) * N
                    sample_codes = encoding_indices[start_idx:end_idx].detach()
                    if mask is not None:
                        valid_b = mask[b].bool()
                        sample_codes = sample_codes[valid_b]
                    if sample_codes.numel() == 0:
                        continue
                    sample_indices = sample_codes.unique()
                    
                    if label == 0:
                        self.normal_code_usage[sample_indices] += 1
                    else:
                        self.cancer_code_usage[sample_indices] += 1

                    patch_counts = torch.bincount(
                        sample_codes, minlength=self.num_embeddings
                    ).float()
                    if label == 0:
                        self.normal_patch_count += patch_counts
                    else:
                        self.cancer_patch_count += patch_counts

        else:
            vq_loss = torch.tensor(0.0, device=z.device)
        
        quantized = z_flat + (quantized - z_flat).detach()
        if valid_flat is not None:
            quantized = quantized * valid_flat.unsqueeze(-1).float()
        
        quantized = quantized.view(B, N, D)
        encoding_indices = encoding_indices.view(B, N)
        
        return quantized, encoding_indices, vq_loss
    
    def _ema_update(self, z_flat: torch.Tensor, encoding_indices: torch.Tensor):
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()

        new_cluster_size = self.ema_cluster_size * self.ema_decay + \
                           (1 - self.ema_decay) * encodings.sum(0)
        if self.healthy_frozen and self.frozen_code_mask.any():
            update_mask = ~self.frozen_code_mask
            self.ema_cluster_size.data[update_mask] = new_cluster_size.data[update_mask]
        else:
            self.ema_cluster_size.data.copy_(new_cluster_size.data)

        n = self.ema_cluster_size.sum()
        normalized_cluster_size = (
            (self.ema_cluster_size + self.epsilon) /
            (n + self.num_embeddings * self.epsilon) * n
        )
        if self.healthy_frozen and self.frozen_code_mask.any():
            update_mask = ~self.frozen_code_mask
            self.ema_cluster_size.data[update_mask] = normalized_cluster_size.data[update_mask]
        else:
            self.ema_cluster_size.data.copy_(normalized_cluster_size.data)

        dw = encodings.t() @ z_flat.detach()
        new_ema_w = self.ema_w * self.ema_decay + (1 - self.ema_decay) * dw
        if self.healthy_frozen and self.frozen_code_mask.any():
            update_mask = ~self.frozen_code_mask
            self.ema_w.data[update_mask] = new_ema_w.data[update_mask]
        else:
            self.ema_w.data.copy_(new_ema_w.data)

        new_weight = self.ema_w / self.ema_cluster_size.unsqueeze(1)
        if self.healthy_frozen and self.frozen_code_mask.any():
            update_mask = ~self.frozen_code_mask
            self.embedding.weight.data[update_mask] = new_weight.data[update_mask]
        else:
            self.embedding.weight.data.copy_(new_weight.data)
    
    def get_code_statistics(self, epoch_only=False) -> dict:
        usage_count = self.epoch_code_usage if epoch_only else self.code_usage_count
        
        stats = {
            'num_embeddings': self.num_embeddings,
            'codes_used': (usage_count > 0).sum().item(),
            'usage_ratio': (usage_count > 0).sum().item() / self.num_embeddings,
        }
        
        if epoch_only:
            stats['epoch_codes_used'] = stats['codes_used']
            stats['epoch_usage_ratio'] = stats['usage_ratio']
        
        if self.use_dynamic_partition and self.healthy_code_mask.any():
            healthy_mask_used = (usage_count > 0) & self.healthy_code_mask
            cancer_mask_used = (usage_count > 0) & (~self.healthy_code_mask)
            
            stats['healthy_codes_used'] = healthy_mask_used.sum().item()
            stats['cancer_codes_used'] = cancer_mask_used.sum().item()
            stats['total_healthy_codes'] = self.healthy_code_mask.sum().item()
            stats['total_cancer_codes'] = (~self.healthy_code_mask).sum().item()
        else:
            healthy_usage = usage_count[:self.num_healthy_codes].sum().item()
            cancer_usage = usage_count[self.num_healthy_codes:].sum().item()
            
            stats['healthy_codes_used'] = (usage_count[:self.num_healthy_codes] > 0).sum().item()
            stats['cancer_codes_used'] = (usage_count[self.num_healthy_codes:] > 0).sum().item()
            stats['total_healthy_codes'] = self.num_healthy_codes
            stats['total_cancer_codes'] = self.num_cancer_codes
        
        return stats
    
    def reset_epoch_statistics(self):
        self.epoch_code_usage.zero_()
    
    def codes_to_segmentation_mask(
        self,
        code_ids: torch.Tensor,
        soft: bool = False
    ) -> torch.Tensor:
        if soft:
            scores = self.get_cancer_scores()
            return scores[code_ids].float()

        if self.use_dynamic_partition and self.healthy_code_mask.any():
            return (~self.healthy_code_mask[code_ids]).long()
        else:
            return (code_ids >= self.num_healthy_codes).long()

    def get_cancer_scores(self) -> torch.Tensor:
        return torch.sigmoid(self.cancer_logit)

    def compute_soft_cancer_logits(
        self, z_e: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        B, N, D = z_e.shape
        z_flat = z_e.reshape(-1, D)
        dists = torch.cdist(z_flat, self.embedding.weight.detach())
        soft_weights = F.softmax(-dists / temperature, dim=-1)
        soft_logits = (soft_weights * self.cancer_logit.unsqueeze(0)).sum(dim=-1)
        return soft_logits.view(B, N)

    def compute_embedding_separation_loss(
        self, margin: float = 2.0
    ) -> torch.Tensor:
        if not self.healthy_code_mask.any() or self.healthy_code_mask.all():
            return torch.tensor(0.0, device=self.embedding.weight.device)
        healthy_embs = self.embedding.weight[self.healthy_code_mask]
        cancer_embs = self.embedding.weight[~self.healthy_code_mask]
        healthy_center = healthy_embs.mean(dim=0)
        cancer_center = cancer_embs.mean(dim=0)
        center_dist = (healthy_center - cancer_center).norm()
        return F.relu(margin - center_dist)

    def compute_pseudo_routing_loss(
        self,
        z_e: torch.Tensor,
        labels: torch.Tensor,
        masks: torch.Tensor,
        attention: torch.Tensor,
        top_k_ratio: float = 0.15,
        margin: float = 0.5,
    ) -> torch.Tensor:
        if not self.healthy_code_mask.any() or self.healthy_code_mask.all():
            return torch.tensor(0.0, device=z_e.device)

        B, N, D = z_e.shape
        emb = self.embedding.weight.detach()
        healthy_emb = emb[self.healthy_code_mask]
        cancer_emb = emb[~self.healthy_code_mask]
        att = attention.detach()

        loss_sum = torch.tensor(0.0, device=z_e.device)
        count = 0

        for b in range(B):
            if masks is not None:
                valid = masks[b].bool()
            else:
                valid = torch.ones(N, dtype=torch.bool, device=z_e.device)
            n_valid = valid.sum().item()
            if n_valid == 0:
                continue

            z_b = z_e[b, valid]
            d_h = torch.cdist(z_b, healthy_emb).min(dim=1)[0]
            d_c = torch.cdist(z_b, cancer_emb).min(dim=1)[0]

            if labels[b] == 0:
                loss_sum = loss_sum + F.relu(d_h - d_c + margin).mean()
                count += 1
            else:
                att_b = att[b, valid]
                k = max(1, int(top_k_ratio * n_valid))
                _, top_idx = att_b.topk(k)
                loss_sum = loss_sum + F.relu(d_c[top_idx] - d_h[top_idx] + margin).mean()
                count += 1

        if count == 0:
            return torch.tensor(0.0, device=z_e.device)
        return loss_sum / count

    def get_cancer_scores_stats(self) -> torch.Tensor:
        has_patch_stats = (self.normal_patch_count.sum() > 0 or
                          self.cancer_patch_count.sum() > 0)
        has_usage_stats = (self.normal_code_usage.sum() > 0 or
                          self.cancer_code_usage.sum() > 0)
        if not (has_patch_stats or has_usage_stats):
            return torch.zeros(self.num_embeddings, device=self.embedding.weight.device)

        use_patch = has_patch_stats
        if use_patch:
            total_n = self.normal_patch_count.sum()
            total_c = self.cancer_patch_count.sum()
            c_freq = self.cancer_patch_count / (total_c + 1e-8)
            n_freq = self.normal_patch_count / (total_n + 1e-8)
        else:
            total_n = self.normal_code_usage.sum()
            total_c = self.cancer_code_usage.sum()
            c_freq = self.cancer_code_usage / (total_c + 1e-8)
            n_freq = self.normal_code_usage / (total_n + 1e-8)

        enrichment = c_freq / (n_freq + 1e-8)
        total = (self.cancer_patch_count + self.normal_patch_count
                 if use_patch
                 else self.cancer_code_usage + self.normal_code_usage)
        c_ratio = torch.where(
            total > 0,
            (self.cancer_patch_count if use_patch
             else self.cancer_code_usage) / total,
            torch.zeros_like(total)
        )
        scores = 0.5 * enrichment + 0.5 * (c_ratio / (c_ratio.max() + 1e-8))
        scores = scores.clamp(min=0.0)
        score_max = scores.max()
        if score_max > 0:
            scores = scores / score_max
        return scores

    def repartition(self, threshold: float = 0.7, min_cancer_codes: int = None,
                    strategy: str = 'enrichment'):
        if strategy == 'enrichment':
            return self.mark_healthy_codes(threshold=threshold,
                                           min_cancer_codes=min_cancer_codes)
        elif strategy == 'topk':
            if min_cancer_codes is None:
                min_cancer_codes = max(
                    int(self.num_embeddings *
                        (1.0 - self.num_healthy_codes / self.num_embeddings)),
                    4
                )
            scores = self.get_cancer_scores()
            active = (self.normal_code_usage + self.cancer_code_usage) > 0
            scores[~active] = -1.0
            topk = torch.topk(scores, k=min(min_cancer_codes,
                                             int(active.sum().item())),
                               largest=True)
            self.healthy_code_mask = torch.ones(self.num_embeddings,
                                                dtype=torch.bool,
                                                device=self.healthy_code_mask.device)
            for idx in topk.indices:
                if scores[idx] > 0:
                    self.healthy_code_mask[idx] = False

            num_healthy = self.healthy_code_mask.sum().item()
            num_cancer = int((~self.healthy_code_mask).sum().item())
            print(f"[TopK Repartition] Healthy={num_healthy}, Cancer={num_cancer}")
            cancer_indices = torch.where(~self.healthy_code_mask)[0]
            print(f"Cancer code indices: {cancer_indices.tolist()}")
            return num_healthy, num_cancer
        else:
            return self.mark_healthy_codes(threshold=threshold,
                                           min_cancer_codes=min_cancer_codes)
    
    def mark_healthy_codes(self, threshold: float = 0.7, min_cancer_codes: int = None):
        total_usage = self.normal_code_usage + self.cancer_code_usage
        active_mask = total_usage > 0
        n_active = active_mask.sum().item()

        total_normal_samples = self.normal_code_usage.sum()
        total_cancer_samples = self.cancer_code_usage.sum()

        use_patch_stats = (self.normal_patch_count.sum() > 0 or
                           self.cancer_patch_count.sum() > 0)

        if use_patch_stats:
            total_normal_patches = self.normal_patch_count.sum()
            total_cancer_patches = self.cancer_patch_count.sum()
            cancer_freq = torch.where(
                total_cancer_patches > 0,
                self.cancer_patch_count / total_cancer_patches,
                torch.zeros_like(self.cancer_patch_count)
            )
            normal_freq = torch.where(
                total_normal_patches > 0,
                self.normal_patch_count / total_normal_patches,
                torch.zeros_like(self.normal_patch_count)
            )
            print(f"[Partition] Using patch-level stats: "
                  f"{int(total_normal_patches.item())} normal, "
                  f"{int(total_cancer_patches.item())} cancer patches")
        else:
            cancer_freq = torch.where(
                total_cancer_samples > 0,
                self.cancer_code_usage / total_cancer_samples,
                torch.zeros_like(self.cancer_code_usage)
            )
            normal_freq = torch.where(
                total_normal_samples > 0,
                self.normal_code_usage / total_normal_samples,
                torch.zeros_like(self.normal_code_usage)
            )
            print(f"[Partition] Using sample-level stats: "
                  f"{int(total_normal_samples.item())} normal, "
                  f"{int(total_cancer_samples.item())} cancer samples")

        cancer_enrichment = cancer_freq / (normal_freq + 1e-8)

        total_patch_or_usage = (self.cancer_patch_count + self.normal_patch_count
                                if use_patch_stats else total_usage)
        cancer_ratio = torch.where(
            total_patch_or_usage > 0,
            (self.cancer_patch_count if use_patch_stats
             else self.cancer_code_usage) / total_patch_or_usage,
            torch.zeros_like(total_patch_or_usage)
        )

        cancer_score = 0.5 * cancer_enrichment + 0.5 * (cancer_ratio / (cancer_ratio.max() + 1e-8))
        cancer_score[~active_mask] = -1.0

        if min_cancer_codes is None:
            min_cancer_codes = max(
                int(self.num_embeddings * (1.0 - self.num_healthy_codes / self.num_embeddings)),
                4
            )

        min_cancer_codes = min(min_cancer_codes, n_active)

        normal_ratio = torch.where(
            active_mask,
            (self.normal_code_usage / total_usage),
            torch.ones_like(total_usage)
        )
        self.healthy_code_mask = (normal_ratio >= threshold) | (~active_mask)
        num_cancer = int((~self.healthy_code_mask).sum().item())

        if num_cancer < min_cancer_codes:
            n_needed = min_cancer_codes - num_cancer
            already_cancer = ~self.healthy_code_mask
            candidate_scores = cancer_score.clone()
            candidate_scores[already_cancer] = -2.0
            candidate_scores[~active_mask] = -2.0
            topk = torch.topk(candidate_scores, k=min(n_needed, n_active),
                              largest=True)
            for idx in topk.indices:
                if candidate_scores[idx] > -2.0:
                    self.healthy_code_mask[idx] = False

        num_healthy = self.healthy_code_mask.sum().item()
        num_cancer = int((~self.healthy_code_mask).sum().item())

        self.register_buffer('cancer_score', cancer_score, persistent=True)

        print(f"\n=== Codebook Cleaning Complete ===")
        print(f"Healthy codes identified: {num_healthy}/{self.num_embeddings}")
        print(f"Cancer codes identified: {num_cancer}/{self.num_embeddings}")
        print(f"Threshold: {threshold:.2f}, Min cancer codes: {min_cancer_codes}")
        print(f"Active codes: {n_active}/{self.num_embeddings}")

        healthy_indices = torch.where(self.healthy_code_mask)[0]
        cancer_indices = torch.where(~self.healthy_code_mask)[0]
        print(f"Healthy code indices: {healthy_indices[:10].tolist()}...")
        print(f"Cancer code indices: {cancer_indices.tolist()}")

        if cancer_indices.numel() > 0:
            for ci in cancer_indices:
                ci_int = ci.item()
                print(f"  Code {ci_int}: cancer_enrichment={cancer_enrichment[ci_int]:.3f}, "
                      f"cancer_ratio={cancer_ratio[ci_int]:.3f}, "
                      f"cancer_score={cancer_score[ci_int]:.3f}")

        return num_healthy, num_cancer
    
    def complete_phase1(self, threshold: float = 0.7):
        self.phase1_complete = True
        num_healthy, num_cancer = self.mark_healthy_codes(threshold)

        if self._healthy_dist_count.item() > 0:
            mean_val = self._healthy_dist_ema_mean.item()
            std_val = max(self._healthy_dist_ema_var.item(), 0.0) ** 0.5
            self.healthy_distance_threshold.fill_(
                mean_val + self.gate_std_factor * std_val
            )
            print(f"[Gate] Healthy distance threshold = {self.healthy_distance_threshold.item():.4f} "
                  f"(mean={mean_val:.4f}, std={std_val:.4f}, k={self.gate_std_factor})")
        else:
            self.healthy_distance_threshold.fill_(1.0)
            print(f"[Gate] No healthy distance stats collected, using default threshold=1.0")

        print(f"\n>>> Phase 1 Complete: Switching to Dynamic Partition <<<\n")
        return num_healthy, num_cancer
    
    def enable_dynamic_partition(self):
        self.use_dynamic_partition = True
        print(f">>> Dynamic Partition Enabled <<<")

    def _embedding_grad_hook(self, grad: torch.Tensor) -> torch.Tensor:
        if grad is None:
            return grad
        if self.healthy_frozen and self.frozen_code_mask.any():
            mask = self.frozen_code_mask.to(grad.device)
            grad = grad.clone()
            grad[mask] = 0
        return grad

    def freeze_healthy_codes(self, mask_source: str = 'initial'):
        if mask_source == 'dynamic' and self.healthy_code_mask.any():
            freeze_mask = self.healthy_code_mask.clone()
        else:
            freeze_mask = torch.zeros_like(self.frozen_code_mask)
            freeze_mask[:self.num_healthy_codes] = True
        self.frozen_code_mask = freeze_mask
        self.healthy_frozen = True

    def unfreeze_healthy_codes(self):
        self.frozen_code_mask.zero_()
        self.healthy_frozen = False

    def revive_dead_codes(
        self,
        min_epoch_usage: float = 1.0,
        noise_std: float = 0.01,
        use_epoch_stats: bool = True
    ) -> int:
        usage = self.epoch_code_usage if use_epoch_stats else self.code_usage_count
        dead_mask = usage < min_epoch_usage
        if self.healthy_frozen and self.frozen_code_mask.any():
            dead_mask = dead_mask & (~self.frozen_code_mask)

        dead_indices = torch.where(dead_mask)[0]
        if dead_indices.numel() == 0:
            return 0

        live_indices = torch.where(~dead_mask)[0]
        with torch.no_grad():
            if live_indices.numel() > 0:
                sample_idx = torch.randint(
                    0,
                    live_indices.numel(),
                    (dead_indices.numel(),),
                    device=self.embedding.weight.device
                )
                source = self.embedding.weight[live_indices[sample_idx]]
                noise = torch.randn_like(source) * noise_std
                new_values = source + noise
            else:
                scale = 1.0 / max(1, self.num_embeddings)
                new_values = torch.empty(
                    dead_indices.numel(),
                    self.embedding_dim,
                    device=self.embedding.weight.device
                ).uniform_(-scale, scale)

            self.embedding.weight.data[dead_indices] = new_values
            if self.use_ema:
                live_cluster_sizes = self.ema_cluster_size[~dead_mask]
                if live_cluster_sizes.numel() > 0 and live_cluster_sizes.sum() > 0:
                    init_cluster_size = max(self.epsilon, live_cluster_sizes.mean().item() * 0.1)
                else:
                    init_cluster_size = 1.0
                self.ema_w[dead_indices] = new_values * init_cluster_size
                self.ema_cluster_size[dead_indices] = init_cluster_size

        return int(dead_indices.numel())
