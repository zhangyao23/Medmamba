import logging
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .losses import CombinedLoss
from .config import Config


class MILTrainer:
    def __init__(
        self,
        config: Config,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        self.loss_fn = CombinedLoss(**config.loss)
        
        self.gradient_accumulation_steps = config.training.get('gradient_accumulation_steps', 1)
        self.mixed_precision = config.training.get('mixed_precision', False)
        
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.writer = None
        if config.logging.get('tensorboard', False):
            log_dir = config.logging.get('log_dir', 'logs/')
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        
        self.global_step = 0
        self.epoch = 0
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            slices = batch['slices'].to(self.device)
            labels = batch['labels'].to(self.device)
            masks = batch['masks'].to(self.device)
            
            if batch_idx < 5:
                label_0_count = (labels == 0).sum().item()
                label_1_count = (labels == 1).sum().item()
                logging.info(f"Batch {batch_idx}: label_0={label_0_count}, label_1={label_1_count}")
            
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    loss, loss_dict = self._forward_pass(slices, labels, masks)
                    loss = loss / self.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss, loss_dict = self._forward_pass(slices, labels, masks)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss_dict['total']
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'cls': f"{loss_dict['cls']:.4f}",
                'nu': f"{loss_dict['nu']:.4f}",
                'vq': f"{loss_dict['vq']:.4f}"
            })
            
            if self.writer and batch_idx % self.config.logging.get('print_freq', 10) == 0:
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'train/{key}_loss', value, self.global_step)
            
            self.global_step += 1
        
        if self.scheduler:
            self.scheduler.step()
        
        self.epoch += 1
        
        return total_loss / num_batches
    
    def _forward_pass(
        self,
        slices: torch.Tensor,
        labels: torch.Tensor,
        masks: torch.Tensor
    ) -> tuple:
        B, N, C, H, W = slices.shape
        
        feature_map = self.model.backbone.forward_feature_map(slices)
        B, N, Cf, Hf, Wf = feature_map.shape
        
        tokens = feature_map.permute(0, 1, 3, 4, 2).reshape(B, N * Hf * Wf, Cf)
        
        is_neg = (labels == 0)
        quantized, codes, vq_loss = self.model.codebook(tokens, is_negative_bag=is_neg)
        
        quantized = quantized.view(B, N, Hf, Wf, Cf)
        pooled = quantized.mean(dim=(2, 3))
        
        context_features = self.model.mamba(pooled, mask=masks)
        
        logits, attention = self.model.mil_head(context_features, mask=masks)
        
        loss, loss_dict = self.loss_fn(logits, attention, labels, vq_loss, mask=masks)
        
        return loss, loss_dict
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation", leave=False):
                slices = batch['slices'].to(self.device)
                labels = batch['labels'].to(self.device)
                masks = batch['masks'].to(self.device)
                
                feature_map = self.model.backbone.forward_feature_map(slices)
                B, N, Cf, Hf, Wf = feature_map.shape
                
                tokens = feature_map.permute(0, 1, 3, 4, 2).reshape(B, N * Hf * Wf, Cf)
                quantized, codes, _ = self.model.codebook(tokens)
                
                quantized = quantized.view(B, N, Hf, Wf, Cf)
                pooled = quantized.mean(dim=(2, 3))
                
                context = self.model.mamba(pooled, mask=masks)
                logits, attention = self.model.mil_head(context, mask=masks)
                
                all_preds.append(logits)
                all_labels.append(labels)
        
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        from src.utils import evaluate_bag_level
        metrics = evaluate_bag_level(all_preds, all_labels)
        
        return metrics
    
    def save_checkpoint(self, filepath: str, **kwargs):
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            **kwargs
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, filepath)
        logging.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.mixed_precision and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logging.info(f"Checkpoint loaded from {filepath}")
