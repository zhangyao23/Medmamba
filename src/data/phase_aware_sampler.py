import torch
from torch.utils.data import Sampler
import random
from typing import Iterator, List, Union


class PhaseAwareBatchSampler(Sampler):
    def __init__(
        self,
        labels: List[int],
        batch_size: int = 4,
        phase1_normal_ratio: float = 0.8,
        is_phase1: bool = True
    ):
        self.batch_size = batch_size
        self.phase1_normal_ratio = phase1_normal_ratio
        self.phase = 'phase1_mixed' if is_phase1 else 'phase2_dynamic'
        self._warned_small_batch = False
        self.epoch = 0
        
        self.neg_indices = [i for i, label in enumerate(labels) if label == 0]
        self.pos_indices = [i for i, label in enumerate(labels) if label == 1]
        
        self.num_neg = len(self.neg_indices)
        self.num_pos = len(self.pos_indices)
        self._phase2_normal_ratio = None
        if self.num_pos == 0:
            self.phase = 'warmup_negative_only'
        
        self._update_num_batches()
    
    def _phase1_mixed_counts(self):
        if self.batch_size <= 1:
            return 1, 0
        num_normal = int(round(self.batch_size * self.phase1_normal_ratio))
        num_normal = max(1, min(self.batch_size - 1, num_normal))
        num_cancer = self.batch_size - num_normal
        if (
            self.batch_size < 4 and
            abs((num_normal / self.batch_size) - self.phase1_normal_ratio) > 0.2 and
            not self._warned_small_batch
        ):
            print(
                f"[PhaseAwareBatchSampler] warning: batch_size={self.batch_size} "
                f"cannot accurately realize phase1_normal_ratio={self.phase1_normal_ratio:.2f}, "
                f"using {num_normal}:{num_cancer} (normal:cancer)."
            )
            self._warned_small_batch = True
        return num_normal, num_cancer

    def _phase2_dynamic_counts(self):
        if self.batch_size <= 1:
            return 1, 0
        if self._phase2_normal_ratio is not None:
            num_normal = int(round(self.batch_size * self._phase2_normal_ratio))
            num_normal = max(1, min(self.batch_size - 1, num_normal))
            num_cancer = self.batch_size - num_normal
        else:
            num_cancer = max(1, self.batch_size // 2)
            num_normal = self.batch_size - num_cancer
            if num_normal == 0:
                num_normal = 1
                num_cancer = self.batch_size - 1
        return num_normal, num_cancer

    def set_phase2_normal_ratio(self, ratio: float):
        self._phase2_normal_ratio = ratio
        if self.phase == 'phase2_dynamic':
            self._update_num_batches()

    def _update_num_batches(self):
        if self.num_pos == 0:
            self.phase = 'warmup_negative_only'
        if self.phase == 'warmup_negative_only':
            self.num_normal_per_batch = self.batch_size
            self.num_cancer_per_batch = 0
            self.num_batches = self.num_neg // max(1, self.batch_size)
        elif self.phase == 'phase1_mixed':
            self.num_normal_per_batch, self.num_cancer_per_batch = self._phase1_mixed_counts()
            if self.num_cancer_per_batch == 0:
                self.num_batches = self.num_neg // max(1, self.num_normal_per_batch)
            else:
                self.num_batches = min(
                    self.num_neg // max(1, self.num_normal_per_batch),
                    self.num_pos // max(1, self.num_cancer_per_batch)
                )
        else:
            self.num_normal_per_batch, self.num_cancer_per_batch = self._phase2_dynamic_counts()
            self.num_batches = min(
                self.num_neg // max(1, self.num_normal_per_batch),
                self.num_pos // max(1, self.num_cancer_per_batch)
            )
        self.num_batches = max(1, self.num_batches)
    
    def set_phase(self, phase: Union[bool, str]):
        if isinstance(phase, bool):
            new_phase = 'phase1_mixed' if phase else 'phase2_dynamic'
        else:
            new_phase = phase
        valid_phases = {'warmup_negative_only', 'phase1_mixed', 'phase2_dynamic'}
        if new_phase not in valid_phases:
            raise ValueError(f"Unsupported phase: {new_phase}")
        if self.phase != new_phase:
            self.phase = new_phase
            self._update_num_batches()
            print(f"\n>>> Sampler switched to {self.phase} <<<")
            print(f"Batch composition: {self.num_normal_per_batch} Normal + {self.num_cancer_per_batch} Cancer")
    
    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.epoch)
        neg_indices = self.neg_indices.copy()
        pos_indices = self.pos_indices.copy()
        rng.shuffle(neg_indices)
        rng.shuffle(pos_indices)
        
        pos_idx = 0
        neg_idx = 0
        
        for _ in range(self.num_batches):
            batch = []
            
            for _ in range(self.num_cancer_per_batch):
                batch.append(pos_indices[pos_idx % self.num_pos])
                pos_idx += 1
            
            for _ in range(self.num_normal_per_batch):
                batch.append(neg_indices[neg_idx % self.num_neg])
                neg_idx += 1
            
            rng.shuffle(batch)
            yield batch
    
    def __len__(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int):
        self.epoch = epoch
