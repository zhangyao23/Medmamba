import torch
from torch.utils.data import Sampler
import random
from typing import Iterator, List


class BalancedBatchSampler(Sampler):
    def __init__(self, labels: List[int], batch_size: int = 2):
        self.batch_size = batch_size
        
        self.neg_indices = [i for i, label in enumerate(labels) if label == 0]
        self.pos_indices = [i for i, label in enumerate(labels) if label == 1]
        
        self.num_neg = len(self.neg_indices)
        self.num_pos = len(self.pos_indices)
        
        self.num_batches = min(self.num_neg, self.num_pos * (batch_size - 1))
        
    def __iter__(self) -> Iterator[List[int]]:
        random.shuffle(self.neg_indices)
        random.shuffle(self.pos_indices)
        
        pos_idx = 0
        neg_idx = 0
        
        for _ in range(self.num_batches):
            batch = []
            
            batch.append(self.pos_indices[pos_idx % self.num_pos])
            pos_idx += 1
            
            for _ in range(self.batch_size - 1):
                batch.append(self.neg_indices[neg_idx % self.num_neg])
                neg_idx += 1
            
            random.shuffle(batch)
            yield batch
    
    def __len__(self) -> int:
        return self.num_batches
