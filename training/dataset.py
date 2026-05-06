"""
training/dataset.py
===================
Dataset utilities for GRPO training.
"""
from typing import List, Dict, Any
from torch.utils.data import Dataset

class GRPODataset(Dataset):
    """Dataset for GRPO training with rollouts."""
    
    def __init__(self, records: List[Dict[str, Any]]):
        """
        Initialize dataset.
        
        Args:
            records: List of training records
        """
        self.records = records
    
    def __len__(self) -> int:
        return len(self.records)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]
