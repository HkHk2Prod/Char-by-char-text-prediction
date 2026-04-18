from abc import ABC
import torch.nn as nn

class BaseCharModel(ABC, nn.Module):
    """
    I made an ABC to wrap the models around it. It does nothing at the moment since all models have basically the same interface. 
    I made it just in case I will need something later.
    """

    def __init__(self, vocab_size: int, cfg: dict):
        super().__init__()
        self.vocab_size = vocab_size
        self.cfg = cfg

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
