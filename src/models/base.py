from abc import ABC
from pathlib import Path

import torch
import torch.nn as nn


class BaseCharModel(ABC, nn.Module):
    """
    I made an ABC to wrap the models around it. It does nothing at the moment since all models have basically the same interface. 
    I made it just in case I will need something later.
    """

    model_name: str = ""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

    def save(self, file_path: Path | str) -> None:
        torch.save({
            "state_dict": self.state_dict(),
            "config": self.cfg, 
            "model_name": self.model_name
            },
            file_path
        )

    @classmethod
    def load(cls, file_path: Path | str) -> "BaseCharModel":
        from src.models.registry import registry

        checkpoint = torch.load(file_path, map_location="cpu")
        model = registry.build(checkpoint["model_name"], checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    
