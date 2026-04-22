from abc import ABC
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from src.utils import detach_state


@dataclass
class GenState:
    """State carried between generation steps. Interpretation is model-specific."""

    last_logits: torch.Tensor  # (1, V) — logits for the next token
    hidden: object = None  # recurrent state, KV cache, id buffer, etc.


class BaseCharModel(ABC, nn.Module):
    """
    I made an ABC to wrap the models around it.
    WARNING: It implements the generation for recurrent models.
    If the model is not recurrent (i.e. Transformers), one should redefine prime and generate_step.
    """

    model_name: str = ""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

    def save(self, file_path: Path | str) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self.cfg,
                "model_name": self.model_name,
            },
            file_path,
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
