# src/inference/predictor.py
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from src.data.dataset import CharVocab
from src.models.base import BaseCharModel
from src.utils import detach_state


class BasePredictor(ABC):
    """Shared sampling logic + generation loop skeleton."""

    def __init__(
        self, model: BaseCharModel, vocab: CharVocab, device: str | None = None
    ):
        self.model = model
        self.vocab = vocab
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def generate(self, prompt: str, length: int = 200, temperature: float = 1.0) -> str:
        self.model.eval()
        prompt_ids = torch.tensor(
            [self.vocab.encode(prompt)], dtype=torch.long, device=self.device
        )

        state = self._prime(prompt_ids)
        generated = list(prompt)

        for _ in range(length):
            next_idx = self._sample_temperature(state["logits"], temperature)
            generated.append(self.vocab.decode([next_idx]))
            state = self._step(state, next_idx)

        return "".join(generated)

    @abstractmethod
    def _prime(self, prompt_ids: torch.Tensor) -> dict: ...

    @abstractmethod
    def _step(self, state: dict, next_id: int) -> dict: ...

    @staticmethod
    def _sample_temperature(logits: torch.Tensor, temperature: float) -> int:
        probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())


class RecurrentPredictor(BasePredictor):
    def _prime(self, prompt_ids):
        logits, h = self.model(prompt_ids)
        return {"logits": logits[:, -1, :], "h": detach_state(h)}

    def _step(self, state, next_id):
        x = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
        logits, h = self.model(x, state["h"])
        return {"logits": logits[:, -1, :], "h": detach_state(h)}


class NonRecurrentPredictor(BasePredictor):
    def _prime(self, prompt_ids):
        seq_len: int = self.model.seq_len  # type: ignore[attr-defined]
        ctx = prompt_ids[:, -seq_len:]
        logits, _ = self.model(ctx)
        return {"logits": logits[:, -1, :], "ctx": ctx}

    def _step(self, state, next_id):
        seq_len: int = self.model.seq_len  # type: ignore[attr-defined]
        next_tok = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
        ctx = torch.cat([state["ctx"], next_tok], dim=1)[:, -seq_len:]
        logits, _ = self.model(ctx)
        return {"logits": logits[:, -1, :], "ctx": ctx}


def make_predictor(
    model: BaseCharModel, vocab: CharVocab, device: str | None = None
) -> BasePredictor:
    cls = RecurrentPredictor if model.is_recurrent else NonRecurrentPredictor
    return cls(model, vocab, device)
