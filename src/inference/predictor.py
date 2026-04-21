import torch
import torch.nn.functional as F

from src.data.dataset import CharVocab
from src.models.base import BaseCharModel
from src.utils import detach_state


class Predictor:
 
    def __init__(self, model: BaseCharModel, vocab: CharVocab, device: str | None = None):
        self.model   = model
        self.vocab = vocab
        self.device  = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    # ── Public API ────────────────────────────────────────────────────────────
 
    @torch.no_grad()
    def generate(
        self,
        prompt:    str,
        length:    int   = 200,
        temperature: float = 1.0,
    ) -> str:
        self.model.eval()
        encoded_prompt = self.vocab.encode(prompt)
        encoded_prompt = torch.tensor([encoded_prompt], dtype=torch.long, device=self.device)

        logits, h = self.model(encoded_prompt)          # (1, T, V)
        logits        = logits[:, -1, :]             # (1, V) — last position only
 
        generated = list(prompt)
 
        for _ in range(length):
            next_idx = self._sample_temperature(logits, temperature)
            generated.append(self.vocab.decode([next_idx]))
            x = torch.tensor([[next_idx]], dtype=torch.long, device=self.device)
            logits, h = self.model(x, detach_state(h))
            logits = logits[:, -1, :]
 
        return "".join(generated)
 
    @staticmethod
    def _sample_temperature(logits: torch.Tensor, temperature: float) -> int:
        probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item()) # int cast is to silence the type checker        return int(torch.multinomial(probs, num_samples=1).item()) # int cast is to silence the type checker