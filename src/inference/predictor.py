import torch
import torch.nn.functional as F
 
from src.models.base import BaseCharModel
from src.data.dataset import TextDataset
 
 
class Predictor:
 
    def __init__(self, model: BaseCharModel, dataset: TextDataset, device: str | None = None):
        self.model   = model
        self.dataset = dataset
        self.device  = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    # ── Public API ────────────────────────────────────────────────────────────
 
    @torch.no_grad()
    def generate(
        self,
        prompt:    str,
        length:    int   = 500,
        temperature: float = 1.0,
    ) -> str:
        vocab = self.dataset.vocab
        encoded_prompt = vocab.encode(prompt)
        encoded_prompt = torch.tensor([encoded_prompt], dtype=torch.long, device=self.device)

        logits, h = self.model(encoded_prompt)          # (1, T, V)
        logits        = logits[:, -1, :]             # (1, V) — last position only
 
        generated = list(prompt)
 
        for _ in range(length):
            next_idx = self._sample_temperature(logits, temperature)
            generated.append(vocab.decode([next_idx]))
            x = torch.tensor([[next_idx]], dtype=torch.long, device=self.device)
            logits, h = self.model(x, h)
            logits = logits[:, -1, :]
 
        return "".join(generated)
 
    @staticmethod
    def _sample_temperature(logits: torch.Tensor, temperature: float) -> int:
        probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item()) # int cast is to silence the type checker