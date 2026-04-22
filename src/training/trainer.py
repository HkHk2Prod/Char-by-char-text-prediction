import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.base import BaseCharModel
from src.training.callbacks import Callback
from src.utils import detach_state


@dataclass
class TrainerConfig:
    grad_clip: float = 5.0
    epochs: int = 20
    save_dir: Path | str = "Model_files"


class Trainer:

    def __init__(
        self,
        model: BaseCharModel,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        cfg: TrainerConfig | None = None,
        device: str | None = None,
        callbacks: list[Callback] | None = None,
    ):

        self.model = model
        self.train_loader = train_loader
        self.cfg = cfg or TrainerConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.callbacks = callbacks or []
        self._stop = False

        self.model.to(self.device)

        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()

        # Timestamped run directory
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(self.cfg.save_dir) / f"{model.model_name}_{ts}"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def stop(self) -> None:
        self._stop = True

    # ── Public API ────────────────────────────────────────────────────────────

    def train(self) -> None:
        self._fire("on_train_start")

        for epoch in range(1, self.cfg.epochs + 1):
            self._fire("on_epoch_start", epoch)
            train_metrics = self._train_epoch(epoch)
            self._fire("on_epoch_end", epoch, train_metrics)

            if self._stop:
                break

        self._fire("on_train_end")

    # ── Train epoch ───────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = total_acc = 0.0

        for i, (x, y) in enumerate(self.train_loader, 1):
            x, y = x.to(self.device), y.to(self.device)
            h = None

            logits, h = self.model(x, h)
            h = detach_state(h)

            # logits: (B, T, V) → (B*T, V) | y: (B, T) → (B*T,)
            loss = self.criterion(logits.view(-1, self.model.vocab_size), y.view(-1))

            self.optimizer.zero_grad()
            loss.backward()
            if self.cfg.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            total_acc += (logits.detach().argmax(-1) == y).float().mean().item()

            self._fire("on_batch_end", epoch, i, total_loss / i)

        n = len(self.train_loader)
        return {"loss": total_loss / n, "acc": total_acc / n}

    def _fire(self, hook: str, *args) -> None:
        for cb in self.callbacks:
            getattr(cb, hook)(*args, trainer=self)
