import time
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.base import BaseCharModel


def perplexity(loss: float) -> float:
    return math.exp(loss)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == targets).float().mean().item()


@dataclass
class TrainerConfig:
    # Optimizer cfg
    learning_rate:    float = 3e-4
    weight_decay:     float = 1e-2
    grad_clip:        float = 5.0

    # Training cfg
    epochs:           int   = 20
    eval_every:       int   = 1   # Epochs
    log_every:        int   = 50  # Batches


class Trainer:

    def __init__(
        self,
        model:        BaseCharModel,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        cfg:          TrainerConfig | None = None,
        device:       str | None = None,
    ):
        
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg or TrainerConfig()
        self.device       = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr           = self.cfg.learning_rate,
            weight_decay = self.cfg.weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()


        print(f"[trainer] device        : {self.device}")
        print(f"[trainer] model         : {type(model).__name__}  ({model.count_parameters():,} params)")
        print(f"[trainer] train batches : {len(train_loader)}")
        print(f"[trainer] val batches   : {len(val_loader)}")

    # ── Public API ────────────────────────────────────────────────────────────

    def train(self) -> None:

        for epoch in range(1, self.cfg.epochs + 1):

            self._train_epoch(epoch)

            val_metrics = {}
            if epoch % self.cfg.eval_every == 0:
                t_val = time.time()
                val_metrics = self._eval_epoch()
                val_elapsed = time.time() - t_val

                msg = (f"  [val]  loss {val_metrics['loss']:.4f}"
                    f"  ppl {perplexity(val_metrics['loss']):7.2f}"
                    f"  acc {val_metrics['acc']:.3f}"
                    f"  ({val_elapsed:.1f}s)")
                print(msg)

        print("\n[trainer] done.")

    # ── Train epoch ───────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = total_acc = 0.0
        

        for i, (x, y) in enumerate(self.train_loader, 1):
            x, y = x.to(self.device), y.to(self.device)
            h = None

            logits, h = self.model(x, h)
            h = self._detach_state(h)

            # logits: (B, T, V) → (B*T, V) | y: (B, T) → (B*T,)
            loss = self.criterion(logits.view(-1, self.model.vocab_size), y.view(-1))

            self.optimizer.zero_grad()
            loss.backward()
            if self.cfg.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            
            self.optimizer.step()

            total_loss += loss.item()
            total_acc  += accuracy(logits.detach(), y)

            if i % self.cfg.log_every == 0:
                avg = total_loss / i
                print(f"  epoch {epoch:3d}  step {i:5d}/{len(self.train_loader)}"
                      f"  loss {avg:.4f}  ppl {perplexity(avg):7.2f}")

        n = len(self.train_loader)
        return {"loss": total_loss / n, "acc": total_acc / n}

    # ── Val epoch ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _eval_epoch(self) -> dict:
        self.model.eval()
        total_loss = total_acc = 0.0
        

        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            h = None

            logits, h = self.model(x, h)

            total_loss += self.criterion(
                logits.view(-1, self.model.vocab_size), y.view(-1)
            ).item()
            total_acc += accuracy(logits, y)

        n = len(self.val_loader)
        return {"loss": total_loss / n, "acc": total_acc / n}
    
    @staticmethod
    def _detach_state(state):
        if state is None:
            return None
        if isinstance(state, tuple):
            return tuple(s.detach() for s in state)
        return state.detach()



