import json
import math
import time
from dataclasses import asdict
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from src.models.base import BaseCharModel
from src.utils import detach_state

if TYPE_CHECKING:
    from src.inference.predictor import Predictor
    from src.training.trainer import Trainer


# ── Helpers ───────────────────────────────────────────────────────────────────


def perplexity(loss: float) -> float:
    return math.exp(loss)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == targets).float().mean().item()


# ── Base ──────────────────────────────────────────────────────────────────────


class Callback:
    def on_train_start(self, trainer: "Trainer") -> None:
        pass

    def on_epoch_start(self, epoch: int, trainer: "Trainer") -> None:
        pass

    def on_batch_end(
        self, epoch: int, step: int, loss: float, trainer: "Trainer"
    ) -> None:
        pass

    def on_epoch_end(self, epoch: int, train_metrics: dict, trainer: "Trainer") -> None:
        pass

    def on_val_end(self, epoch: int, val_metrics: dict, trainer: "Trainer") -> None:
        pass

    def on_train_end(self, trainer: "Trainer") -> None:
        pass


class ModelInfoCallback(Callback):
    def on_train_start(self, trainer: "Trainer") -> None:
        model = trainer.model
        cfg = trainer.cfg

        print(f"\n{'═' * 55}")
        print(f"  model      : {type(model).__name__}")
        print(f"  parameters : {model.count_parameters():,}")
        print(f"  device     : {trainer.device}")
        print(f"  epochs     : {cfg.epochs}")
        print(f"  batches    : {len(trainer.train_loader)}")
        print(f"  save dir    : {trainer.save_dir}")
        print(f"{'═' * 55}\n")


class ConfigSaverCallback(Callback):

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ):

        self.optimizer = optimizer
        self.scheduler = scheduler

    def on_train_start(self, trainer: "Trainer") -> None:
        optimizer_cfg = {
            "type": type(self.optimizer).__name__,
            **{
                k: v for k, v in self.optimizer.param_groups[0].items() if k != "params"
            },  # exclude the actual parameter tensors
        }

        scheduler_cfg = (
            {
                "type": type(self.scheduler).__name__,
                **{k: v for k, v in self.scheduler.state_dict().items()},
            }
            if self.scheduler
            else None
        )

        with open(trainer.save_dir / "config.json", "w") as f:
            json.dump(
                {
                    "trainer": asdict(trainer.cfg),
                    "model": trainer.model.cfg,
                    "optimizer": optimizer_cfg,
                    "scheduler": scheduler_cfg,
                },
                f,
                indent=2,
            )


class EpochProgressCallback(Callback):
    """
    Print epoch header at the start and elapsed time at the end.
    """

    def __init__(self):
        self._t0: float = 0.0

    def on_epoch_start(self, epoch: int, trainer: "Trainer") -> None:
        self._t0 = time.time()
        print(f"── epoch {epoch}/{trainer.cfg.epochs} {'─' * 40}")

    def on_epoch_end(self, epoch: int, train_metrics: dict, trainer: "Trainer") -> None:
        elapsed = time.time() - self._t0
        elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        print(f"  epoch {epoch} done in {elapsed}")


class BatchLogCallback(Callback):
    """Prints loss and perplexity every N batches during training."""

    def __init__(self, log_every: int = 50):
        self.log_every = log_every

    def on_batch_end(
        self, epoch: int, step: int, loss: float, trainer: "Trainer"
    ) -> None:
        if step % self.log_every != 0:
            return
        print(
            f"  step {step:5d}/{len(trainer.train_loader)}"
            f"  loss {loss:.4f}  ppl {perplexity(loss):7.2f}"
        )


@torch.no_grad()
def _run_eval(model, loader, criterion, device) -> dict:
    """
    Auxilary function to run test/val dataset evaluation of the model.
    """

    model.eval()
    total_loss = total_acc = 0.0
    h = None

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, h = model(x, h)
        h = detach_state(h)
        total_loss += criterion(logits.view(-1, model.vocab_size), y.view(-1)).item()
        total_acc += accuracy(logits, y)

    n = len(loader)
    return {"loss": total_loss / n, "acc": total_acc / n}


class EvalCallback(Callback):
    """
    Runs validation and fires on_val_end so downstream callbacks receive metrics.
    Must be listed before LogCallback, CheckpointCallback, GenerationCallback,
    and EarlyStoppingCallback.
    """

    def __init__(self, val_loader: DataLoader, eval_every: int = 1):
        self.val_loader = val_loader
        self.eval_every = eval_every

    def on_epoch_end(self, epoch: int, train_metrics: dict, trainer: "Trainer") -> None:
        if epoch % self.eval_every != 0:
            return

        t0 = time.time()
        val_metrics = _run_eval(
            trainer.model, self.val_loader, trainer.criterion, trainer.device
        )
        elapsed = time.time() - t0

        print(
            f"  [val] loss {val_metrics['loss']:.4f}  "
            f"ppl {perplexity(val_metrics['loss']):7.2f}  "
            f"acc {val_metrics['acc']:.3f}  "
            f"({elapsed:.1f}s)"
        )

        trainer._fire("on_val_end", epoch, val_metrics)


class LogCallback(Callback):
    """
    Prints the epoch summary line and writes metrics to metrics.jsonl.
    Caches val metrics from on_val_end (fired by EvalCallback) until on_epoch_end.
    """

    def __init__(self):
        self._val_cache: dict | None = None

    def on_val_end(self, epoch: int, val_metrics: dict, trainer: "Trainer") -> None:
        self._val_cache = val_metrics

    def on_epoch_end(self, epoch: int, train_metrics: dict, trainer: "Trainer") -> None:
        val_m = self._val_cache
        self._val_cache = None

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_ppl": perplexity(train_metrics["loss"]),
        }
        if val_m is not None:
            row |= {
                "val_loss": val_m["loss"],
                "val_acc": val_m["acc"],
                "val_ppl": perplexity(val_m["loss"]),
            }

        with open(trainer.save_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")

        val_str = (
            f"  val loss {val_m['loss']:.4f}  val ppl {perplexity(val_m['loss']):7.2f}"
            if val_m is not None
            else ""
        )
        print(
            f"  [summary] train loss {train_metrics['loss']:.4f}"
            f"  train ppl {perplexity(train_metrics['loss']):7.2f}"
            f"{val_str}"
        )

    def on_train_end(self, trainer: "Trainer") -> None:
        print(f"\n[log] metrics saved to {trainer.save_dir / 'metrics.jsonl'}")


# ── Checkpoint ────────────────────────────────────────────────────────────────


class CheckpointCallback(Callback):
    """Saves numbered checkpoints every N epochs and best.ckpt on val improvement."""

    def __init__(self, save_every: int = 1):
        self.save_every = save_every
        self._best = float("inf")
        self._best_accuracy = 0

    def on_epoch_end(self, epoch: int, train_metrics: dict, trainer: "Trainer") -> None:
        if epoch % self.save_every == 0:
            trainer.model.save(trainer.save_dir / f"epoch_{epoch:03d}.ckpt")

    def on_val_end(self, epoch: int, val_metrics: dict, trainer: "Trainer") -> None:
        if val_metrics["loss"] < self._best:
            self._best = val_metrics["loss"]
            self._best_accuracy = val_metrics["acc"]
            trainer.model.save(trainer.save_dir / "best.ckpt")
            print(f"  [ckpt] new best → {self._best:.4f}")

    def on_train_end(self, trainer: "Trainer") -> None:
        print(
            f"[ckpt] best val loss {self._best:.4f}, ppl {perplexity(self._best):.2f}, acc {self._best_accuracy:.4f}."
        )


# ── Generation ────────────────────────────────────────────────────────────────


class GenerationCallback(Callback):
    """Sample from the model after validation to track generation quality."""

    def __init__(
        self,
        predictor: "Predictor",
        prompts: list[str],
        seq_length: int = 200,
        every: int = 1,
        temperature: float = 1.0,
    ):
        self.predictor = predictor
        self.prompts = prompts
        self.seq_length = seq_length
        self.every = every
        self.temperature = temperature

    def on_val_end(self, epoch: int, val_metrics: dict, trainer: "Trainer") -> None:
        if epoch % self.every != 0:
            return

        print(f"\n── generation @ epoch {epoch} {'─' * 34}")
        for prompt in self.prompts:
            text = self.predictor.generate(
                prompt=prompt,
                length=self.seq_length,
                temperature=self.temperature,
            )
            print(f"\n[prompt] {prompt!r}")
            print(text)
        print(f"{'─' * 56}\n")


# ── Early stopping ────────────────────────────────────────────────────────────


class EarlyStoppingCallback(Callback):
    """Stop training when val loss stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self._best = float("inf")
        self._counter = 0

    def on_val_end(self, epoch: int, val_metrics: dict, trainer: "Trainer") -> None:
        if val_metrics["loss"] < self._best - self.min_delta:
            self._best = val_metrics["loss"]
            self._counter = 0
        else:
            self._counter += 1
            print(f"  [early stop] no improvement {self._counter}/{self.patience}")
            if self._counter >= self.patience:
                trainer.stop()


# ── LR logger ─────────────────────────────────────────────────────────────────


class LRLoggerCallback(Callback):
    """Print the learning rate at the start of each epoch."""

    def on_epoch_start(self, epoch: int, trainer: "Trainer") -> None:
        lr = trainer.optimizer.param_groups[0]["lr"]
        print(f"  [lr] {lr:.2e}")


class TestEvalCallback(Callback):
    """
    Evaluates on the test set once at the end of training.
    Should be the last callback in the list.
    """

    def __init__(
        self,
        test_loader: DataLoader,
        predictor: "Predictor",
        prompts: list[str],
        seq_length: int = 200,
        temperature: float = 1.0,
    ):
        self.test_loader = test_loader
        self.predictor = predictor
        self.prompts = prompts
        self.seq_length = seq_length
        self.temperature = temperature

    def on_train_end(self, trainer: "Trainer") -> None:

        # load best checkpoint before evaluating
        best_ckpt = trainer.save_dir / "best.ckpt"
        if best_ckpt.exists():
            trainer.model = BaseCharModel.load(best_ckpt)
            trainer.model.to(trainer.device)
            print("  [test] loaded best.ckpt")

        print("\n── final generation ────────────────────────────────")
        for prompt in self.prompts:
            text = self.predictor.generate(
                prompt=prompt,
                length=self.seq_length,
                temperature=self.temperature,
            )
            print(f"\n[prompt] {prompt!r}")
            print(text)

        print("\n── final test evaluation ───────────────────────────")
        metrics = _run_eval(
            trainer.model, self.test_loader, trainer.criterion, trainer.device
        )
        print(
            f"  [test] loss {metrics['loss']:.4f}  "
            f"ppl {perplexity(metrics['loss']):7.2f}  "
            f"acc {metrics['acc']:.3f}"
        )
        print(f"{'─' * 52}\n")


class SchedulerCallback(Callback):
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_epoch_end(self, epoch, train_metrics, trainer) -> None:
        self.scheduler.step()
