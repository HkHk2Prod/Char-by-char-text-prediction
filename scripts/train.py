import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import CharVocab, TextDataset
from src.inference.predictor import make_predictor
from src.models.autodiscover import autodiscover
from src.models.registry import registry
from src.training.callbacks import (
    BatchLogCallback,
    CheckpointCallback,
    ConfigSaverCallback,
    EarlyStoppingCallback,
    EpochProgressCallback,
    EvalCallback,
    GenerationCallback,
    LogCallback,
    ModelInfoCallback,
    SchedulerCallback,
    TestEvalCallback,
)
from src.training.trainer import Trainer, TrainerConfig


def parse_args():
    p = argparse.ArgumentParser(description="Train a character-level model.")

    p.add_argument("--model", required=True)
    p.add_argument("--dataset", choices=["test", "shakespeare"], default="test")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--data_dir", default="data/raw")
    p.add_argument("--seq_len", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default=None)
    p.add_argument("--lower_case", action="store_true")

    # Model hyperparameters
    p.add_argument("--embed_size", type=int, default=64)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)

    # Transformer-specific
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--ffn_dim", type=int, default=512)

    # Trainer params
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--save_dir", default="Model_files")
    p.add_argument("--grad_clip", type=float, default=5.0)

    return p.parse_args()


def sample_prompts(text: str, n: int = 3, length: int = 20) -> list[str]:
    """
    Axilary function. It samples prompts from the text if there are no prompts given.
    In the main the text used is test dataset.
    """
    if len(text) < length:
        raise ValueError(
            f"Text too short for prompt length {length} — "
            f"got {len(text)} chars, need at least {length}."
        )
    max_start = len(text) - length
    n = min(n, max_start + 1)  # can't have more prompts than positions
    starts = random.sample(range(max_start + 1), n)
    return [text[i : i + length] for i in starts]


def prepare_prompts(
    prompts: list[str], vocab: CharVocab, lower_case: bool
) -> list[str]:
    """
    Checks that prompts consist of chars in vocab and removes invalid prompts.
    If lower_case == True, it makes all promts lower case.
    """

    if lower_case:
        prompts = [p.lower() for p in prompts]

    valid = []
    for prompt in prompts:
        unknown = set(prompt) - set(vocab.chars)
        if unknown:
            print(f"[warning] dropping prompt {prompt!r} — unknown chars: {unknown}")
        else:
            valid.append(prompt)

    if not valid:
        print(
            "No valid prompts remaining after vocabulary check. The text generation won't be done during training."
        )

    return valid


def build_model_config(args: argparse.Namespace) -> dict:
    """Assemble model config from CLI args."""
    return {
        "model": args.model,
        "embed_size": args.embed_size,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "num_heads": args.num_heads,
        "ffn_dim": args.ffn_dim,
        "seq_len": args.seq_len,
    }


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    prompts = None
    model_cfg = build_model_config(args)

    match args.dataset:
        case "shakespeare":
            file_name = "tiny_shakespeare.txt"
            prompts = ["ROMEO:", "To be or not to be"]
        case "test":
            file_name = "test_text.txt"
        case _:
            raise ValueError("Unknown dataset")

    datasets = TextDataset.generate_test_train(
        file_path=data_dir / file_name,
        seq_len=args.seq_len,
        val_test_ratio=(args.val_ratio, args.test_ratio),
        lower_case=args.lower_case,
    )

    vocab = datasets[0].vocab

    loaders = {
        name: DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=(name == "train"),
            num_workers=4,
            drop_last=(
                name != "train"
            ),  # During the test/val the hidden state is transfered from one batch to another.
            # The last batch has smaller size, so it creates a mismatch and error.
            # I did the easiest fix to drop the last batch.
            # The training loop resets the hidden state(since it shuffles),
            # so there is no problem with the mismatch.
            pin_memory=device.startswith("cuda"),
        )
        for name, ds in zip(["train", "val", "test"], datasets)
    }

    autodiscover()
    model_cfg["vocab_size"] = len(vocab)
    model = registry.build(args.model, config=model_cfg)
    predictor = make_predictor(model, vocab, device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    trainer_cfg = TrainerConfig(
        grad_clip=args.grad_clip,
        epochs=args.epochs,
        save_dir=args.save_dir,
    )
    prompts = prompts or sample_prompts(
        datasets[2].text or ""
    )  # We sample_prompts from test text.
    prompts = prepare_prompts(prompts=prompts, vocab=vocab, lower_case=args.lower_case)

    callbacks = [
        ConfigSaverCallback(
            optimizer=optimizer,
            scheduler=scheduler,  # pass None if not using one
        ),
        ModelInfoCallback(vocab=vocab),
        EpochProgressCallback(),
        BatchLogCallback(log_every=None),
        EvalCallback(loaders["val"]),
        LogCallback(),
        CheckpointCallback(),
        GenerationCallback(predictor=predictor, prompts=prompts),
        EarlyStoppingCallback(patience=5),
        TestEvalCallback(loaders["test"], predictor=predictor, prompts=prompts),
        SchedulerCallback(scheduler=scheduler),
    ]

    trainer = Trainer(
        model=model,
        train_loader=loaders["train"],
        cfg=trainer_cfg,
        optimizer=optimizer,
        callbacks=callbacks,
        device=device,
    )

    trainer.train()


if __name__ == "__main__":
    main()
