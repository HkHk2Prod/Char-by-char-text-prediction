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
from src.training.config import RunConfig
from src.training.trainer import Trainer, TrainerConfig


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


def main() -> None:
    cfg = RunConfig.parse()
    data_dir = Path(cfg.data_dir)
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    prompts = None

    match cfg.dataset:
        case "shakespeare":
            file_name = "tiny_shakespeare.txt"
            prompts = ["ROMEO:", "To be or not to be"]
        case "test":
            file_name = "test_text.txt"
        case _:
            raise ValueError("Unknown dataset")

    datasets = TextDataset.generate_test_train(
        file_path=data_dir / file_name,
        seq_len=cfg.seq_len,
        val_test_ratio=(cfg.val_ratio, cfg.test_ratio),
        lower_case=cfg.lower_case,
    )

    vocab = datasets[0].vocab

    loaders = {
        name: DataLoader(
            ds,
            batch_size=cfg.batch_size,
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
    model = registry.build(cfg.model, config=cfg.model_config(len(vocab)))
    predictor = make_predictor(model, vocab, device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    trainer_cfg = TrainerConfig(
        grad_clip=cfg.grad_clip,
        epochs=cfg.epochs,
        save_dir=cfg.save_dir,
    )
    prompts = prompts or sample_prompts(
        datasets[2].text or ""
    )  # We sample_prompts from test text.
    prompts = prepare_prompts(prompts=prompts, vocab=vocab, lower_case=cfg.lower_case)

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
