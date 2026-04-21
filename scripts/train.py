import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import TextDataset
from src.inference.predictor import Predictor
from src.models.autodiscover import autodiscover
from src.models.registry import registry
from src.models.RNN import RNNModel
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
    TestEvalCallback,
)
from src.training.trainer import Trainer, TrainerConfig


def parse_args():
    p = argparse.ArgumentParser(description="Train a character-level model.")

    p.add_argument("--model", required=True, choices=["rnn"])
    p.add_argument("--dataset", choices=["tiny_shakespeare"], default="tiny_shakespeare")
    p.add_argument("--val_ratio", default=0.1)
    p.add_argument("--test_ratio", default=0.1)
    p.add_argument("--data_dir", default="data/raw")
    p.add_argument("--seq_length", default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default=None)

    return p.parse_args()

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    match args.dataset:
        case "tiny_shakespeare":
            file_name = "tiny_shakespeare.txt"
        case _:
            raise ValueError("Unknown dataset")

    datasets = TextDataset.generate_test_train(file_path = data_dir / file_name,
                                                seq_length=args.seq_length,
                                                val_test_ratio=(args.val_ratio, args.test_ratio),
                                            ) 

    vocab = datasets[0].vocab

    loaders = {name: DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(name == "train"),
        num_workers=4,
        drop_last=(name != "train"),
        pin_memory=device.startswith("cuda"),
    ) for name, ds in zip(["train", "val", "test"], datasets)}

    autodiscover()
    model = RNNModel(cfg={"vocab_size": len(vocab)})
    predictor = Predictor(model, vocab, device=device)

    cfg = TrainerConfig()
    prompts = ["ROMEO:", "To be or not to be"]
    callbacks = [
        ConfigSaverCallback(),
        ModelInfoCallback(),
        EpochProgressCallback(),
        BatchLogCallback(log_every=50),    # ← new
        EvalCallback(loaders["val"]),
        LogCallback(),
        CheckpointCallback(),
        GenerationCallback(predictor=predictor, prompts=prompts),
        EarlyStoppingCallback(patience=5),
        TestEvalCallback(loaders["test"], predictor=predictor, prompts=prompts),
    ]


    trainer=Trainer(model=model, 
            train_loader=loaders["train"], 
            cfg=cfg,
            callbacks=callbacks,
            device=device)
    
    trainer.train()




if __name__ == "__main__":
    main()
              