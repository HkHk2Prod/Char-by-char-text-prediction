import argparse

from pathlib import Path
from torch.utils.data import DataLoader

from src.data.dataset import TextDataset
from src.models.RNN import RNNModel
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

    return p.parse_args()

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
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
    vocab_size = len(vocab)

    loaders = {name: DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(name=="train"),
        num_workers=4,
        pin_memory=True,
    ) for name, ds in zip(["train", "val", "test"], datasets)}

    cfg = TrainerConfig()

    model = RNNModel(vocab_size=vocab_size,
                     cfg={})

    trainer=Trainer(model=model, 
            train_loader=loaders["train"], 
            val_loader=loaders["val"], 
            cfg=cfg)
    
    trainer.train()




if __name__ == "__main__":
    main()
       