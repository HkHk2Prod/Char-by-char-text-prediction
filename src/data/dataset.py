from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class CharVocab:

    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.char_to_int = {ch: i for i, ch in enumerate(self.chars)}
        self.int_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def __len__(self):
        return len(self.chars)

    def encode(self, text):
        return [self.char_to_int[c] for c in text]

    def decode(self, indices):
        return "".join(self.int_to_char[i] for i in indices)

    def __str__(self):
        return f"size: {len(self)}" f"  chars: {repr(''.join(self.chars))}"


class TextDataset(Dataset):
    def __init__(
        self,
        seq_len: int = 100,
        file_path: str | Path | None = None,
        text: str | None = None,
        vocab: CharVocab | None = None,
    ):
        self.seq_len = seq_len
        self.text = text
        if file_path:
            with open(file_path, "r", encoding="utf-8") as f:
                self.text = f.read()
        assert isinstance(self.text, str)
        self.vocab = vocab or CharVocab(self.text)

    def __len__(self):
        assert self.text is not None
        return len(self.text) - self.seq_len

    def __getitem__(self, idx):
        assert self.text is not None
        chunk = self.text[idx : idx + self.seq_len + 1]
        encoded = self.vocab.encode(chunk)

        x = torch.tensor(encoded[:-1], dtype=torch.long)
        y = torch.tensor(encoded[1:], dtype=torch.long)
        return x, y

    @classmethod
    def generate_test_train(
        cls,
        file_path: str | Path | None = None,
        text: str | None = None,
        seq_len: int = 100,
        vocab: CharVocab | None = None,
        val_test_ratio: tuple[float, float] = (0.1, 0.1),
        lower_case: bool = False,
    ) -> tuple[TextDataset, TextDataset, TextDataset]:
        train_ratio = 1 - sum(val_test_ratio)
        val_ratio, test_ratio = val_test_ratio
        assert train_ratio >= 0 and val_ratio >= 0 and test_ratio >= 0
        if file_path:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        assert text is not None

        if lower_case:
            text = text.lower()

        # If the vocab was not preset in the argument, we create a unified vocab for all datasets.
        vocab = vocab or CharVocab(text)
        n = len(text)
        val_start = int(n * train_ratio)
        test_start = int(n * (train_ratio + val_ratio))
        train_ds = TextDataset(text=text[:val_start], seq_len=seq_len, vocab=vocab)
        val_ds = TextDataset(
            text=text[val_start:test_start], seq_len=seq_len, vocab=vocab
        )
        test_ds = TextDataset(text=text[test_start:n], seq_len=seq_len, vocab=vocab)
        return train_ds, val_ds, test_ds
