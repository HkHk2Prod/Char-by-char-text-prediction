import torch
import torch.nn as nn

from src.models.registry import registry

from .base import BaseCharModel


@registry.register("lstm")
class RNNModel(BaseCharModel):
    """
    LSTM with embedding, dropout, and multiple layers.
    Hidden state is a tuple (h, c) unlike RNN's single tensor.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg=cfg)

        self.embed_size = cfg.get("embed_size", 64)
        self.hidden_size = cfg.get("hidden_size", 256)
        self.num_layers = cfg.get("num_layers", 2)
        self.dropout = cfg.get("dropout", 0.3)
        self.vocab_size = cfg["vocab_size"]

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)

        self.rnn = nn.LSTM(
            input_size=self.embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            batch_first=True,  # input/output shape: (batch, seq, features)
        )

        self.dropout_layer = nn.Dropout(self.dropout)
        self.head = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        h: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = self.embedding(x)
        output, h = self.rnn(x, h)
        output = self.dropout_layer(output)
        logits = self.head(output)
        assert isinstance(h, tuple)
        return logits, h
