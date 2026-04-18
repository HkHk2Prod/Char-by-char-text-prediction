import torch
import torch.nn as nn
from .base import BaseCharModel

class RNNModel(BaseCharModel):
    """
    Simple RNN with embedding, dropout, and few layers. 
    """

    def __init__(self, vocab_size: int, cfg: dict):
        super().__init__(vocab_size=vocab_size, cfg=cfg)

        self.embed_size  = cfg.get("embed_size", 64)
        self.hidden_size = cfg.get("hidden_size", 256)
        self.num_layers  = cfg.get("num_layers", 2)
        self.dropout     = cfg.get("dropout", 0.3)

        self.embedding = nn.Embedding(vocab_size, self.embed_size)

        self.rnn = nn.RNN(
            input_size  = self.embed_size,
            hidden_size = self.hidden_size,
            num_layers  = self.num_layers,
            dropout     = self.dropout if self.num_layers > 1 else 0.0,
            batch_first = True,    # input/output shape: (batch, seq, features)
        )

        self.dropout_layer = nn.Dropout(self.dropout)
        self.head          = nn.Linear(self.hidden_size, vocab_size)
            
    def forward(
            self,
            x: torch.Tensor,
            h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(x)
        output, h = self.rnn(x, h)
        output = self.dropout_layer(output)
        logits = self.head(output)
        assert isinstance(h, torch.Tensor)
        return logits, h

