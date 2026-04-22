import torch
import torch.nn as nn

from src.models.registry import registry

from .base import BaseCharModel


@registry.register("transformer")
class TransformerModel(BaseCharModel):
    """
    Decoder-only transformer for character-level prediction.

    Uses nn.TransformerEncoder with a causal mask — despite the name
    it is just a stack of self-attention blocks with no cross-attention,
    which is exactly what a decoder-only model needs.

    Unlike RNN/LSTM/GRU there is no recurrent state — returns None as state
    to satisfy the BaseCharModel interface.

    Architecture:
        TokenEmbedding + PositionalEmbedding
        → Dropout
        → N x TransformerEncoderLayer  (pre-norm, GELU, causal mask)
        → LayerNorm
        → Linear head  (weights tied to token embedding)
    """

    is_recurrent = False

    def __init__(self, cfg: dict):
        super().__init__(cfg=cfg)

        self.embed_size = cfg.get("embed_size", 64)
        self.num_heads = cfg.get("num_heads", 4)
        self.num_layers = cfg.get("num_layers", 4)
        self.ffn_dim = cfg.get("ffn_dim", 256)
        self.dropout = cfg.get("dropout", 0.1)
        self.seq_len = cfg.get("seq_len", 100)
        self.vocab_size = cfg["vocab_size"]

        self.token_emb = nn.Embedding(self.vocab_size, self.embed_size)
        self.pos_emb = nn.Embedding(self.seq_len, self.embed_size)
        self.drop = nn.Dropout(self.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_size,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            dropout=self.dropout,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
            enable_nested_tensor=False,
        )

        self.norm = nn.LayerNorm(self.embed_size)
        self.head = nn.Linear(self.embed_size, self.vocab_size, bias=False)

        # Weight tying — share embedding and output projection matrix.
        # Reduces parameters and improves perplexity.
        self.head.weight = self.token_emb.weight

    def forward(
        self,
        x: torch.Tensor,
        h: None = None,  # unused — transformer has no recurrent state
    ) -> tuple[torch.Tensor, None]:
        B, T = x.shape
        assert (
            T <= self.seq_len
        ), f"Sequence length {T} exceeds model max {self.seq_len}"

        pos = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        x = self.drop(self.token_emb(x) + self.pos_emb(pos))  # (B, T, E)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        x = self.transformer(x, mask=mask, is_causal=True)  # (B, T, E)
        x = self.norm(x)
        logits = self.head(x)  # (B, T, V)
        return logits, None
