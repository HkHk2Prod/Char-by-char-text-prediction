"""
Single source of truth for all training configuration.

Adding a new argument:
    1. Add a field to RunConfig with its default value.
    Done. argparse, YAML loading, and defaults all pick it up automatically.

Usage:
    cfg = RunConfig.parse()

    cfg = RunConfig(model="lstm", epochs=50)

    cfg.save("experiments/.../config.yaml")
    cfg = RunConfig.load("experiments/.../config.yaml")
"""

import argparse
import typing
from dataclasses import asdict, dataclass, fields

import yaml


@dataclass
class RunConfig:
    # Model
    model: str = "rnn"
    checkpoint: str | None = None

    # Data
    dataset: str = "test"
    data_dir: str = "data/raw"
    seq_len: int = 100
    batch_size: int = 64
    num_workers: int = 0
    lower_case: bool = False
    test_ratio: float = 0.05
    val_ratio: float = 0.05

    # Optimisation
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 5.0

    # Model hyperparameters
    embed_size: int = 64
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    num_heads: int = 4
    ffn_dim: int = 256

    # Callbacks
    log_every: int | None = None
    eval_every: int = 1
    save_every: int = 1
    gen_every: int = 1
    gen_length: int = 200
    patience: int = 5

    # Misc
    device: str | None = None
    save_dir: str = "Model_files"

    # ── Parsing ───────────────────────────────────────────────────────────────

    @classmethod
    def parse(cls) -> "RunConfig":
        """
        Build RunConfig from CLI args, optionally layered on top of a YAML file.

        Priority (highest to lowest):
            1. CLI args        (non-None values)
            2. YAML config     (if --config is provided)
            3. Dataclass defaults
        """
        args = cls._build_parser().parse_args()
        return cls._resolve(args)

    @classmethod
    def _build_parser(cls) -> argparse.ArgumentParser:
        """Auto-generate argparse arguments from dataclass fields and their types."""
        p = argparse.ArgumentParser(description="Train a character-level model.")
        hints = typing.get_type_hints(cls)  # resolves "int | None" → int | None

        p.add_argument("--config", default=None, help="Path to YAML config")

        for f in fields(cls):
            typ = hints[f.name]
            name = f"--{f.name}"
            origin = getattr(typ, "__origin__", None)
            args = getattr(typ, "__args__", None)

            # bool → store_true flag
            if typ is bool:
                p.add_argument(name, action="store_true", default=None)

            # list[str] → nargs="+"
            elif origin is list:
                p.add_argument(name, nargs="+", default=None)

            # X | None → extract X as the type
            elif origin is typing.Union or str(origin) == "types.UnionType":
                if not args or not (
                    inner := next((a for a in args if a is not type(None)), None)
                ):
                    p.add_argument(name, default=None)
                    continue
                p.add_argument(name, type=inner, default=None)
            # plain int / float / str
            elif typ in (int, float, str):
                p.add_argument(name, type=typ, default=None)

            else:
                # fallback — accept as string
                p.add_argument(name, default=None)

        return p

    @classmethod
    def _resolve(cls, args: argparse.Namespace) -> "RunConfig":
        base = asdict(cls())

        if args.config is not None:
            with open(args.config) as f:
                base.update(yaml.safe_load(f) or {})

        cli = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
        base.update(cli)

        # Coerce values to the correct type from the dataclass field hints
        hints = typing.get_type_hints(cls)
        for k, v in base.items():
            if v is None:
                continue
            typ = hints.get(k)
            # unwrap X | None → X
            args_ = getattr(typ, "__args__", None)
            if args_:
                typ = next((a for a in args_ if a is not type(None)), None)
            if typ in (int, float, str, bool) and not isinstance(v, typ):
                base[k] = typ(v)

        return cls(**base)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str) -> "RunConfig":
        with open(path) as f:
            return cls(**yaml.safe_load(f))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def model_config(self, vocab_size: int) -> dict:
        return {
            "model": self.model,
            "vocab_size": vocab_size,
            "embed_size": self.embed_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
            "seq_len": self.seq_len,
        }

    def __str__(self) -> str:
        lines = [f"  {k}: {v}" for k, v in asdict(self).items()]
        return "RunConfig:\n" + "\n".join(lines)
