from dataclasses import dataclass
from typing import Tuple

####################
# Config
####################

@dataclass
class Config:
    d_model: int = 128
    num_heads: int = 4
    d_ff: int = 256
    max_steps: int = 6
    dropout: float = 0.1
    act_epsilon: float = 0.01

    tasks: Tuple[str, ...] = ("copy", "reverse")
    num_symbols: int = 12 
    min_len: int = 4
    max_len: int = 10

    train_examples: int = 20000
    valid_examples: int = 500

    batch_size: int = 64
    lr:float = 3e-4
    warmup_steps: int = 400
    clip_threshold: float = 1.0
    epochs: int = 12
    label_smoothing: float = 0.0

    act_ponder_penalty: float = 1e-2

    seed: int = 0
    device: str = "cpu"