from dataclasses import dataclass
from typing import Tuple, List
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    lr: float = 3e-4
    warmup_steps: int = 400
    clip_threshold: float = 1.0
    epochs: int = 12
    label_smoothing: float = 0.0

    act_ponder_penalty: float = 1e-2

    seed: int = 0
    device: str = "cpu"


#############################
# universal / special tokens
#############################

PAD, SOS, EOS = 0, 1, 2
NUM_SPECIAL = 3


def vocab_size(cfg: Config) -> int:
    return NUM_SPECIAL + cfg.num_symbols


def symbol_id(k: int) -> int:
    return NUM_SPECIAL + k


def decode_tokens(ids: List[int]) -> List[str]:
    out = []
    for i in ids:
        if i == PAD:
            out.append("<pad>")
        elif i == SOS:
            out.append("<sos>")
        elif i == EOS:
            out.append("<eos>")
        else:
            out.append(f"s{i - NUM_SPECIAL}")
    return out


##########################
# Data
##########################


def make_example(rng: random.Random, cfg: Config) -> Tuple[str, List[int], List[int]]:
    task = rng.choice(cfg.tasks)
    n = rng.randint(cfg.min_len, cfg.max_len)
    src = [symbol_id(rng.randrange(cfg.num_symbols)) for _ in range(n)]
    if task == "copy":
        tgt = list(src)
    elif task == "reverse":
        tgt = list(reversed(src))
    elif task == "sort":
        tgt = sorted(src)
    else:
        raise ValueError(task)
    return task, src, tgt


def build_dataset(cfg: Config):
    rng = random.Random(cfg.seed)
    train = [make_example(rng, cfg) for _ in range(cfg.train_examples)]
    rng_v = random.Random(cfg.seed + 10_000)
    valid = [make_example(rng_v, cfg) for _ in range(cfg.valid_examples)]
    return train, valid


def tensorize(example) -> Tuple[List[int], List[int]]:
    _, src, tgt = example
    src_ids = list(src) + [EOS]
    tgt_ids = [SOS] + list(tgt) + [EOS]
    return src_ids, tgt_ids


def make_batches(examples, cfg: Config, rng=None, shuffle=True):
    data = [tensorize(e) for e in examples]
    data.sort(key=lambda x: len(x[0]))  # length bucket to minimize padding
    batches = []
    for i in range(0, len(data), cfg.batch_size):
        chunk = data[i : i + cfg.batch_size]
        src_len = max(len(s) for s, _ in chunk)
        tgt_len = max(len(t) for _, t in chunk)
        src = torch.full((len(chunk), src_len), PAD, dtype=torch.long)
        tgt = torch.full((len(chunk), tgt_len), PAD, dtype=torch.long)
        for j, (s, t) in enumerate(chunk):
            src[j, : len(s)] = torch.tensor(s)
            tgt[j, : len(t)] = torch.tensor(t)
        batches.append((src, tgt))
    if shuffle:
        (rng or random.Random(cfg.seed)).shuffle(batches)
    return batches


################################################
# Coordinate (position + timestep) embedding
################################################


def coordinate_embedding(seq_len: int, step: int, d_model: int, device) -> torch.Tensor:
    pos = (
        torch.arange(seq_len, device=device).unsqueeze(1).float()
    )  # positions 0..L-1 as a column (L,1), so it broadcasts over dims
    j = torch.arange(
        0, d_model, 2, device=device
    ).float()  # even dim indices 0,2,4,..... -> one entry per sin/cos pair (d/2,)
    div = torch.pow(
        10000.0, j / d_model
    )  # per-dimension divisors 10000^(2j/d): big->low freq, small->high freq

    # The timestep term uses the SAME per-dimension frequency as the position term
    # (sin/cos of t / 10000^{2j/d}), NOT a scalar sin(t).
    t = torch.tensor(
        float(step), device=device
    )  # current recurrent step as a scalar tensor
    pe = torch.zeros(
        seq_len, d_model, device=device
    )  # empty table (L, d_model) to fill with the coordinate embedding
    pe[:, 0::2] = torch.sin(pos / div) + torch.sin(
        t / div
    )  # EVEN dims: sine of position + sine of timestep, summed (Eqn 6)
    pe[:, 1::2] = torch.cos(pos / div) + torch.cos(
        t / div
    )  # ODD dims: cosine of position + cosine of timestep, summed (Eqn 7)
    return pe  # (L, d_model) one coordinate embedding for THIS step, added to the state in act_halt
