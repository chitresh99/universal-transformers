import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


######################
# Config
######################


@dataclass
class Config:
    d_model: int = 128
    num_heads: int = 4
    d_ff: int = 256
    max_steps: int = 6
    dropout: float = 0.1
    act_epsilon: float = 0.01

    tasks: Tuple[str, ...] = ("copy", "reverse", "sort")
    num_symbols: int = 12
    min_len: int = 4
    max_len: int = 10

    train_examples: int = 20000
    valid_examples: int = 500

    batch_size: int = 64
    lr: float = 3e-4
    warmup_steps: int = 400
    clip_threshold: float = 1.0
    epochs: int = 30
    label_smoothing: float = 0.0

    act_ponder_penalty: float = 1e-3

    seed: int = 0
    device: str = "cpu"


######################
# Vocabulary
######################


PAD, SOS, EOS = 0, 1, 2
# each source starts with a task token so one model can learn all tasks.
# without it, the same source would have several correct targets.
TASK_NAMES = ("copy", "reverse", "sort")
NUM_SPECIAL = 3 + len(TASK_NAMES)


def vocab_size(cfg: Config) -> int:
    return NUM_SPECIAL + cfg.num_symbols


def task_id(task: str) -> int:
    return 3 + TASK_NAMES.index(task)


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
        elif i < NUM_SPECIAL:
            out.append(f"<{TASK_NAMES[i - 3]}>")
        else:
            out.append(f"s{i - NUM_SPECIAL}")
    return out


######################
# Data
######################


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
    task, src, tgt = example
    src_ids = [task_id(task)] + list(src) + [EOS]
    tgt_ids = [SOS] + list(tgt) + [EOS]
    return src_ids, tgt_ids


def make_batches(examples, cfg: Config, rng=None, shuffle=True):
    # sorting by length keeps padding low; shuffling the batch order
    # (not the examples) keeps training from seeing lengths in order.
    data = [tensorize(e) for e in examples]
    data.sort(key=lambda x: len(x[0]))
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


######################
# Coordinate embedding
######################


def coordinate_embedding(seq_len: int, step: int, d_model: int, device) -> torch.Tensor:
    # paper eqns 6-7: a sinusoidal position term plus a sinusoidal timestep
    # term, both over the same frequency bank, so the block knows where each
    # position is and which refinement step it is on.
    pos = torch.arange(seq_len, device=device).unsqueeze(1).float()
    j = torch.arange(0, d_model, 2, device=device).float()
    div = torch.pow(10000.0, j / d_model)
    t = torch.tensor(float(step), device=device)
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(pos / div) + torch.sin(t / div)
    pe[:, 1::2] = torch.cos(pos / div) + torch.cos(t / div)
    return pe


######################
# Model
######################


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.d_model % cfg.num_heads == 0
        self.h = cfg.num_heads
        self.dk = cfg.d_model // cfg.num_heads
        self.q = nn.Linear(cfg.d_model, cfg.d_model)
        self.k = nn.Linear(cfg.d_model, cfg.d_model)
        self.v = nn.Linear(cfg.d_model, cfg.d_model)
        self.o = nn.Linear(cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, query, key_value, mask=None):
        B, Lq, _ = query.shape
        Lk = key_value.size(1)

        def split(x, L):
            return x.view(B, L, self.h, self.dk).transpose(1, 2)

        Q = split(self.q(query), Lq)
        K = split(self.k(key_value), Lk)
        V = split(self.v(key_value), Lk)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = self.drop(F.softmax(scores, dim=-1))
        ctx = torch.matmul(attn, V)
        ctx = ctx.transpose(1, 2).contiguous().view(B, Lq, self.h * self.dk)
        return self.o(ctx)


class Transition(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(self, x):
        return self.net(x)


class UTBlock(nn.Module):
    def __init__(self, cfg: Config, is_decoder: bool):
        super().__init__()
        self.is_decoder = is_decoder
        self.self_attn = MultiHeadSelfAttention(cfg)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        if is_decoder:
            self.cross_attn = MultiHeadSelfAttention(cfg)
            self.norm_cross = nn.LayerNorm(cfg.d_model)
        self.transition = Transition(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        # halting unit for act. created here, not lazily, so the optimizer
        # always sees its parameters.
        self.halt = nn.Linear(cfg.d_model, 1)

    def forward(self, h, self_mask=None, enc=None, cross_mask=None):
        # one refinement step (paper eqns 4-5). h already includes the
        # coordinate embedding. the same block is reused at every step.
        a = self.self_attn(h, h, mask=self_mask)
        h = self.norm1(h + self.drop(a))
        if self.is_decoder:
            c = self.cross_attn(h, enc, mask=cross_mask)
            h = self.norm_cross(h + self.drop(c))
        t = self.transition(h)
        h = self.norm2(h + self.drop(t))
        return h


######################
# Adaptive Computation Time
######################


def act_halt(block: UTBlock, h0, cfg: Config, self_mask=None, enc=None, cross_mask=None):
    # dynamic halting from paper appendix c (act, graves 2016). each position
    # accumulates halting probability step by step and stops once the sum
    # passes the threshold. all tensors below are per position, shape (b, l).
    device = h0.device
    B, L, D = h0.shape
    threshold = 1.0 - cfg.act_epsilon

    halting_prob = torch.zeros(B, L, device=device)
    remainders = torch.zeros(B, L, device=device)
    n_updates = torch.zeros(B, L, device=device)
    state = h0

    step = 0
    while step < cfg.max_steps:
        still_running = (halting_prob < 1.0).float()

        coord = coordinate_embedding(L, step + 1, D, device).unsqueeze(0)
        p = torch.sigmoid(block.halt(state + coord)).squeeze(-1)

        # positions that cross the threshold this step halt now; the rest
        # keep running.
        new_halted = ((halting_prob + p * still_running) > threshold).float() * still_running
        still_running = ((halting_prob + p * still_running) <= threshold).float() * still_running

        halting_prob = halting_prob + p * still_running
        # a halting position gets the leftover probability (the remainder)
        # as its final weight, so its weights over all steps sum to 1.
        remainders = remainders + new_halted * (1.0 - halting_prob)
        halting_prob = halting_prob + new_halted * remainders
        n_updates = n_updates + still_running + new_halted

        # weight of this step's update: p if still running, the remainder if
        # halting now, 0 if already halted.
        update_weights = (p * still_running + new_halted * remainders).unsqueeze(-1)

        transformed = block(state + coord, self_mask=self_mask, enc=enc, cross_mask=cross_mask)
        # a weight of 0 leaves the state unchanged, which is how halted
        # positions stay frozen.
        state = transformed * update_weights + state * (1.0 - update_weights)

        step += 1
        if still_running.sum() == 0:
            break

    # n_updates is a count with no gradient; the remainders are what the
    # optimizer can move, so they carry the penalty. averaged over all
    # positions, padding included.
    ponder_cost = (n_updates + remainders).mean()
    return state, ponder_cost, n_updates.detach()


######################
# Universal Transformer
######################


class UniversalTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        V = vocab_size(cfg)
        self.src_embed = nn.Embedding(V, cfg.d_model, padding_idx=PAD)
        self.tgt_embed = nn.Embedding(V, cfg.d_model, padding_idx=PAD)
        # embeddings are scaled by sqrt(d_model) in encode/decode. starting at
        # std d_model^-0.5 makes them about 1 after scaling, the same size as
        # the coordinate embedding. pytorch's default (std 1) would swamp it.
        for emb in (self.src_embed, self.tgt_embed):
            nn.init.normal_(emb.weight, 0.0, cfg.d_model ** -0.5)
            with torch.no_grad():
                emb.weight[PAD].zero_()
        self.encoder = UTBlock(cfg, is_decoder=False)
        self.decoder = UTBlock(cfg, is_decoder=True)
        self.generator = nn.Linear(cfg.d_model, V)
        self.scale = math.sqrt(cfg.d_model)

    def encode(self, src):
        src_mask = (src != PAD).unsqueeze(1).unsqueeze(1)
        h = self.src_embed(src) * self.scale
        enc, enc_pc, enc_n = act_halt(self.encoder, h, self.cfg, self_mask=src_mask)
        return enc, src_mask, enc_pc, enc_n

    def decode(self, tgt_in, enc, src_mask):
        B, Lt = tgt_in.shape
        device = tgt_in.device
        pad_mask = (tgt_in != PAD).unsqueeze(1).unsqueeze(1)
        causal = torch.tril(torch.ones(Lt, Lt, device=device)).bool()
        self_mask = pad_mask & causal.unsqueeze(0).unsqueeze(0)
        h = self.tgt_embed(tgt_in) * self.scale
        dec, dec_pc, dec_n = act_halt(
            self.decoder, h, self.cfg, self_mask=self_mask, enc=enc, cross_mask=src_mask
        )
        return self.generator(dec), dec_pc, dec_n

    def forward(self, src, tgt_in):
        enc, src_mask, enc_pc, enc_n = self.encode(src)
        logits, dec_pc, dec_n = self.decode(tgt_in, enc, src_mask)
        return logits, enc_pc + dec_pc, enc_n, dec_n


######################
# Training
######################


def lr_at(step, cfg: Config):
    # linear warmup, then decay with 1 / sqrt(step). the last factor rescales
    # it so the peak, at step == warmup_steps, equals cfg.lr.
    step = max(step, 1)
    return cfg.lr * min(step ** -0.5, step * cfg.warmup_steps ** -1.5) * (cfg.warmup_steps ** 0.5)


def compute_loss(model, src, tgt, cfg: Config):
    # teacher forcing: the decoder reads the target shifted right and predicts
    # it shifted left.
    tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
    logits, ponder_cost, enc_n, dec_n = model(src, tgt_in)
    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        tgt_out.reshape(-1),
        ignore_index=PAD,
        label_smoothing=cfg.label_smoothing,
    )
    loss = ce + cfg.act_ponder_penalty * ponder_cost
    # for logging only. includes padding positions, so it is lower than the
    # per-task figures from ponder_by_task.
    avg_steps = 0.5 * (enc_n.float().mean().item() + dec_n.float().mean().item())
    return loss, ce.item(), avg_steps


def train(model, train_examples, cfg: Config, log_every=1):
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), eps=1e-9)
    rng = random.Random(cfg.seed)
    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        batches = make_batches(train_examples, cfg, rng=rng)
        tot_ce, tot_steps, nb = 0.0, 0.0, 0
        for src, tgt in batches:
            src, tgt = src.to(cfg.device), tgt.to(cfg.device)
            global_step += 1
            for g in opt.param_groups:
                g["lr"] = lr_at(global_step, cfg)
            opt.zero_grad()
            loss, ce, avg_steps = compute_loss(model, src, tgt, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_threshold)
            opt.step()
            tot_ce += ce
            tot_steps += avg_steps
            nb += 1
        if epoch % log_every == 0 or epoch == 1:
            print(
                f"  epoch {epoch:3d}  ce {tot_ce/nb:6.4f}  "
                f"ppl {math.exp(tot_ce/nb):7.3f}  avg_steps {tot_steps/nb:4.2f}"
            )
    return model


######################
# Decoding
######################


@torch.no_grad()
def greedy_decode(model, src, cfg: Config, max_len=None):
    model.eval()
    max_len = max_len or (cfg.max_len + 4)
    enc, src_mask, _, _ = model.encode(src)
    ys = torch.full((src.size(0), 1), SOS, dtype=torch.long, device=src.device)
    for _ in range(max_len):
        logits, _, _ = model.decode(ys, enc, src_mask)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        ys = torch.cat([ys, nxt], dim=1)
        # decodes the whole batch until every sequence has produced <eos>;
        # strip() cuts each one at its own first <eos>.
        if (nxt == EOS).all():
            break
    return ys


def strip(ids: List[int]) -> List[int]:
    out = []
    for i in ids:
        if i == SOS:
            continue
        if i == EOS:
            break
        if i == PAD:
            continue
        out.append(i)
    return out


######################
# Evaluation
######################


@torch.no_grad()
def evaluate(model, examples, cfg: Config):
    model.eval()
    char_correct, char_total, seq_correct, seq_total = 0, 0, 0, 0
    for batch in make_batches(examples, cfg, shuffle=False):
        src, tgt = batch
        src = src.to(cfg.device)
        pred = greedy_decode(model, src, cfg)
        for b in range(src.size(0)):
            p = strip(pred[b].tolist())
            g = strip(tgt[b].tolist())
            seq_total += 1
            seq_correct += int(p == g)
            # character accuracy compares position by position, so a missing
            # or extra token shifts everything after it.
            for k in range(len(g)):
                char_total += 1
                char_correct += int(k < len(p) and p[k] == g[k])
    return char_correct / max(char_total, 1), seq_correct / max(seq_total, 1)


@torch.no_grad()
def evaluate_by_task(model, examples, cfg: Config):
    results = {}
    for task in cfg.tasks:
        subset = [e for e in examples if e[0] == task]
        if subset:
            results[task] = evaluate(model, subset, cfg)
    return results


@torch.no_grad()
def ponder_by_task(model, examples, cfg: Config):
    """average ponder steps per task, on content positions only.

    encoder: source symbols, excluding <task>, <eos> and padding.
    decoder: teacher-forced target positions, excluding padding.
    at cap: share of encoder content positions that hit max_steps.
    """
    model.eval()
    results = {}
    for task in cfg.tasks:
        subset = [e for e in examples if e[0] == task]
        if not subset:
            continue
        enc_sum = enc_cnt = enc_cap = dec_sum = dec_cnt = 0.0
        for src, tgt in make_batches(subset, cfg, shuffle=False):
            src, tgt = src.to(cfg.device), tgt.to(cfg.device)
            _, _, enc_n, dec_n = model(src, tgt[:, :-1])
            content = (src >= NUM_SPECIAL).float()
            enc_sum += (enc_n * content).sum().item()
            enc_cnt += content.sum().item()
            enc_cap += ((enc_n >= cfg.max_steps).float() * content).sum().item()
            real = (tgt[:, :-1] != PAD).float()
            dec_sum += (dec_n * real).sum().item()
            dec_cnt += real.sum().item()
        results[task] = (enc_sum / enc_cnt, dec_sum / dec_cnt, enc_cap / enc_cnt)
    return results


######################
# Checks
######################


def check_shapes(model, cfg: Config):
    B, Ls, Lt = 4, 7, 9
    V = vocab_size(cfg)
    src = torch.randint(NUM_SPECIAL, V, (B, Ls))
    tgt_in = torch.randint(NUM_SPECIAL, V, (B, Lt))
    logits, ponder_cost, enc_n, dec_n = model(src, tgt_in)
    assert logits.shape == (B, Lt, V), logits.shape
    assert enc_n.shape == (B, Ls), enc_n.shape
    assert enc_n.max().item() <= cfg.max_steps, "ponder exceeds max_steps"
    print(f"  logits          {tuple(logits.shape)}")
    print(f"  enc ponder      {tuple(enc_n.shape)}  (steps per source position)")
    print(f"  dec ponder      {tuple(dec_n.shape)}  (steps per target position)")
    print(f"  ponder range    [{enc_n.min():.0f}, {enc_n.max():.0f}] of max {cfg.max_steps}")
    print(f"  max_steps (T)   {cfg.max_steps}")
    print(f"  parameters      {sum(p.numel() for p in model.parameters()):,}")
    print(f"  tied block?     yes")


def check_overfit(cfg: Config):
    # a correct model must memorise 32 examples. dropout and the ponder penalty
    # are off so nothing works against memorisation, and the small batch and
    # short warmup let the learning rate actually reach its peak.
    small = Config(**{**cfg.__dict__, "epochs": 300, "batch_size": 8,
                      "train_examples": 32, "warmup_steps": 100, "lr": 1e-3,
                      "act_ponder_penalty": 0.0, "dropout": 0.0})
    rng = random.Random(small.seed)
    subset = [make_example(rng, small) for _ in range(32)]
    model = UniversalTransformer(small).to(small.device)
    train(model, subset, small, log_every=50)
    char, seq = evaluate(model, subset, small)
    print(f"  memorized       char-acc {char:.0%}  seq-acc {seq:.0%}  of 32 examples")
    return seq


######################
# Main
######################


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    train_ex, valid_ex = build_dataset(cfg)

    print("Data")
    task, src, tgt = train_ex[0]
    print(f"  task            {task}")
    print(f"  source          {decode_tokens(src)}")
    print(f"  target          {decode_tokens(tgt)}")
    si, ti = tensorize(train_ex[0])
    print(f"  source ids      {si}   (<task> + content + <eos>)")
    print(f"  target ids      {ti}   (<sos> + content + <eos>)")
    print(f"  vocab           {vocab_size(cfg)} ({cfg.num_symbols} content symbols)")
    print(f"  tasks           {cfg.tasks}   len {cfg.min_len}-{cfg.max_len}")
    print(f"  examples        {len(train_ex)} train / {len(valid_ex)} valid")

    print("\nShapes")
    model = UniversalTransformer(cfg).to(cfg.device)
    check_shapes(model, cfg)

    print("\nOverfit check")
    check_overfit(cfg)

    print("\nTraining")
    model = UniversalTransformer(cfg).to(cfg.device)
    train(model, train_ex, cfg, log_every=5)

    print("\nHeld-out accuracy (char-acc / seq-acc)")
    for task, (char, seq) in evaluate_by_task(model, valid_ex, cfg).items():
        print(f"  {task:8s}        char {char:.1%}   seq {seq:.1%}")
    char, seq = evaluate(model, valid_ex, cfg)
    print(f"  {'overall':8s}        char {char:.1%}   seq {seq:.1%}")

    print(f"\nPonder by task (held-out, content positions, max {cfg.max_steps})")
    for task, (enc_avg, dec_avg, at_cap) in ponder_by_task(model, valid_ex, cfg).items():
        print(f"  {task:8s}        enc {enc_avg:.2f}   dec {dec_avg:.2f}   enc at cap {at_cap:.0%}")

    print("\nSamples (with per-position ponder times)")
    model.eval()
    for ex in valid_ex[:3]:
        si, _ = tensorize(ex)
        src = torch.tensor([si], device=cfg.device)
        with torch.no_grad():
            enc, src_mask, _, enc_n = model.encode(src)
            pred = greedy_decode(model, src, cfg)
        p = strip(pred[0].tolist())
        print(f"  task            {ex[0]}")
        print(f"  source          {decode_tokens(ex[1])}")
        print(f"  target          {decode_tokens(ex[2])}")
        print(f"  predicted       {decode_tokens(p)}")
        ponder = enc_n[0].tolist()
        print(f"  enc ponder/pos  {[int(x) for x in ponder]}  "
              f"(avg {sum(ponder)/len(ponder):.2f} of max {cfg.max_steps}, incl. <task> and <eos>)")
        print()


if __name__ == "__main__":
    main()