"""
reference.py — Reference PyTorch implementation of the toy decoder.

PURPOSE
=======
Generates numerical fixtures that the TypeScript implementation in
src/lib/transformer/ must match to within 1e-5. This is the authoritative
numerical source of truth for the project. Run it manually whenever the
toy-model definition in SPEC.md §4 changes; commit the regenerated fixtures.

WHY A SEPARATE PYTHON SCRIPT?
=============================
Two reasons:
  1. PyTorch is the de facto reference for "what the right answer is"
     in transformer maths. By generating fixtures from it we make sure
     our hand-written TS isn't just self-consistent but actually correct.
  2. It documents the toy model in a second, executable form. If our
     TS and this script disagree, one of them has a bug.

USAGE
=====
    pip install torch numpy
    python scripts/reference.py

Outputs JSON files into tests/unit/fixtures/. These are checked into git.

The TS unit tests load the JSON and assert max-abs-error < 1e-5 against
their own forward passes with the same seed and inputs.

CONVENTIONS — must match SPEC.md §4 and src/lib/transformer/* exactly
=====================================================================
- Pre-norm decoder block: x -> x + Attn(LN(x)); h -> h + FFN(LN(h))
- GELU: torch's "tanh" approximation (matches our TS impl)
- Positional encoding: sinusoidal (Vaswani 2017 §3.5)
- Causal mask: upper triangular -inf above the main diagonal
- Softmax: numerically stable (subtract row-max before exp)
- LayerNorm: epsilon = 1e-5
- Char tokenizer over the 64-char alphabet defined below
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants — keep in sync with SPEC.md §4 and src/lib/transformer/tokenizer.ts
# ---------------------------------------------------------------------------
ALPHABET = (
    "abcdefghijklmnopqrstuvwxyz"  # 26
    "0123456789"  # 10
    " .,!?;:'\"()-\n"  # 13
    "+-*/=<>[]{}@#$%^&_|"  # 19 (extras to round out to >=64)
)[:64]
assert len(ALPHABET) == 64, f"alphabet must be 64 chars, got {len(ALPHABET)}"
CHAR_TO_ID = {c: i for i, c in enumerate(ALPHABET)}

VOCAB_SIZE = 64
SEED = 42
EPS = 1e-5

# Default toy-model config (SPEC §4).
DEFAULT_CONFIG = dict(
    seq_len=8,
    d_model=16,
    n_heads=2,
    d_ff=32,
    n_blocks=2,
    vocab_size=VOCAB_SIZE,
    seed=SEED,
)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
def encode(text: str, seq_len: int) -> list[int]:
    """Encode `text` to token ids, truncate or pad with id 0 to `seq_len`."""
    ids = [CHAR_TO_ID.get(c, 0) for c in text[:seq_len]]
    while len(ids) < seq_len:
        ids.append(0)
    return ids


# ---------------------------------------------------------------------------
# Positional encoding (sinusoidal, Vaswani 2017)
# ---------------------------------------------------------------------------
def positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe


# ---------------------------------------------------------------------------
# Weight init — deterministic from a seed.
# ---------------------------------------------------------------------------
def init_weights(config: dict) -> dict[str, torch.Tensor]:
    """Initialise all weights from a single seed.

    Uses normal(0, 0.02) like GPT-2 for the matrices; zeros for biases;
    γ=1, β=0 for layernorm. Order of draws is significant — keep in sync
    with src/lib/transformer/model.ts.
    """
    g = torch.Generator().manual_seed(config["seed"])
    d_model = config["d_model"]
    d_ff = config["d_ff"]
    vocab_size = config["vocab_size"]
    n_blocks = config["n_blocks"]

    def normal(*shape: int) -> torch.Tensor:
        return torch.empty(*shape).normal_(mean=0.0, std=0.02, generator=g)

    weights: dict[str, torch.Tensor] = {
        "tok_emb": normal(vocab_size, d_model),
    }
    for i in range(n_blocks):
        prefix = f"block_{i}"
        weights[f"{prefix}.ln1.gamma"] = torch.ones(d_model)
        weights[f"{prefix}.ln1.beta"] = torch.zeros(d_model)
        weights[f"{prefix}.attn.W_q"] = normal(d_model, d_model)
        weights[f"{prefix}.attn.W_k"] = normal(d_model, d_model)
        weights[f"{prefix}.attn.W_v"] = normal(d_model, d_model)
        weights[f"{prefix}.attn.W_o"] = normal(d_model, d_model)
        weights[f"{prefix}.ln2.gamma"] = torch.ones(d_model)
        weights[f"{prefix}.ln2.beta"] = torch.zeros(d_model)
        weights[f"{prefix}.ffn.W1"] = normal(d_model, d_ff)
        weights[f"{prefix}.ffn.b1"] = torch.zeros(d_ff)
        weights[f"{prefix}.ffn.W2"] = normal(d_ff, d_model)
        weights[f"{prefix}.ffn.b2"] = torch.zeros(d_model)
    weights["ln_final.gamma"] = torch.ones(d_model)
    weights["ln_final.beta"] = torch.zeros(d_model)
    # We tie the output head to the token embedding (common in tiny GPTs).
    return weights


# ---------------------------------------------------------------------------
# Ops — written explicitly (no nn.Module) so each op maps 1:1 to a TS file.
# ---------------------------------------------------------------------------
def layernorm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + EPS) * gamma + beta


def gelu(x: torch.Tensor) -> torch.Tensor:
    # tanh approximation — matches our TS impl (see src/lib/transformer/gelu.ts).
    return (
        0.5
        * x
        * (
            1.0
            + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
            )
        )
    )


def causal_mask(seq_len: int) -> torch.Tensor:
    """Upper triangular -inf above the main diagonal; 0 elsewhere."""
    m = torch.zeros(seq_len, seq_len)
    m.masked_fill_(torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool(), float("-inf"))
    return m


def multi_head_attention(
    x: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_o: torch.Tensor,
    n_heads: int,
) -> dict[str, torch.Tensor]:
    """Causal multi-head self-attention. Returns the output and a trace dict.

    Shapes:
        x:   [seq_len, d_model]
        W_*: [d_model, d_model]
        out: [seq_len, d_model]
    """
    seq_len, d_model = x.shape
    d_head = d_model // n_heads
    assert d_model % n_heads == 0

    Q = x @ W_q  # [S, D]
    K = x @ W_k
    V = x @ W_v

    # Reshape to [n_heads, S, d_head]
    Q_h = Q.view(seq_len, n_heads, d_head).transpose(0, 1)
    K_h = K.view(seq_len, n_heads, d_head).transpose(0, 1)
    V_h = V.view(seq_len, n_heads, d_head).transpose(0, 1)

    scores = Q_h @ K_h.transpose(-2, -1) / math.sqrt(d_head)  # [H, S, S]
    mask = causal_mask(seq_len)
    masked = scores + mask  # broadcasts over heads
    weights = F.softmax(masked, dim=-1)
    head_out = weights @ V_h  # [H, S, d_head]

    concat = head_out.transpose(0, 1).contiguous().view(seq_len, d_model)
    out = concat @ W_o

    return {
        "Q": Q,
        "K": K,
        "V": V,
        "scores": scores,
        "mask": mask,
        "weights": weights,
        "head_out": head_out,
        "out": out,
    }


def ffn(
    x: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    return gelu(x @ W1 + b1) @ W2 + b2


def block_forward(x: torch.Tensor, w: dict, prefix: str, n_heads: int) -> torch.Tensor:
    """Pre-norm decoder block (SPEC §4)."""
    h = x + multi_head_attention(
        layernorm(x, w[f"{prefix}.ln1.gamma"], w[f"{prefix}.ln1.beta"]),
        w[f"{prefix}.attn.W_q"],
        w[f"{prefix}.attn.W_k"],
        w[f"{prefix}.attn.W_v"],
        w[f"{prefix}.attn.W_o"],
        n_heads,
    )["out"]
    h = h + ffn(
        layernorm(h, w[f"{prefix}.ln2.gamma"], w[f"{prefix}.ln2.beta"]),
        w[f"{prefix}.ffn.W1"],
        w[f"{prefix}.ffn.b1"],
        w[f"{prefix}.ffn.W2"],
        w[f"{prefix}.ffn.b2"],
    )
    return h


def forward(token_ids: list[int], config: dict, weights: dict) -> dict:
    """Full forward pass. Returns a dict with logits and intermediates."""
    seq_len = config["seq_len"]
    d_model = config["d_model"]
    n_blocks = config["n_blocks"]
    n_heads = config["n_heads"]

    ids = torch.tensor(token_ids, dtype=torch.long)
    tok_emb = weights["tok_emb"][ids]  # [S, D]
    pos_emb = positional_encoding(seq_len, d_model)
    x = tok_emb + pos_emb

    for i in range(n_blocks):
        x = block_forward(x, weights, f"block_{i}", n_heads)

    x_final = layernorm(x, weights["ln_final.gamma"], weights["ln_final.beta"])
    logits = x_final @ weights["tok_emb"].T  # tied head

    return {
        "tok_emb": tok_emb,
        "pos_emb": pos_emb,
        "x_after_emb": tok_emb + pos_emb,
        "x_final": x_final,
        "logits": logits,
    }


# ---------------------------------------------------------------------------
# Fixture writer
# ---------------------------------------------------------------------------
def to_jsonable(t: torch.Tensor) -> list:
    return t.detach().cpu().to(torch.float64).tolist()


def write_fixture(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Animation states (brief 27): the site's animations, rebuilt from scratch.
#
# The site's widgets do not use the PyTorch-seeded weights above: they draw
# their weights in the browser with mulberry32 + Box-Muller
# (src/lib/transformer/random.ts, init.ts). To check what the animations show,
# this section ports that initialiser, runs the reference ops above in
# float64 on those weights, and builds the animations' state lists the way
# src/lib/anim/*-steps.ts describe them. tests/unit/anim-frames.test.ts
# requires the TypeScript states to match these (to 1e-12) and the captions
# built from both to be identical; tests/e2e/anim-frames.spec.ts requires the
# captions on the page to be the ones built from these states.
#
# This section adds a file (animations.json); it changes no other fixture.
# ---------------------------------------------------------------------------
U32 = 0xFFFFFFFF


def mulberry32(seed: int):
    """Port of src/lib/transformer/random.ts:mulberry32 (uint32 arithmetic)."""
    state = seed & U32

    def rng() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & U32
        t = state
        t = ((t ^ (t >> 15)) * (t | 1)) & U32
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & U32)) & U32
        return ((t ^ (t >> 14)) & U32) / 4294967296

    return rng


def normal_sampler(rng):
    """Port of random.ts:normalSampler (Box-Muller, one sample per call)."""

    def sample() -> float:
        u1 = max(rng(), 1e-12)
        u2 = rng()
        return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

    return sample


def init_weights_site(config: dict) -> dict[str, torch.Tensor]:
    """Port of src/lib/transformer/init.ts: the weights the site's widgets use.

    Same draw order as the TypeScript (embedding table, then per block
    W_q, W_k, W_v, W_o, W1, W2), N(0, 0.02), float64, same keys as
    `init_weights`.
    """
    sample = normal_sampler(mulberry32(config["seed"]))
    d, f, v = config["d_model"], config["d_ff"], config["vocab_size"]

    def normal(rows: int, cols: int) -> torch.Tensor:
        return torch.tensor(
            [[sample() * 0.02 for _ in range(cols)] for _ in range(rows)],
            dtype=torch.float64,
        )

    ones = lambda n: torch.ones(n, dtype=torch.float64)  # noqa: E731
    zeros = lambda n: torch.zeros(n, dtype=torch.float64)  # noqa: E731
    w: dict[str, torch.Tensor] = {"tok_emb": normal(v, d)}
    for i in range(config["n_blocks"]):
        p = f"block_{i}"
        w[f"{p}.ln1.gamma"], w[f"{p}.ln1.beta"] = ones(d), zeros(d)
        for name in ("W_q", "W_k", "W_v", "W_o"):
            w[f"{p}.attn.{name}"] = normal(d, d)
        w[f"{p}.ln2.gamma"], w[f"{p}.ln2.beta"] = ones(d), zeros(d)
        w[f"{p}.ffn.W1"], w[f"{p}.ffn.b1"] = normal(d, f), zeros(f)
        w[f"{p}.ffn.W2"], w[f"{p}.ffn.b2"] = normal(f, d), zeros(d)
    w["ln_final.gamma"], w["ln_final.beta"] = ones(d), zeros(d)
    return w


def positional_encoding_f64(seq_len: int, d_model: int) -> torch.Tensor:
    """`positional_encoding` in float64 (the site computes in doubles)."""
    pe = torch.zeros(seq_len, d_model, dtype=torch.float64)
    position = torch.arange(0, seq_len, dtype=torch.float64).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float64)
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe


def site_trace(token_ids: list[int], config: dict, w: dict) -> dict:
    """Forward pass with every intermediate, using the reference ops above."""
    S, D, H = config["seq_len"], config["d_model"], config["n_heads"]
    ids = torch.tensor(token_ids, dtype=torch.long)
    tok_emb = w["tok_emb"][ids]
    pos_emb = positional_encoding_f64(S, D)
    x = tok_emb + pos_emb
    out = {"tok_emb": tok_emb, "pos_emb": pos_emb, "x0": x, "blocks": []}
    for i in range(config["n_blocks"]):
        p = f"block_{i}"
        ln1 = layernorm(x, w[f"{p}.ln1.gamma"], w[f"{p}.ln1.beta"])
        attn = multi_head_attention(
            ln1, w[f"{p}.attn.W_q"], w[f"{p}.attn.W_k"], w[f"{p}.attn.W_v"],
            w[f"{p}.attn.W_o"], H,
        )
        h = x + attn["out"]
        ln2 = layernorm(h, w[f"{p}.ln2.gamma"], w[f"{p}.ln2.beta"])
        pre = ln2 @ w[f"{p}.ffn.W1"] + w[f"{p}.ffn.b1"]
        act = gelu(pre)
        y = h + (act @ w[f"{p}.ffn.W2"] + w[f"{p}.ffn.b2"])
        out["blocks"].append(
            {"ln1": ln1, "attn": attn, "h": h, "ln2": ln2, "pre": pre,
             "act": act, "ffn_out": act @ w[f"{p}.ffn.W2"] + w[f"{p}.ffn.b2"],
             "y": y}
        )
        x = y
    out["x_final"] = layernorm(x, w["ln_final.gamma"], w["ln_final.beta"])
    out["logits"] = out["x_final"] @ w["tok_emb"].T
    return out


def vec(t: torch.Tensor) -> list[float]:
    return [float(v) for v in t.tolist()]


def embed_states(text: str, config: dict, w: dict, pos: int) -> list[dict]:
    """Port of src/lib/anim/embed-steps.ts:embedStates."""
    S, D = config["seq_len"], config["d_model"]
    ids = encode(text, S)
    tr = site_trace(ids, config, w)
    p = max(0, min(S - 1, pos))
    tok, pe, x = vec(tr["tok_emb"][p]), vec(tr["pos_emb"][p]), vec(tr["x0"][p])
    base = {"pos": p, "id": ids[p], "char": ALPHABET[ids[p]], "elem": -1,
            "tok": tok, "pe": pe, "total": S, "row": -1, "rowChar": ""}
    empty = lambda: [None] * D  # noqa: E731
    out = [dict(base, kind=k, sum=empty(), done=[]) for k in ("ids", "lookup", "position")]
    for e in range(D):
        out.append(dict(base, kind="add", elem=e,
                        sum=[x[k] if k <= e else None for k in range(D)],
                        done=[p] if e == D - 1 else []))
    done = [p]
    for q in range(S):
        if q == p:
            continue
        done.append(q)
        out.append(dict(base, kind="row", sum=list(x), done=list(done),
                        row=q, rowChar=ALPHABET[ids[q]]))
    return out


def attention_states(text: str, config: dict, w: dict, head, row: int) -> list[dict]:
    """Port of src/lib/anim/attention-steps.ts:attentionStates (block 0)."""
    S, D, H = config["seq_len"], config["d_model"], config["n_heads"]
    dk = D // H
    tr = site_trace(encode(text, S), config, w)["blocks"][0]["attn"]
    Q, K, V = tr["Q"], tr["K"], tr["V"]
    weights = tr["weights"]  # [H, S, S], F.softmax in float64

    def head_states(h: int, r0: int) -> list[dict]:
        Qh = Q[:, h * dk:(h + 1) * dk]
        Kh = K[:, h * dk:(h + 1) * dk]
        Vh = V[:, h * dk:(h + 1) * dk]
        raw_all = Qh @ Kh.T
        states, rows_done = [], []
        for i in range(r0, S):
            base = {"head": h, "row": i, "dk": dk, "rowsDone": list(rows_done),
                    "concat": None, "projected": None}
            raw = vec(raw_all[i])
            for j in range(S):
                states.append(dict(base, phase="dot", j=j,
                                   dots=[v if k <= j else None for k, v in enumerate(raw)],
                                   scores=None, exps=None, sum=None, weights=None, out=None))
            scaled = [v / math.sqrt(dk) for v in raw]
            states.append(dict(base, phase="scale", j=-1, dots=raw, scores=scaled,
                               exps=None, sum=None, weights=None, out=None))
            masked = [v if k <= i else None for k, v in enumerate(scaled)]
            states.append(dict(base, phase="mask", j=-1, dots=raw, scores=masked,
                               exps=None, sum=None, weights=None, out=None))
            m = max(scaled[: i + 1])
            exps = [None if v is None else math.exp(v - m) for v in masked]
            total = sum(e for e in exps if e is not None)
            wrow = vec(weights[h, i])
            wts = [v if k <= i else None for k, v in enumerate(wrow)]
            soft = dict(base, dots=raw, scores=masked, j=-1, out=None)
            states.append(dict(soft, phase="exp", exps=exps, sum=None, weights=None))
            states.append(dict(soft, phase="sum", exps=exps, sum=total, weights=None))
            states.append(dict(soft, phase="softmax", exps=exps, sum=total, weights=wts))
            for j in range(i + 1):
                partial = (weights[h, i, : j + 1].unsqueeze(1) * Vh[: j + 1]).sum(0)
                states.append(dict(base, phase="wsum", j=j, dots=raw, scores=masked,
                                   exps=exps, sum=total, weights=wts, out=vec(partial)))
            rows_done.append(i)
        return states

    r0 = max(0, min(S - 1, row))
    heads = list(range(H)) if head == "all" else [max(0, min(H - 1, head))]
    out = []
    for h in heads:
        out.extend(head_states(h, r0))
    concat = (tr["head_out"].transpose(0, 1).contiguous().view(S, D)).tolist()
    tail = {"head": -1, "row": S - 1, "dk": dk, "j": -1, "dots": [], "scores": None,
            "exps": None, "sum": None, "weights": None, "out": None, "rowsDone": [],
            "concat": concat}
    out.append(dict(tail, phase="concat", projected=None))
    out.append(dict(tail, phase="project", projected=tr["out"].tolist()))
    return out


def ffn_states(text: str, config: dict, w: dict, pos: int) -> list[dict]:
    """Port of src/lib/anim/ffn-steps.ts:ffnStates (block 0)."""
    S = config["seq_len"]
    b0 = site_trace(encode(text, S), config, w)["blocks"][0]
    p = max(0, min(S - 1, pos))
    x, pre, act, y = (vec(b0[k][p]) for k in ("ln2", "pre", "act", "ffn_out"))
    base = {"pos": p, "neuron": -1, "x": x, "out": None, "fired": None}
    out = [dict(base, phase="input", pre=None, act=None),
           dict(base, phase="expand", pre=pre, act=None)]
    for n in range(len(pre)):
        out.append(dict(base, phase="gelu", neuron=n, pre=pre,
                        act=[v if k <= n else None for k, v in enumerate(act)]))
    out.append(dict(base, phase="contract", pre=pre, act=list(act), out=y))
    out.append(dict(base, phase="fired", pre=pre, act=list(act), out=y,
                    fired=[n for n, u in enumerate(pre) if u > 0]))
    return out


def ts_softmax(x: list[float]) -> list[float]:
    """Port of src/lib/transformer/softmax.ts (no mask), summed in order."""
    m = max(x)
    e = [math.exp(v - m) for v in x]
    s = 0.0
    for v in e:
        s += v
    return [v / s for v in e]


def sample_from_probs(probs: list[float], rng) -> int:
    """Port of sampling.ts:sampleFromProbs."""
    total = 0.0
    for p in probs:
        total += p
    u = rng() * total
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if u < acc:
            return i
    return len(probs) - 1


def overview_states(text: str, config: dict, w: dict, rounds: int) -> list[dict]:
    """Port of src/lib/anim/overview-steps.ts:overviewStates."""
    S = config["seq_len"]
    rng = mulberry32(config["seed"])
    t = text[: S - 1] if text else " "
    context = encode(t, len(t))
    out: list[dict] = []
    rnd = 0
    while rnd < rounds and len(context) < S:
        p = len(context) - 1
        tr = site_trace(context + [0] * (S - len(context)), config, w)
        base = {"round": rnd, "pos": p, "block": -1, "context": list(context),
                "id": context[p], "top": None, "sampled": None, "prob": None}
        out.append(dict(base, kind="token", vec=None, delta=None))
        out.append(dict(base, kind="embed", vec=vec(tr["tok_emb"][p]), delta=None))
        x = tr["x0"][p]
        out.append(dict(base, kind="position", vec=vec(x), delta=vec(tr["pos_emb"][p])))
        for b, bt in enumerate(tr["blocks"]):
            h, y = bt["h"][p], bt["y"][p]
            out.append(dict(base, kind="attn", block=b, vec=vec(h), delta=vec(h - x)))
            out.append(dict(base, kind="ffn", block=b, vec=vec(y), delta=vec(y - h)))
            x = y
        xf = vec(tr["x_final"][p])
        out.append(dict(base, kind="norm", vec=xf, delta=None))
        logits = vec(tr["logits"][p])
        probs = ts_softmax(logits)
        top = sorted(range(len(logits)), key=lambda i: (-probs[i], i))[:5]
        top = [{"id": i, "logit": logits[i], "prob": probs[i]} for i in top]
        out.append(dict(base, kind="logits", vec=xf, delta=None, top=top))
        nxt = sample_from_probs(probs, rng)
        picked = {"top": top, "sampled": nxt, "prob": probs[nxt]}
        out.append(dict(base, kind="sample", vec=xf, delta=None, **picked))
        context = context + [nxt]
        out.append(dict(base, kind="append", context=list(context), vec=xf, delta=None, **picked))
        rnd += 1
    return out


# The animations' default inputs (the widgets' defaults).
ANIM_CONFIG = dict(DEFAULT_CONFIG)  # seq_len 8, d_model 16, 2 heads, d_ff 32, 2 blocks, seed 42
ANIM_TEXT = "hello!"
HERO_TEXT = "hello"
HERO_ROUNDS = 3


def animation_fixture() -> dict:
    config = ANIM_CONFIG
    w = init_weights_site(config)
    S = config["seq_len"]
    tr = site_trace(encode(ANIM_TEXT, S), config, w)
    return {
        "note": "Animation states from scripts/reference.py (float64, site weights: mulberry32 init port).",
        "config": config,
        "weights_sample": {
            "tok_emb_row0": vec(w["tok_emb"][0]),
            "block_1.ffn.W2_last_row": vec(w["block_1.ffn.W2"][-1]),
        },
        "embed": {"text": ANIM_TEXT, "pos": 2, "states": embed_states(ANIM_TEXT, config, w, 2)},
        "attention": [
            {"text": ANIM_TEXT, "head": 0, "row": 3,
             "states": attention_states(ANIM_TEXT, config, w, 0, 3)},
            {"text": ANIM_TEXT, "head": "all", "row": 6,
             "states": attention_states(ANIM_TEXT, config, w, "all", 6)},
        ],
        "ffn": {"text": ANIM_TEXT, "pos": 5, "states": ffn_states(ANIM_TEXT, config, w, 5)},
        "overview": {"text": HERO_TEXT, "rounds": HERO_ROUNDS,
                     "states": overview_states(HERO_TEXT, config, w, HERO_ROUNDS)},
        "logits_hello": vec(tr["logits"][5]),
    }


def write_animation_fixture(out_dir: Path) -> None:
    path = out_dir / "animations.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(animation_fixture(), fh, allow_nan=False, separators=(",", ":"))
        fh.write("\n")
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Animation states for brief 27B (chapters 05-07 and the playground's loop).
# A second file, animations_b.json, so animations.json (27A) is untouched.
# ---------------------------------------------------------------------------
def ln_states(text: str, config: dict, pos: int, gamma: float, beta: float) -> list[dict]:
    """Port of src/lib/anim/ln-steps.ts (residualTrace + lnStates), block 0."""
    S, D, H = config["seq_len"], config["d_model"], config["n_heads"]
    w = init_weights_site(dict(config, n_blocks=1))
    g = torch.full((D,), float(gamma), dtype=torch.float64)
    b = torch.full((D,), float(beta), dtype=torch.float64)
    ids = encode(text, S)
    x0 = w["tok_emb"][torch.tensor(ids, dtype=torch.long)] + positional_encoding_f64(S, D)
    p0 = "block_0"
    ln1 = layernorm(x0, g, b)
    attn = multi_head_attention(
        ln1, w[f"{p0}.attn.W_q"], w[f"{p0}.attn.W_k"], w[f"{p0}.attn.W_v"],
        w[f"{p0}.attn.W_o"], H,
    )["out"]
    h = x0 + attn
    ln2 = layernorm(h, g, b)
    f = ffn(ln2, w[f"{p0}.ffn.W1"], w[f"{p0}.ffn.b1"], w[f"{p0}.ffn.W2"], w[f"{p0}.ffn.b2"])
    p = max(0, min(S - 1, pos))
    out: list[dict] = []
    for sub, x_t, d_t in ((1, x0[p], attn[p]), (2, h[p], f[p])):
        x, delta = vec(x_t), vec(d_t)
        mean = float(x_t.mean())
        variance = float(x_t.var(unbiased=False))
        inv = 1 / math.sqrt(variance + EPS)
        centred = [v - mean for v in x]
        scaled = [(v - mean) * inv for v in x]
        lnv = [(v - mean) * inv * gamma + beta for v in x]
        total = [v + d for v, d in zip(x, delta)]
        base = {"sub": sub, "pos": p, "x": x, "gamma": gamma, "beta": beta,
                "ln": None, "delta": None, "out": None}
        st = {"mean": mean, "variance": variance}
        out.append(dict(base, phase="input", mean=None, variance=None, values=x))
        out.append(dict(base, phase="mean", mean=mean, variance=None, values=x))
        out.append(dict(base, **st, phase="var", values=x))
        out.append(dict(base, **st, phase="centre", values=centred))
        out.append(dict(base, **st, phase="scale", values=scaled))
        out.append(dict(base, **st, phase="affine", values=lnv, ln=lnv))
        out.append(dict(base, **st, phase="sublayer", values=lnv, ln=lnv, delta=delta))
        out.append(dict(base, **st, phase="residual", values=lnv, ln=lnv, delta=delta, out=total))
    return out


def l2(v: list[float]) -> float:
    s = 0.0
    for x in v:
        s += x * x
    return math.sqrt(s)


def top_tokens(logits: list[float], k: int = 5) -> list[dict]:
    """Port of overview-steps.ts:topTokens over ts_softmax."""
    probs = ts_softmax(logits)
    order = sorted(range(len(logits)), key=lambda i: (-probs[i], i))[:k]
    return [{"id": i, "logit": logits[i], "prob": probs[i]} for i in order]


def stack_states(text: str, config: dict, pos: int) -> list[dict]:
    """Port of src/lib/anim/stack-steps.ts:stackStates."""
    S, N = config["seq_len"], config["n_blocks"]
    w = init_weights_site(config)
    tr = site_trace(encode(text, S), config, w)
    p = max(0, min(S - 1, pos))

    def lens(x: torch.Tensor) -> list[dict]:
        xf = layernorm(x, w["ln_final.gamma"], w["ln_final.beta"])
        return top_tokens(vec(xf @ w["tok_emb"].T))

    x = tr["x0"][p]
    rows = [{"layer": 0, "vec": vec(x), "attnNorm": None, "ffnNorm": None, "lens": lens(x)}]
    base = {"pos": p, "nBlocks": N, "top": None}
    out = [dict(base, kind="embed", block=-1, vec=vec(x), delta=None, rows=list(rows))]
    for b, bt in enumerate(tr["blocks"]):
        h, y = bt["h"][p], bt["y"][p]
        dA, dF = vec(h - x), vec(y - h)
        out.append(dict(base, kind="attn", block=b, vec=vec(h), delta=dA, rows=list(rows)))
        rows.append({"layer": b + 1, "vec": vec(y), "attnNorm": l2(dA),
                     "ffnNorm": l2(dF), "lens": lens(y)})
        out.append(dict(base, kind="ffn", block=b, vec=vec(y), delta=dF, rows=list(rows)))
        x = y
    out.append(dict(base, kind="final", block=-1, vec=vec(tr["x_final"][p]), delta=None,
                    rows=list(rows), top=top_tokens(vec(tr["logits"][p]))))
    return out


def top_k_mask(logits: list[float], k: int) -> list[float]:
    """Port of sampling.ts:topKMask."""
    n = len(logits)
    if k <= 0 or k >= n:
        return list(logits)
    threshold = sorted(logits, reverse=True)[k - 1]
    out, kept = [], 0
    for v in logits:
        if v > threshold or (v == threshold and kept < k):
            out.append(v)
            kept += 1
        else:
            out.append(-math.inf)
    return out


def top_p_mask(logits: list[float], p: float) -> list[float]:
    """Port of sampling.ts:topPMask (a stable sort, like Array.prototype.sort)."""
    if p <= 0 or p >= 1:
        return list(logits)
    probs = ts_softmax(logits)
    order = sorted(range(len(logits)), key=lambda i: -probs[i])
    keep, cum = set(), 0.0
    for i in order:
        keep.add(i)
        cum += probs[i]
        if cum >= p:
            break
    return [v if i in keep else -math.inf for i, v in enumerate(logits)]


def ts_softmax_masked(x: list[float]) -> list[float]:
    """softmax.ts on a vector with −∞ entries (they get probability 0)."""
    m = max(x)
    e = [math.exp(v - m) if math.isfinite(v) else 0.0 for v in x]
    s = 0.0
    for v in e:
        s += v
    return [v / s for v in e]


def gen_states(text: str, config: dict, opts: dict) -> list[dict]:
    """Port of src/lib/anim/gen-steps.ts:genStates."""
    S = config["seq_len"]
    w = init_weights_site(config)
    rng = mulberry32(config["seed"])
    t = text[: S - 1] if text else " "
    context = encode(t, len(t))
    out: list[dict] = []
    rnd = 0
    while rnd < opts["rounds"] and len(context) < S:
        L = len(context)
        p = L - 1
        tr = site_trace(context + [0] * (S - L), config, w)
        cache = [{"K": bt["attn"]["K"][:L].tolist(), "V": bt["attn"]["V"][:L].tolist()}
                 for bt in tr["blocks"]]
        fresh = list(range(L)) if rnd == 0 else [L - 1]
        base = {"phase": "forward", "round": rnd, "pos": p, "seqLen": S,
                "context": list(context), "cache": cache, "fresh": fresh,
                "mode": opts["mode"], "tau": opts["temperature"], "k": opts["k"],
                "p": opts["p"], "logits": None, "probs1": None, "probsT": None,
                "kept": None, "final": None, "u": None, "sampled": None}
        out.append(base)
        logits = vec(tr["logits"][p])
        s1 = dict(base, logits=logits)
        out.append(dict(s1, phase="logits"))
        probs1 = ts_softmax(logits)
        out.append(dict(s1, phase="softmax", probs1=probs1))
        tempered = [v / opts["temperature"] for v in logits]
        probs_t = ts_softmax(tempered)
        s2 = dict(s1, probs1=probs1, probsT=probs_t)
        final, kept = probs_t, None
        out.append(dict(s2, phase="temperature",
                        final=final if opts["mode"] == "temperature" else None))
        if opts["mode"] != "temperature":
            masked = (top_k_mask(tempered, opts["k"]) if opts["mode"] == "top-k"
                      else top_p_mask(tempered, opts["p"]))
            final = ts_softmax_masked(masked)
            kept = [i for i, v in enumerate(masked) if math.isfinite(v)]
            out.append(dict(s2, phase="truncate", kept=kept, final=final))
        u = rng()
        sampled = sample_from_probs(final, lambda: u)
        s3 = dict(s2, kept=kept, final=final, u=u, sampled=sampled)
        out.append(dict(s3, phase="draw"))
        context = context + [sampled]
        out.append(dict(s3, phase="append", context=list(context)))
        rnd += 1
    return out


GEN_TEXT = "hello"


def animation_fixture_b() -> dict:
    cfg = ANIM_CONFIG
    return {
        "note": "Brief 27B animation states from scripts/reference.py "
                "(float64, site weights: mulberry32 init port).",
        "config": cfg,
        "layernorm": [
            {"text": ANIM_TEXT, "pos": 5, "gamma": 1, "beta": 0,
             "states": ln_states(ANIM_TEXT, cfg, 5, 1.0, 0.0)},
            {"text": ANIM_TEXT, "pos": 2, "gamma": 2, "beta": 0.5,
             "states": ln_states(ANIM_TEXT, cfg, 2, 2.0, 0.5)},
        ],
        "stacking": [
            {"text": ANIM_TEXT, "pos": 5, "nBlocks": 4,
             "states": stack_states(ANIM_TEXT, dict(cfg, n_blocks=4), 5)},
            {"text": ANIM_TEXT, "pos": 3, "nBlocks": 2,
             "states": stack_states(ANIM_TEXT, cfg, 3)},
        ],
        "generation": [
            {"text": GEN_TEXT, "seed": 42,
             "opts": {"mode": "top-k", "temperature": 0.1, "k": 5, "p": 0.9, "rounds": 3}},
            {"text": GEN_TEXT, "seed": 42,
             "opts": {"mode": "top-p", "temperature": 0.5, "k": 5, "p": 0.9, "rounds": 3}},
            {"text": "the", "seed": 7,
             "opts": {"mode": "temperature", "temperature": 1.0, "k": 5, "p": 0.9, "rounds": 8}},
        ],
    }


def write_animation_fixture_b(out_dir: Path) -> None:
    fx = animation_fixture_b()
    for run in fx["generation"]:
        run["states"] = gen_states(run["text"], dict(ANIM_CONFIG, seed=run["seed"]), run["opts"])
    path = out_dir / "animations_b.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(fx, fh, allow_nan=False, separators=(",", ":"))
        fh.write("\n")
    print(f"wrote {path}")


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "tests" / "unit" / "fixtures"
    config = DEFAULT_CONFIG
    weights = init_weights(config)

    # Per-op fixtures with small fixed inputs ----------------------------------
    g = torch.Generator().manual_seed(123)
    x_small = torch.empty(config["seq_len"], config["d_model"]).normal_(generator=g)

    write_fixture(
        out_dir / "softmax.json",
        {
            "input": to_jsonable(x_small),
            "output_dim_neg1": to_jsonable(F.softmax(x_small, dim=-1)),
        },
    )

    write_fixture(
        out_dir / "layernorm.json",
        {
            "input": to_jsonable(x_small),
            "gamma": to_jsonable(torch.ones(config["d_model"])),
            "beta": to_jsonable(torch.zeros(config["d_model"])),
            "epsilon": EPS,
            "output": to_jsonable(
                layernorm(x_small, torch.ones(config["d_model"]), torch.zeros(config["d_model"]))
            ),
        },
    )

    write_fixture(
        out_dir / "gelu.json",
        {
            "input": to_jsonable(x_small),
            "output": to_jsonable(gelu(x_small)),
        },
    )

    # Matmul: [S,D] @ [D,D]
    A = x_small
    g2 = torch.Generator().manual_seed(7)
    B = torch.empty(config["d_model"], config["d_model"]).normal_(generator=g2)
    write_fixture(
        out_dir / "matmul.json",
        {
            "A": to_jsonable(A),
            "B": to_jsonable(B),
            "output": to_jsonable(A @ B),
        },
    )

    write_fixture(
        out_dir / "causal_mask.json",
        {
            "seq_len": config["seq_len"],
            "output": to_jsonable(causal_mask(config["seq_len"])),
        },
    )

    # Single attention forward
    attn = multi_head_attention(
        x_small,
        weights["block_0.attn.W_q"],
        weights["block_0.attn.W_k"],
        weights["block_0.attn.W_v"],
        weights["block_0.attn.W_o"],
        config["n_heads"],
    )
    write_fixture(
        out_dir / "attention.json",
        {
            "input": to_jsonable(x_small),
            "W_q": to_jsonable(weights["block_0.attn.W_q"]),
            "W_k": to_jsonable(weights["block_0.attn.W_k"]),
            "W_v": to_jsonable(weights["block_0.attn.W_v"]),
            "W_o": to_jsonable(weights["block_0.attn.W_o"]),
            "n_heads": config["n_heads"],
            "Q": to_jsonable(attn["Q"]),
            "K": to_jsonable(attn["K"]),
            "V": to_jsonable(attn["V"]),
            "scores": to_jsonable(attn["scores"]),
            "weights": to_jsonable(attn["weights"]),
            "output": to_jsonable(attn["out"]),
        },
    )

    # Full-model forward fixture
    sample_text = "hello!"
    token_ids = encode(sample_text, config["seq_len"])
    fwd = forward(token_ids, config, weights)
    write_fixture(
        out_dir / "forward.json",
        {
            "config": config,
            "input_text": sample_text,
            "token_ids": token_ids,
            "tok_emb": to_jsonable(fwd["tok_emb"]),
            "pos_emb": to_jsonable(fwd["pos_emb"]),
            "x_after_emb": to_jsonable(fwd["x_after_emb"]),
            "x_final": to_jsonable(fwd["x_final"]),
            "logits": to_jsonable(fwd["logits"]),
            "weights": {k: to_jsonable(v) for k, v in weights.items()},
        },
    )

    # Manifest so the TS side knows what to expect
    write_fixture(
        out_dir / "manifest.json",
        {
            "version": 1,
            "tolerance": 1.0e-5,
            "config": config,
            "alphabet": ALPHABET,
            "files": [
                "softmax.json",
                "layernorm.json",
                "gelu.json",
                "matmul.json",
                "causal_mask.json",
                "attention.json",
                "forward.json",
            ],
        },
    )

    # Animation states (brief 27): a separate file, not in the manifest
    # (verify-maths checks ops; the frame tests read this one).
    write_animation_fixture(out_dir)
    # Brief 27B: chapters 05-07 and the generation loop, in their own file.
    write_animation_fixture_b(out_dir)

    print("\nAll fixtures written.")
    print("Commit tests/unit/fixtures/ to git.")


if __name__ == "__main__":
    main()
