"""
model.py — Part 1's decoder in NumPy (float64), with a hand-written backward
pass for every operation. The reference for src/lib/training/*.

This is the model the site's lessons run (src/lib/transformer/*):
  - weights from the site's own initialiser (mulberry32 + Box-Muller,
    N(0, 0.02), src/lib/transformer/init.ts), ported below with the same
    draw order, so Part 2 trains exactly the model Part 1 shows;
  - token embedding + sinusoidal positions (no learned positions);
  - N pre-norm blocks: h = x + Attn(LN1(x)), y = h + FFN(LN2(h));
    attention has no biases, the FFN has b1, b2, GELU is the tanh form;
  - final LayerNorm, then logits = x_f E^T (the head is tied to E);
  - loss: mean next-token cross-entropy over every position.

Every backward function takes the upstream gradient and what the forward
pass saved, and returns the gradients of its inputs and parameters. The
gradient checks (checks.py) compare them with PyTorch autograd and with
finite differences.

Shapes: B = batch, S = sequence length, D = d_model, H = heads,
d_k = D / H, F = d_ff, V = vocabulary size.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# The alphabet: must match src/lib/transformer/tokenizer.ts (and
# scripts/reference.py). '-' appears twice; like the TypeScript Map, the
# later index wins.
# ---------------------------------------------------------------------------
ALPHABET = (
    "abcdefghijklmnopqrstuvwxyz" "0123456789" " .,!?;:'\"()-\n" "+-*/=<>[]{}@#$%"
)
assert len(ALPHABET) == 64
CHAR_TO_ID = {c: i for i, c in enumerate(ALPHABET)}
EPS = 1e-5
U32 = 0xFFFFFFFF

PART1_CONFIG = dict(seq_len=8, d_model=16, n_heads=2, d_ff=32, n_blocks=2, vocab_size=64, seed=42)


def encode_chars(text: str) -> list[int]:
    """Characters to ids (every character must be in the alphabet)."""
    return [CHAR_TO_ID[c] for c in text]


# ---------------------------------------------------------------------------
# The site's random numbers and initialiser (random.ts, init.ts).
# ---------------------------------------------------------------------------
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


def param_names(config: dict) -> list[str]:
    """Parameter names in the order the optimisers and fixtures use."""
    names = ["tok_emb"]
    for i in range(config["n_blocks"]):
        p = f"block_{i}"
        names += [f"{p}.ln1.gamma", f"{p}.ln1.beta"]
        names += [f"{p}.attn.{m}" for m in ("W_q", "W_k", "W_v", "W_o")]
        names += [f"{p}.ln2.gamma", f"{p}.ln2.beta"]
        names += [f"{p}.ffn.W1", f"{p}.ffn.b1", f"{p}.ffn.W2", f"{p}.ffn.b2"]
    names += ["ln_final.gamma", "ln_final.beta"]
    return names


def init_params(config: dict) -> dict[str, np.ndarray]:
    """Port of src/lib/transformer/init.ts: same draws, same order."""
    sample = normal_sampler(mulberry32(config["seed"]))
    d, f, v = config["d_model"], config["d_ff"], config["vocab_size"]

    def normal(rows: int, cols: int) -> np.ndarray:
        return np.array([[sample() * 0.02 for _ in range(cols)] for _ in range(rows)])

    w: dict[str, np.ndarray] = {"tok_emb": normal(v, d)}
    for i in range(config["n_blocks"]):
        p = f"block_{i}"
        w[f"{p}.ln1.gamma"], w[f"{p}.ln1.beta"] = np.ones(d), np.zeros(d)
        for name in ("W_q", "W_k", "W_v", "W_o"):
            w[f"{p}.attn.{name}"] = normal(d, d)
        w[f"{p}.ln2.gamma"], w[f"{p}.ln2.beta"] = np.ones(d), np.zeros(d)
        w[f"{p}.ffn.W1"], w[f"{p}.ffn.b1"] = normal(d, f), np.zeros(f)
        w[f"{p}.ffn.W2"], w[f"{p}.ffn.b2"] = normal(f, d), np.zeros(d)
    w["ln_final.gamma"], w["ln_final.beta"] = np.ones(d), np.zeros(d)
    return {k: w[k] for k in param_names(config)}


def positional_encoding(S: int, D: int) -> np.ndarray:
    """Port of embeddings.ts:positionalEncoding (Vaswani 2017 §3.5)."""
    half = D // 2
    factor = -math.log(10000) / D
    pe = np.zeros((S, D))
    for pos in range(S):
        for i in range(half):
            angle = pos * math.exp(2 * i * factor)
            pe[pos, 2 * i] = math.sin(angle)
            if 2 * i + 1 < D:
                pe[pos, 2 * i + 1] = math.cos(angle)
    return pe


# ---------------------------------------------------------------------------
# Forward and backward, one operation at a time.
# ---------------------------------------------------------------------------
def layernorm_fwd(x, g, b):
    """y = g * (x - mu) / sqrt(var + eps) + b over the last axis (population var)."""
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    r = 1.0 / np.sqrt(var + EPS)
    xhat = (x - mu) * r
    return xhat * g + b, (xhat, r, g)


def layernorm_bwd(dy, cache):
    """dx = r (dxhat - mean(dxhat) - xhat mean(dxhat xhat)); dg = sum dy xhat; db = sum dy."""
    xhat, r, g = cache
    axes = tuple(range(dy.ndim - 1))
    dg = (dy * xhat).sum(axes)
    db = dy.sum(axes)
    dxhat = dy * g
    dx = r * (dxhat - dxhat.mean(-1, keepdims=True) - xhat * (dxhat * xhat).mean(-1, keepdims=True))
    return dx, dg, db


C_GELU = math.sqrt(2 / math.pi)


def gelu_fwd(u):
    """GELU, tanh form: 0.5 u (1 + tanh(c (u + 0.044715 u^3)))."""
    t = np.tanh(C_GELU * (u + 0.044715 * u**3))
    return 0.5 * u * (1 + t), (u, t)


def gelu_bwd(dy, cache):
    """GELU'(u) = 0.5 (1 + t) + 0.5 u (1 - t^2) c (1 + 3 * 0.044715 u^2)."""
    u, t = cache
    d = 0.5 * (1 + t) + 0.5 * u * (1 - t * t) * C_GELU * (1 + 3 * 0.044715 * u * u)
    return dy * d


def softmax_masked(s):
    """Row softmax of s with the causal mask (future positions get weight 0)."""
    S = s.shape[-1]
    mask = np.triu(np.ones((S, S), dtype=bool), 1)
    z = np.where(mask, -np.inf, s)
    m = z.max(-1, keepdims=True)
    e = np.exp(z - m)
    return e / e.sum(-1, keepdims=True)


def attention_fwd(x, Wq, Wk, Wv, Wo, H):
    """Causal multi-head attention, no biases. x: [B,S,D] -> [B,S,D]."""
    B, S, D = x.shape
    dk = D // H
    Q, K, V = x @ Wq, x @ Wk, x @ Wv

    def split(t):  # [B,S,D] -> [B,H,S,dk]
        return t.reshape(B, S, H, dk).transpose(0, 2, 1, 3)

    Qh, Kh, Vh = split(Q), split(K), split(V)
    scale = 1 / math.sqrt(dk)
    scores = (Qh @ Kh.transpose(0, 1, 3, 2)) * scale
    P = softmax_masked(scores)
    Oh = P @ Vh
    C = Oh.transpose(0, 2, 1, 3).reshape(B, S, D)
    out = C @ Wo
    return out, (x, Qh, Kh, Vh, P, C, Wq, Wk, Wv, Wo, scale)


def attention_bwd(dout, cache):
    """Backward of attention_fwd. The softmax step: dS = P * (dP - sum(dP * P))."""
    x, Qh, Kh, Vh, P, C, Wq, Wk, Wv, Wo, scale = cache
    B, S, D = x.shape
    H, dk = Qh.shape[1], Qh.shape[3]
    dWo = np.einsum("bsi,bsj->ij", C, dout)
    dC = dout @ Wo.T
    dOh = dC.reshape(B, S, H, dk).transpose(0, 2, 1, 3)
    dP = dOh @ Vh.transpose(0, 1, 3, 2)
    dVh = P.transpose(0, 1, 3, 2) @ dOh
    dS = P * (dP - (dP * P).sum(-1, keepdims=True))
    dQh = (dS @ Kh) * scale
    dKh = (dS.transpose(0, 1, 3, 2) @ Qh) * scale

    def merge(t):  # [B,H,S,dk] -> [B,S,D]
        return t.transpose(0, 2, 1, 3).reshape(B, S, D)

    dQ, dK, dV = merge(dQh), merge(dKh), merge(dVh)
    dWq = np.einsum("bsi,bsj->ij", x, dQ)
    dWk = np.einsum("bsi,bsj->ij", x, dK)
    dWv = np.einsum("bsi,bsj->ij", x, dV)
    dx = dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T
    return dx, dWq, dWk, dWv, dWo, dS


def ffn_fwd(x, W1, b1, W2, b2):
    pre = x @ W1 + b1
    act, gcache = gelu_fwd(pre)
    return act @ W2 + b2, (x, act, gcache, W1, W2)


def ffn_bwd(dy, cache, rec=None):
    x, act, gcache, W1, W2 = cache
    dW2 = np.einsum("bsi,bsj->ij", act, dy)
    db2 = dy.sum((0, 1))
    dact = dy @ W2.T
    dpre = gelu_bwd(dact, gcache)
    dW1 = np.einsum("bsi,bsj->ij", x, dpre)
    db1 = dpre.sum((0, 1))
    if rec is not None:
        rec(".dact", dact)
        rec(".dpre", dpre)
    return dpre @ W1.T, dW1, db1, dW2, db2


def cross_entropy_fwd(logits, targets):
    """Mean over positions of logsumexp(l) - l[target]. Returns (loss, probs, per-position)."""
    m = logits.max(-1, keepdims=True)
    e = np.exp(logits - m)
    s = e.sum(-1, keepdims=True)
    lse = m[..., 0] + np.log(s[..., 0])
    picked = np.take_along_axis(logits, targets[..., None], -1)[..., 0]
    per_pos = lse - picked
    return per_pos.mean(), e / s, per_pos


def cross_entropy_bwd(probs, targets):
    """d loss / d logits = (p - onehot(target)) / (number of positions)."""
    n = targets.size
    d = probs.copy()
    np.put_along_axis(d, targets[..., None], np.take_along_axis(d, targets[..., None], -1) - 1, -1)
    return d / n


def forward(params, config, ids):
    """Forward pass over a batch of ids [B,S]; returns (logits, caches)."""
    H = config["n_heads"]
    B, S = ids.shape
    E = params["tok_emb"]
    x = E[ids] + positional_encoding(S, config["d_model"])[None]
    caches = {"ids": ids, "blocks": []}
    for i in range(config["n_blocks"]):
        p = f"block_{i}"
        ln1, c_ln1 = layernorm_fwd(x, params[f"{p}.ln1.gamma"], params[f"{p}.ln1.beta"])
        a, c_attn = attention_fwd(
            ln1, params[f"{p}.attn.W_q"], params[f"{p}.attn.W_k"],
            params[f"{p}.attn.W_v"], params[f"{p}.attn.W_o"], H,
        )
        h = x + a
        ln2, c_ln2 = layernorm_fwd(h, params[f"{p}.ln2.gamma"], params[f"{p}.ln2.beta"])
        f, c_ffn = ffn_fwd(ln2, params[f"{p}.ffn.W1"], params[f"{p}.ffn.b1"],
                           params[f"{p}.ffn.W2"], params[f"{p}.ffn.b2"])
        y = h + f
        caches["blocks"].append({"x": x, "ln1": c_ln1, "attn": c_attn, "h": h, "ln2": c_ln2, "ffn": c_ffn})
        x = y
    xf, c_lnf = layernorm_fwd(x, params["ln_final.gamma"], params["ln_final.beta"])
    logits = xf @ E.T
    caches.update(x_last=x, xf=xf, lnf=c_lnf)
    return logits, caches


def backward(params, config, caches, dlogits, record=None):
    """Backward pass from d loss / d logits. Returns a dict of parameter gradients.

    `record`, if a list, receives (name, gradient) for the batch's first
    sequence at each stop on the way back, with the names
    src/lib/training/model-grad.ts uses (the chapter 4 animation).
    """
    g = {k: np.zeros_like(v) for k, v in params.items()}

    def rec(name, value):
        if record is not None:
            record.append((name, value[0]))

    E = params["tok_emb"]
    xf = caches["xf"]
    rec("dlogits", dlogits)
    # logits = xf E^T (tied head)
    g["tok_emb"] += np.einsum("bsv,bsd->vd", dlogits, xf)
    dxf = dlogits @ E
    rec("dxf", dxf)
    dx, g["ln_final.gamma"], g["ln_final.beta"] = layernorm_bwd(dxf, caches["lnf"])
    rec("dx_top", dx)
    for i in reversed(range(config["n_blocks"])):
        p = f"block_{i}"
        c = caches["blocks"][i]
        # y = h + FFN(LN2(h)): the residual passes dx straight through
        dln2, g[f"{p}.ffn.W1"], g[f"{p}.ffn.b1"], g[f"{p}.ffn.W2"], g[f"{p}.ffn.b2"] = ffn_bwd(
            dx, c["ffn"], lambda n, val: rec(p + n, val)
        )
        rec(f"{p}.dln2", dln2)
        dh_branch, g[f"{p}.ln2.gamma"], g[f"{p}.ln2.beta"] = layernorm_bwd(dln2, c["ln2"])
        dh = dx + dh_branch
        rec(f"{p}.dh", dh)
        dln1, g[f"{p}.attn.W_q"], g[f"{p}.attn.W_k"], g[f"{p}.attn.W_v"], g[f"{p}.attn.W_o"], dS = attention_bwd(
            dh, c["attn"]
        )
        rec(f"{p}.dscores_h0", dS[:, 0])
        rec(f"{p}.dln1", dln1)
        dx_branch, g[f"{p}.ln1.gamma"], g[f"{p}.ln1.beta"] = layernorm_bwd(dln1, c["ln1"])
        dx = dh + dx_branch
        rec(f"{p}.dx", dx)
    # x0 = E[ids] + P: scatter-add into the embedding rows
    np.add.at(g["tok_emb"], caches["ids"], dx)
    return g


def loss_and_grads(params, config, inputs, targets, record=None):
    """Mean next-token cross-entropy and its gradient for every parameter."""
    logits, caches = forward(params, config, inputs)
    loss, probs, per_pos = cross_entropy_fwd(logits, targets)
    grads = backward(params, config, caches, cross_entropy_bwd(probs, targets), record)
    return loss, grads, per_pos


def loss_only(params, config, inputs, targets) -> float:
    logits, _ = forward(params, config, inputs)
    return float(cross_entropy_fwd(logits, targets)[0])
