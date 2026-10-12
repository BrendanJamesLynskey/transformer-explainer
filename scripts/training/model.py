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

Variants (brief 29B, chapters 7-9), each off by default so Part 1's model
and every 29A fixture are unchanged; each is gradient-checked against
autograd (checks.py):
  config["norm"] = "post"      post-LN blocks (Vaswani 2017): h = LN1(x +
                               Attn(x)), y = LN2(h + FFN(h)); the final LN
                               stays (the tied head expects it)
  config["norm_kind"] = "rms"  RMSNorm instead of LayerNorm (no mean, no beta)
  config["qk_norm"] = True     LayerNorm on each head's queries and keys
                               (gain and shift per head dimension)
  config["z_loss"] = c         adds c * mean(logsumexp(logits)^2) to the loss
                               (PaLM's z-loss); the reported loss stays the
                               cross-entropy alone
  config["mup_base"] = d0      muP (Yang et al. 2022) relative to width d0:
                               hidden matrices start at std 0.02 / sqrt(m),
                               m = d_model / d0, attention scores are scaled by
                               sqrt(d_k0) / d_k, logits by 1 / m; the
                               optimiser scales hidden matrices' learning
                               rates by 1 / m (lr_multipliers)
A `cast` (precision.py) rounds what a matmul reads and writes, and the
gradients, to emulate low-precision training; None is exact float64.
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


def norm_names(prefix: str, config: dict) -> list[str]:
    """A norm's parameters: gain and shift (LayerNorm) or the gain alone (RMSNorm)."""
    if config.get("norm_kind", "layer") == "rms":
        return [f"{prefix}.gamma"]
    return [f"{prefix}.gamma", f"{prefix}.beta"]


def param_names(config: dict) -> list[str]:
    """Parameter names in the order the optimisers and fixtures use."""
    names = ["tok_emb"]
    for i in range(config["n_blocks"]):
        p = f"block_{i}"
        names += norm_names(f"{p}.ln1", config)
        names += [f"{p}.attn.{m}" for m in ("W_q", "W_k", "W_v", "W_o")]
        if config.get("qk_norm"):
            names += [f"{p}.attn.q_norm.gamma", f"{p}.attn.q_norm.beta",
                      f"{p}.attn.k_norm.gamma", f"{p}.attn.k_norm.beta"]
        names += norm_names(f"{p}.ln2", config)
        names += [f"{p}.ffn.W1", f"{p}.ffn.b1", f"{p}.ffn.W2", f"{p}.ffn.b2"]
    names += norm_names("ln_final", config)
    return names


def width_mult(config: dict) -> float:
    """muP's width multiplier m = d_model / base width (1 without muP)."""
    base = config.get("mup_base")
    return config["d_model"] / base if base else 1.0


def is_hidden(name: str) -> bool:
    """muP's 'hidden' weights: the matrices inside the blocks (not the embedding)."""
    return ".attn.W_" in name or ".ffn.W" in name


def lr_multipliers(config: dict) -> dict[str, float] | None:
    """Per-parameter learning-rate factors: muP scales the hidden matrices'
    Adam learning rate by 1 / m (Yang et al. 2022, Table 3); None otherwise."""
    if not config.get("mup_base"):
        return None
    m = width_mult(config)
    return {k: (1.0 / m if is_hidden(k) else 1.0) for k in param_names(config)}


def init_params(config: dict) -> dict[str, np.ndarray]:
    """Port of src/lib/transformer/init.ts: same draws, same order.
    (muP only rescales the hidden matrices' draws by 1 / sqrt(m).)"""
    sample = normal_sampler(mulberry32(config["seed"]))
    d, f, v = config["d_model"], config["d_ff"], config["vocab_size"]
    dk = d // config["n_heads"]
    hidden_scale = 1.0 / math.sqrt(width_mult(config)) if config.get("mup_base") else None

    def normal(rows: int, cols: int, hidden: bool = False) -> np.ndarray:
        a = np.array([[sample() * 0.02 for _ in range(cols)] for _ in range(rows)])
        return a * hidden_scale if (hidden and hidden_scale is not None) else a

    def norm(prefix: str, n: int) -> None:
        w[f"{prefix}.gamma"] = np.ones(n)
        if config.get("norm_kind", "layer") != "rms":
            w[f"{prefix}.beta"] = np.zeros(n)

    w: dict[str, np.ndarray] = {"tok_emb": normal(v, d)}
    for i in range(config["n_blocks"]):
        p = f"block_{i}"
        norm(f"{p}.ln1", d)
        for name in ("W_q", "W_k", "W_v", "W_o"):
            w[f"{p}.attn.{name}"] = normal(d, d, True)
        if config.get("qk_norm"):
            norm(f"{p}.attn.q_norm", dk)
            norm(f"{p}.attn.k_norm", dk)
            if config.get("norm_kind", "layer") == "rms":
                raise ValueError("qk_norm uses LayerNorm")
        norm(f"{p}.ln2", d)
        w[f"{p}.ffn.W1"], w[f"{p}.ffn.b1"] = normal(d, f, True), np.zeros(f)
        w[f"{p}.ffn.W2"], w[f"{p}.ffn.b2"] = normal(f, d, True), np.zeros(d)
    norm("ln_final", d)
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



def rmsnorm_fwd(x, g):
    """y = g * x / sqrt(mean(x^2) + eps) (Zhang & Sennrich 2019): no mean, no shift."""
    r = 1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + EPS)
    xhat = x * r
    return xhat * g, (xhat, r, g)


def rmsnorm_bwd(dy, cache):
    """dx = r (dxhat - xhat mean(dxhat xhat)); dg = sum dy xhat."""
    xhat, r, g = cache
    axes = tuple(range(dy.ndim - 1))
    dg = (dy * xhat).sum(axes)
    dxhat = dy * g
    dx = r * (dxhat - xhat * (dxhat * xhat).mean(-1, keepdims=True))
    return dx, dg


def norm_fwd(x, params, prefix, config):
    """The block's norm: LayerNorm (default) or RMSNorm."""
    if config.get("norm_kind", "layer") == "rms":
        return rmsnorm_fwd(x, params[f"{prefix}.gamma"])
    return layernorm_fwd(x, params[f"{prefix}.gamma"], params[f"{prefix}.beta"])


def norm_bwd(dy, cache, g, prefix, config):
    """Backward of norm_fwd; writes the gain (and shift) gradients into g."""
    if config.get("norm_kind", "layer") == "rms":
        dx, g[f"{prefix}.gamma"] = rmsnorm_bwd(dy, cache)
    else:
        dx, g[f"{prefix}.gamma"], g[f"{prefix}.beta"] = layernorm_bwd(dy, cache)
    return dx


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



class NoCast:
    """Exact float64: every hook returns its argument unchanged."""

    def w(self, W):  # a weight's compute copy
        return W

    def a(self, x):  # an activation a matrix product reads
        return x

    def o(self, x):  # an activation a matrix product writes
        return x

    def g(self, x):  # an activation gradient a matrix product reads
        return x

    def og(self, x):  # an activation gradient a matrix product writes
        return x

    def pg(self, x):  # a weight (or bias) gradient
        return x


NOCAST = NoCast()


def attention_fwd(x, Wq, Wk, Wv, Wo, H, scale=None, qk=None, cast=NOCAST):
    """Causal multi-head attention, no biases. x: [B,S,D] -> [B,S,D].
    `scale` defaults to 1/sqrt(d_k); `qk` = (gq, bq, gk, bk) applies QK-norm.
    With a cast, each product's inputs go through cast.a and its output
    through cast.o (the caller has already cast x)."""
    B, S, D = x.shape
    dk = D // H
    Q, K, V = cast.o(x @ Wq), cast.o(x @ Wk), cast.o(x @ Wv)

    def split(t):  # [B,S,D] -> [B,H,S,dk]
        return t.reshape(B, S, H, dk).transpose(0, 2, 1, 3)

    Qh, Kh, Vh = split(Q), split(K), split(V)
    qk_cache = None
    if qk is not None:
        Qh, cq = layernorm_fwd(Qh, qk[0], qk[1])
        Kh, ck = layernorm_fwd(Kh, qk[2], qk[3])
        qk_cache = (cq, ck)
    Qh, Kh, Vh = cast.a(Qh), cast.a(Kh), cast.a(Vh)
    if scale is None:
        scale = 1 / math.sqrt(dk)
    scores = cast.o((Qh @ Kh.transpose(0, 1, 3, 2)) * scale)
    P = cast.a(softmax_masked(scores))
    Oh = cast.o(P @ Vh)
    C = cast.a(Oh.transpose(0, 2, 1, 3).reshape(B, S, D))
    out = cast.o(C @ Wo)
    return out, (x, Qh, Kh, Vh, P, C, Wq, Wk, Wv, Wo, scale, scores, qk_cache)


def attention_bwd(dout, cache, cast=NOCAST):
    """Backward of attention_fwd. The softmax step: dS = P * (dP - sum(dP * P)).
    Returns dx, dWq, dWk, dWv, dWo, dS and, with QK-norm, its four gradients."""
    x, Qh, Kh, Vh, P, C, Wq, Wk, Wv, Wo, scale, _, qk_cache = cache
    B, S, D = x.shape
    H, dk = Qh.shape[1], Qh.shape[3]
    dout = cast.g(dout)
    dWo = cast.pg(np.einsum("bsi,bsj->ij", C, dout))
    dC = cast.g(cast.og(dout @ Wo.T))
    dOh = dC.reshape(B, S, H, dk).transpose(0, 2, 1, 3)
    dP = cast.og(dOh @ Vh.transpose(0, 1, 3, 2))
    dVh = cast.og(P.transpose(0, 1, 3, 2) @ dOh)
    dS = cast.g(P * (dP - (dP * P).sum(-1, keepdims=True)))
    dQh = cast.og((dS @ Kh) * scale)
    dKh = cast.og((dS.transpose(0, 1, 3, 2) @ Qh) * scale)
    qk_grads = None
    if qk_cache is not None:
        dQh, dgq, dbq = layernorm_bwd(dQh, qk_cache[0])
        dKh, dgk, dbk = layernorm_bwd(dKh, qk_cache[1])
        qk_grads = (dgq, dbq, dgk, dbk)

    def merge(t):  # [B,H,S,dk] -> [B,S,D]
        return t.transpose(0, 2, 1, 3).reshape(B, S, D)

    dQ, dK, dV = cast.g(merge(dQh)), cast.g(merge(dKh)), cast.g(merge(dVh))
    dWq = cast.pg(np.einsum("bsi,bsj->ij", x, dQ))
    dWk = cast.pg(np.einsum("bsi,bsj->ij", x, dK))
    dWv = cast.pg(np.einsum("bsi,bsj->ij", x, dV))
    dx = cast.og(dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T)
    return dx, dWq, dWk, dWv, dWo, dS, qk_grads


def ffn_fwd(x, W1, b1, W2, b2, cast=NOCAST):
    pre = cast.o(x @ W1 + b1)
    act, gcache = gelu_fwd(pre)
    act = cast.a(act)
    return cast.o(act @ W2 + b2), (x, act, gcache, W1, W2)


def ffn_bwd(dy, cache, rec=None, cast=NOCAST):
    x, act, gcache, W1, W2 = cache
    dy = cast.g(dy)
    dW2 = cast.pg(np.einsum("bsi,bsj->ij", act, dy))
    db2 = cast.pg(dy.sum((0, 1)))
    dact = cast.og(dy @ W2.T)
    dpre = cast.g(gelu_bwd(dact, gcache))
    dW1 = cast.pg(np.einsum("bsi,bsj->ij", x, dpre))
    db1 = cast.pg(dpre.sum((0, 1)))
    if rec is not None:
        rec(".dact", dact)
        rec(".dpre", dpre)
    return cast.og(dpre @ W1.T), dW1, db1, dW2, db2


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


def z_loss_terms(logits, probs, coeff):
    """PaLM's z-loss, coeff * mean(lse^2), and its gradient 2 coeff lse p / n."""
    m = logits.max(-1, keepdims=True)
    lse = m[..., 0] + np.log(np.exp(logits - m).sum(-1))
    n = lse.size
    return coeff * float((lse * lse).mean()), (2 * coeff / n) * lse[..., None] * probs


def head_scale(config: dict) -> float | None:
    """muP's logit multiplier 1 / m (None: no multiplier, Part 1's model)."""
    return 1.0 / width_mult(config) if config.get("mup_base") else None


def attn_scale(config: dict) -> float | None:
    """muP scales scores by sqrt(d_k0) / d_k: written as 1/sqrt(d_k) times
    sqrt(d_k0 / d_k), so at the base width it is exactly Part 1's 1/sqrt(d_k)."""
    base = config.get("mup_base")
    if not base:
        return None
    H = config["n_heads"]
    dk = config["d_model"] // H
    return 1 / math.sqrt(dk) * math.sqrt((base // H) / dk)


def forward(params, config, ids, cast=NOCAST):
    """Forward pass over a batch of ids [B,S]; returns (logits, caches)."""
    H = config["n_heads"]
    post = config.get("norm", "pre") == "post"
    scale = attn_scale(config)
    B, S = ids.shape
    E = params["tok_emb"]
    x = E[ids] + positional_encoding(S, config["d_model"])[None]
    caches = {"ids": ids, "blocks": []}
    for i in range(config["n_blocks"]):
        p = f"block_{i}"
        qk = None
        if config.get("qk_norm"):
            qk = tuple(params[f"{p}.attn.{n}"] for n in
                       ("q_norm.gamma", "q_norm.beta", "k_norm.gamma", "k_norm.beta"))
        W = {n: cast.w(params[f"{p}.attn.{n}"]) for n in ("W_q", "W_k", "W_v", "W_o")}
        F = {n: cast.w(params[f"{p}.ffn.{n}"]) for n in ("W1", "b1", "W2", "b2")}
        if post:
            # h = LN1(x + Attn(x)), y = LN2(h + FFN(h))
            a, c_attn = attention_fwd(cast.a(x), W["W_q"], W["W_k"], W["W_v"], W["W_o"], H,
                                      scale, qk, cast)
            h, c_ln1 = norm_fwd(x + a, params, f"{p}.ln1", config)
            f, c_ffn = ffn_fwd(cast.a(h), F["W1"], F["b1"], F["W2"], F["b2"], cast)
            y, c_ln2 = norm_fwd(h + f, params, f"{p}.ln2", config)
            caches["blocks"].append({"x": x, "ln1": c_ln1, "attn": c_attn, "h": h, "ln2": c_ln2,
                                     "ffn": c_ffn})
            x = y
            continue
        ln1, c_ln1 = norm_fwd(x, params, f"{p}.ln1", config)
        a, c_attn = attention_fwd(
            cast.a(ln1), W["W_q"], W["W_k"], W["W_v"], W["W_o"], H, scale, qk, cast,
        )
        h = x + a
        ln2, c_ln2 = norm_fwd(h, params, f"{p}.ln2", config)
        f, c_ffn = ffn_fwd(cast.a(ln2), F["W1"], F["b1"], F["W2"], F["b2"], cast)
        y = h + f
        caches["blocks"].append({"x": x, "ln1": c_ln1, "attn": c_attn, "h": h, "ln2": c_ln2, "ffn": c_ffn})
        x = y
    xf, c_lnf = norm_fwd(x, params, "ln_final", config)
    xf = cast.a(xf)
    Ec = cast.w(E)
    logits = xf @ Ec.T
    mult = head_scale(config)
    if mult is not None:
        logits = logits * mult
    logits = cast.o(logits)
    caches.update(x_last=x, xf=xf, lnf=c_lnf, E=Ec, mult=mult)
    return logits, caches


def backward(params, config, caches, dlogits, record=None, cast=NOCAST):
    """Backward pass from d loss / d logits. Returns a dict of parameter gradients.

    `record`, if a list, receives (name, gradient) for the batch's first
    sequence at each stop on the way back, with the names
    src/lib/training/model-grad.ts uses (the chapter 4 animation).
    """
    g = {k: np.zeros_like(v) for k, v in params.items()}
    post = config.get("norm", "pre") == "post"

    def rec(name, value):
        if record is not None:
            record.append((name, value[0]))

    E = caches["E"]
    xf = caches["xf"]
    rec("dlogits", dlogits)
    if caches["mult"] is not None:
        dlogits = dlogits * caches["mult"]
    dlogits = cast.g(dlogits)
    # logits = xf E^T (tied head)
    g["tok_emb"] += cast.pg(np.einsum("bsv,bsd->vd", dlogits, xf))
    dxf = cast.og(dlogits @ E)
    rec("dxf", dxf)
    dx = norm_bwd(dxf, caches["lnf"], g, "ln_final", config)
    rec("dx_top", dx)
    for i in reversed(range(config["n_blocks"])):
        p = f"block_{i}"
        c = caches["blocks"][i]
        if post:
            # y = LN2(h + FFN(h)); h = LN1(x + Attn(x))
            dz = norm_bwd(dx, c["ln2"], g, f"{p}.ln2", config)
            dh_branch, g[f"{p}.ffn.W1"], g[f"{p}.ffn.b1"], g[f"{p}.ffn.W2"], g[f"{p}.ffn.b2"] = ffn_bwd(
                dz, c["ffn"], lambda n, val: rec(p + n, val), cast)
            dh = dz + dh_branch
            rec(f"{p}.dh", dh)
            du = norm_bwd(dh, c["ln1"], g, f"{p}.ln1", config)
            dx_branch, g[f"{p}.attn.W_q"], g[f"{p}.attn.W_k"], g[f"{p}.attn.W_v"], g[f"{p}.attn.W_o"], dS, qg = (
                attention_bwd(du, c["attn"], cast))
            if qg is not None:
                (g[f"{p}.attn.q_norm.gamma"], g[f"{p}.attn.q_norm.beta"],
                 g[f"{p}.attn.k_norm.gamma"], g[f"{p}.attn.k_norm.beta"]) = qg
            dx = du + dx_branch
            rec(f"{p}.dx", dx)
            continue
        # y = h + FFN(LN2(h)): the residual passes dx straight through
        dln2, g[f"{p}.ffn.W1"], g[f"{p}.ffn.b1"], g[f"{p}.ffn.W2"], g[f"{p}.ffn.b2"] = ffn_bwd(
            dx, c["ffn"], lambda n, val: rec(p + n, val), cast
        )
        rec(f"{p}.dln2", dln2)
        dh_branch = norm_bwd(dln2, c["ln2"], g, f"{p}.ln2", config)
        dh = dx + dh_branch
        rec(f"{p}.dh", dh)
        dln1, g[f"{p}.attn.W_q"], g[f"{p}.attn.W_k"], g[f"{p}.attn.W_v"], g[f"{p}.attn.W_o"], dS, qg = attention_bwd(
            dh, c["attn"], cast
        )
        if qg is not None:
            (g[f"{p}.attn.q_norm.gamma"], g[f"{p}.attn.q_norm.beta"],
             g[f"{p}.attn.k_norm.gamma"], g[f"{p}.attn.k_norm.beta"]) = qg
        rec(f"{p}.dscores_h0", dS[:, 0])
        rec(f"{p}.dln1", dln1)
        dx_branch = norm_bwd(dln1, c["ln1"], g, f"{p}.ln1", config)
        dx = dh + dx_branch
        rec(f"{p}.dx", dx)
    # x0 = E[ids] + P: scatter-add into the embedding rows
    np.add.at(g["tok_emb"], caches["ids"], dx)
    return g


def loss_and_grads(params, config, inputs, targets, record=None, cast=NOCAST, loss_scale=None):
    """Mean next-token cross-entropy and its gradient for every parameter.
    With config["z_loss"], the gradient includes the z-loss term (the
    returned loss is still the cross-entropy). `loss_scale` multiplies
    d loss / d logits before the backward pass (mixed precision); the
    gradients come back scaled."""
    logits, caches = forward(params, config, inputs, cast)
    loss, probs, per_pos = cross_entropy_fwd(logits, targets)
    dlogits = cross_entropy_bwd(probs, targets)
    if config.get("z_loss"):
        dlogits = dlogits + z_loss_terms(logits, probs, config["z_loss"])[1]
    if loss_scale is not None:
        dlogits = dlogits * loss_scale
    grads = backward(params, config, caches, dlogits, record, cast)
    return loss, grads, per_pos


def loss_only(params, config, inputs, targets, cast=NOCAST) -> float:
    logits, _ = forward(params, config, inputs, cast)
    return float(cross_entropy_fwd(logits, targets)[0])


def objective(params, config, inputs, targets) -> float:
    """What the optimiser minimises: the cross-entropy plus, with
    config["z_loss"], the z-loss term (for the finite-difference check)."""
    logits, _ = forward(params, config, inputs)
    ce, probs, _ = cross_entropy_fwd(logits, targets)
    if config.get("z_loss"):
        ce = ce + z_loss_terms(logits, probs, config["z_loss"])[0]
    return float(ce)
