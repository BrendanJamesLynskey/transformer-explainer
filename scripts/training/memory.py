"""
memory.py — what training a decoder holds in memory (chapter 10), the
reference for src/lib/training/memory.ts.

Bytes on one device, for a GPT-style decoder with L layers of width h,
a heads, FFN width f, vocabulary V, sequence length s and micro-batch b:

  parameters      counted exactly from the architecture (our model's
                  structure, or Llama-style: SwiGLU, RMSNorm, untied head)
  weights         params x bytes of the weight format
  gradients       params x bytes of the gradient format
  optimiser       Adam: m and v (2 x 4 bytes); momentum: 4; SGD: 0;
                  plus an FP32 master copy (4 bytes) when the weights are
                  16-bit: the "16 bytes per parameter" of mixed-precision
                  Adam (Rajbhandari et al. 2020, ZeRO, arXiv:1910.02054 §3.1)
  activations     per layer, Korthikanti et al. 2022 (arXiv:2205.05198)
                  §4.1, eq. 1, for 16-bit activations:
                      s b h (34 + 5 a s / h)  bytes
                  selective recomputation (their §5, the attention core
                  recomputed):  34 s b h;  full recomputation (checkpoint
                  each layer's input only):  2 s b h
  logits          s b V x (2 bytes, plus 4 for the FP32 softmax/loss)

`timeline` walks one training step (forward layer by layer, the loss,
backward layer by layer, the optimiser step) and returns the memory in use
after each event, which chapter 10's animation draws.
"""

from __future__ import annotations

OPT_STATE_BYTES = {"adam": 8, "momentum": 4, "sgd": 0}

# LLaMA 7B (Touvron et al. 2023, arXiv:2302.13971, Table 2: 6.7B parameters;
# FFN 11008 = 2/3 x 4d rounded to a multiple of 256) and Part 1's model.
LLAMA_7B = dict(name="LLaMA 7B", L=32, h=4096, a=32, f=11008, V=32000, s=4096, b=1,
                 ffn="swiglu", norm="rms", tied=False, attnBias=False)
PART1_MEM = dict(name="Part 1's model", L=2, h=16, a=2, f=32, V=64, s=8, b=16,
                 ffn="gelu", norm="layer", tied=True, attnBias=False)


def param_count(c: dict) -> int:
    """Exact parameter count of the architecture."""
    h, f, L, V = c["h"], c["f"], c["L"], c["V"]
    norm = h if c["norm"] == "rms" else 2 * h
    attn = 4 * h * h + (4 * h if c["attnBias"] else 0)
    ffn = 3 * h * f if c["ffn"] == "swiglu" else 2 * h * f + f + h
    per_layer = attn + ffn + 2 * norm
    emb = V * h * (1 if c["tied"] else 2)
    return L * per_layer + emb + norm


def bytes_per_param(recipe: dict) -> dict:
    """Weights, gradients, optimiser states and master copy, bytes per parameter."""
    w = recipe["weightBytes"]
    master = 4 if (recipe.get("master") and w < 4) else 0
    return {"weights": w, "grads": recipe["gradBytes"],
            "optimizer": OPT_STATE_BYTES[recipe["optimizer"]], "master": master}


def activations_per_layer(c: dict, recompute: str) -> float:
    """Korthikanti et al. 2022, eq. 1 and §5: bytes of saved activations per layer."""
    s, b, h, a = c["s"], c["b"], c["h"], c["a"]
    if recompute == "none":
        return s * b * h * (34 + 5 * a * s / h)
    if recompute == "selective":
        return 34 * s * b * h
    if recompute == "full":
        return 2 * s * b * h
    raise ValueError(recompute)


def breakdown(c: dict, recipe: dict, recompute: str = "none") -> dict:
    """Bytes of each kind (gradients, activations, logits at their largest),
    and the peak memory over one step."""
    n = param_count(c)
    bpp = bytes_per_param(recipe)
    act = c["L"] * activations_per_layer(c, recompute)
    logits = c["s"] * c["b"] * c["V"] * 6
    # with recomputation, the backward pass re-creates one layer's full
    # activations at a time on top of the checkpoints
    transient = activations_per_layer(c, "none") if recompute != "none" else 0.0
    out = {
        "params": n,
        "weights": n * bpp["weights"],
        "grads": n * bpp["grads"],
        "optimizer": n * bpp["optimizer"],
        "master": n * bpp["master"],
        "activations": act,
        "logits": logits,
        "recompute": transient,
    }
    out["static"] = out["weights"] + out["grads"] + out["optimizer"] + out["master"]
    # the largest total during the step (timeline): gradients are written
    # during the backward pass, as activations are freed, so the peak is not
    # simply everything at once
    peak = 0.0
    for e in timeline(c, recipe, recompute):
        peak = max(peak, e["total"])
    out["peak"] = peak
    out["bytesPerParam"] = bpp["weights"] + bpp["grads"] + bpp["optimizer"] + bpp["master"]
    return out


def timeline(c: dict, recipe: dict, recompute: str = "none") -> list[dict]:
    """Memory in use after each event of one step. Gradients are allocated
    as the backward pass reaches each layer (a layer's share is params/L,
    the embedding's share with the head); activations are freed once
    their layer's backward is done."""
    n = param_count(c)
    bpp = bytes_per_param(recipe)
    L = c["L"]
    full = activations_per_layer(c, "none")
    kept = activations_per_layer(c, recompute)
    logits = c["s"] * c["b"] * c["V"] * 6
    head = c["V"] * c["h"] * bpp["grads"]  # the (tied) embedding / head
    layer_grad = (n * bpp["grads"] - head) / L
    base = {"weights": n * bpp["weights"], "optimizer": n * bpp["optimizer"],
            "master": n * bpp["master"]}
    acts = [0.0] * L
    grads = 0.0
    out = []

    def snap(event: str, layer: int, extra: float = 0.0, lg: float = 0.0) -> None:
        # plain left-to-right sums (Python's sum() compensates floats), the
        # order memory.ts adds in
        sum_acts = 0.0
        for v in acts:
            sum_acts += v
        total = base["weights"] + base["optimizer"] + base["master"] + grads + sum_acts + extra + lg
        out.append({"event": event, "layer": layer, **base, "grads": grads,
                    "activations": sum_acts, "perLayer": list(acts), "transient": extra,
                    "logits": lg, "total": total})

    snap("start", -1)
    for i in range(L):
        acts[i] = kept
        snap("forward", i)
    snap("loss", L, lg=logits)
    grads += head
    snap("head", L, lg=logits)
    for i in reversed(range(L)):
        extra = full if recompute != "none" else 0.0
        if recompute != "none":
            snap("recompute", i, extra=extra)
        grads += layer_grad
        acts[i] = 0.0
        snap("backward", i)
    snap("step", -1)
    return out


def tiny_saved_floats(caches: dict) -> dict:
    """Floats the NumPy model's forward pass keeps for its backward pass,
    per block and per sequence (arrays counted once, by identity)."""
    import numpy as np

    seen: set[int] = set()

    def walk(v) -> int:
        if isinstance(v, np.ndarray):
            if id(v) in seen:
                return 0
            seen.add(id(v))
            return int(v.size)
        if isinstance(v, (tuple, list)):
            return sum(walk(x) for x in v)
        if isinstance(v, dict):
            return sum(walk(x) for x in v.values())
        return 0

    B = caches["ids"].shape[0]
    blocks = caches["blocks"]
    # weights live in the caches too (W_q ... for the backward pass): count
    # them first so they are not counted as activations
    for b in blocks:
        for w in b["attn"][6:10]:
            seen.add(id(w))
        seen.add(id(b["ffn"][3]))
        seen.add(id(b["ffn"][4]))
        for key in ("ln1", "ln2"):
            seen.add(id(b[key][2]))
    per_block = [walk(b) / B for b in blocks]
    return {"perBlock": per_block, "batch": B}
