"""
anim_b.py — the state lists behind chapters 6-10's animations (brief 29B),
built from the reference so the TypeScript builders in
src/lib/anim/training/*.ts can be checked against them (frame tests).

  schedule_states    ch 6   lr(t) of four schedules and their measured
                            validation curves (results.json "schedules")
  depth_states       ch 7   residual-stream RMS and gradient norm through
                            8 blocks, pre-LN vs post-LN, at initialisation
  attention_states   ch 8   one head's attention on "alice was" at a few
                            steps of the high-learning-rate run
  rounding_states    ch 9   the model's real gradients rounded to each
                            format, and FP16 with growing loss scales
  memory_states      ch 10  memory in use through one training step
"""

from __future__ import annotations

import math

import numpy as np

import experiments as X
import memory as MEM
import model as M
import optim as O
import precision as P

PART1 = dict(M.PART1_CONFIG)


def tolist(a):
    return np.asarray(a, dtype=np.float64).tolist()


# ---------------------------------------------------------------------------
# Chapter 6
# ---------------------------------------------------------------------------
def schedule_states(sched: dict, total: int = 2000) -> list[dict]:
    """One state per validation point (every 50 steps): each schedule's
    learning rate at that step and its mean validation loss so far."""
    rows = sched["rows"]
    steps = rows[0]["evalSteps"]
    out = []
    for i, t in enumerate(steps):
        lrs = [O.lr_at(min(t, total - 1), dict(r["spec"], total=total)) for r in rows]
        out.append({"i": i, "step": t, "lrs": lrs, "vals": [r["meanEvals"][i] for r in rows]})
    return out


# ---------------------------------------------------------------------------
# Chapter 7
# ---------------------------------------------------------------------------
def depth_states(std: float) -> list[dict]:
    """Forward: block by block, each norm's residual-stream RMS appears;
    backward: block by block from the top, each gradient norm appears."""
    pre = X.depth_scale(std, "pre")
    post = X.depth_scale(std, "post")
    L = X.DEPTH
    out = []
    for k in range(L + 1):
        out.append({"phase": "forward", "layer": k,
                    "rms": [pre["residualRms"][: k + 1], post["residualRms"][: k + 1]],
                    "grad": [[], []]})
    for k in reversed(range(L)):
        out.append({"phase": "backward", "layer": k,
                    "rms": [pre["residualRms"], post["residualRms"]],
                    "grad": [pre["gradNorm"][k:], post["gradNorm"][k:]]})
    return out


# ---------------------------------------------------------------------------
# Chapter 8
# ---------------------------------------------------------------------------
ATTN_TEXT = "alice was"
ATTN_BLOCK = 1
ATTN_HEAD = 0


def snapshot_round(params: dict) -> dict:
    """The snapshots are published with 9 significant digits (a smaller
    file); both languages compute from the published values."""
    return {k: np.array(np.vectorize(lambda v: float(f"{v:.9g}"))(p), dtype=np.float64)
            if p.size else p for k, p in params.items()}


def attention_states(snap: dict) -> list[dict]:
    """At each snapshot: block 1, head 0's attention weights on the window,
    the largest |score| and the mean row entropy over every head and block
    (causal rows from position 1), with the run's recorded validation loss."""
    ids = M.encode_chars(ATTN_TEXT)
    x = np.array([ids[:-1]])
    evals = {e["step"]: e["val"] for e in snap["evals"]}
    out = []
    for t in snap["steps"]:
        p = snap["params"][t]
        _, c = M.forward(p, PART1, x)
        a = c["blocks"][ATTN_BLOCK]["attn"]
        P_ = a[4][0, ATTN_HEAD]
        stats = X.attention_stats(c, PART1)
        out.append({"step": t, "weights": tolist(P_), "maxScore": stats["maxScore"],
                    "entropy": stats["attnEntropy"], "val": evals.get(t)})
    return out


# ---------------------------------------------------------------------------
# Chapter 9
# ---------------------------------------------------------------------------
GRAD_TEXT = "alice was"
HIST_LO, HIST_HI = -48, 16  # log2 bins of |g|
FP16_SCALES = [0, 4, 8, 12, 16, 20, 24]
FORMATS_SHOWN = ["fp32", "bf16", "fp16", "e5m2", "e4m3"]


def flat_grads(params) -> list[float]:
    ids = M.encode_chars(GRAD_TEXT)
    x, y = np.array([ids[:-1]]), np.array([ids[1:]])
    _, g, _ = M.loss_and_grads(params, PART1, x, y)
    out: list[float] = []
    for k in M.param_names(PART1):
        out.extend(float(v) for v in g[k].reshape(-1))
    return out


def rounding_state(g: list[float], fid: str, k: int) -> dict:
    """|g 2^k| per log2 bin; how many nonzero values round to 0, overflow
    (to inf, or NaN for E4M3), and the mean relative error of the rest."""
    scale = P.pow2(k)
    bins = [0] * (HIST_HI - HIST_LO)
    zero = over = 0
    err, n_ok = 0.0, 0
    for v in g:
        if v == 0:
            continue
        s = v * scale
        b = P.exponent_of(abs(s))
        bins[min(max(b, HIST_LO), HIST_HI - 1) - HIST_LO] += 1
        r = P.round_to(s, fid)
        if r == 0:
            zero += 1
        elif not math.isfinite(r):
            over += 1
        else:
            err += abs(r - s) / abs(s)
            n_ok += 1
    info = P.INFO[fid]
    return {"fmt": fid, "scaleExp": k, "bins": bins, "nonzero": sum(bins), "zero": zero,
            "overflow": over, "meanRelErr": err / n_ok if n_ok else 0.0,
            "minSub": info["min_sub"], "minNormal": info["min_normal"], "max": info["max"]}


def rounding_states(params) -> list[dict]:
    g = flat_grads(params)
    out = [rounding_state(g, f, 0) for f in FORMATS_SHOWN]
    out += [rounding_state(g, "fp16", k) for k in FP16_SCALES[1:]]
    return out


# ---------------------------------------------------------------------------
# Chapter 10
# ---------------------------------------------------------------------------
MIXED = dict(weightBytes=2, gradBytes=2, optimizer="adam", master=True)


def memory_states(arch: dict, recompute: str) -> list[dict]:
    return MEM.timeline(arch, MIXED, recompute)


def fixtures(results: dict, snap: dict, trained: dict) -> dict:
    """File name -> payload, for make_fixtures.py."""
    return {
        "anim_schedules.json": {"states": schedule_states(results["schedules"])},
        "anim_depth.json": {"depth": X.DEPTH, "text": X.DEPTH_TEXT,
                            "stds": {str(s): depth_states(s) for s in X.DEPTH_STDS}},
        "anim_attention.json": {"text": ATTN_TEXT, "block": ATTN_BLOCK, "head": ATTN_HEAD,
                                "states": attention_states(snap)},
        "anim_rounding.json": {"text": GRAD_TEXT, "lo": HIST_LO, "hi": HIST_HI,
                               "states": rounding_states(trained)},
        "anim_memory.json": {"recipe": MIXED, "cases": [
            {"arch": a, "recompute": r, "states": memory_states(a, r)}
            for a in (MEM.LLAMA_7B, MEM.PART1_MEM) for r in ("none", "selective", "full")]},
    }
