"""
isoflop.py — chapter 14's toy IsoFLOP sweep (brief 29 §2a, done in 29C):
Chinchilla's "Approach 2" (Hoffmann et al. 2022, arXiv:2203.15556 §3.2) on
Part 1's model family, on a CPU, in minutes.

For each compute budget C (FLOPs = 6 N D, N parameters, D training tokens)
train models of several widths, each for exactly the tokens the budget
allows (D = C / 6N, so the step count is C / (6 N B S)), with its learning
rate schedule stretched to that length (warmup, then cosine to 10%: a
schedule cut short would favour the smaller models, Chinchilla's point
against Kaplan et al.). Then, per budget, fit a parabola to the final
validation loss against log N; its minimum is the compute-optimal size
N_opt(C). A straight line through log N_opt against log C gives the
exponent a in N_opt ~ C^a (and b = 1 - a for D_opt ~ C^b).

Chinchilla found a ~ 0.49 (Approach 2; 0.46-0.50 over its three
approaches) where Kaplan et al. (2020, arXiv:2001.08361) had 0.73; Llama 3
fitted 0.53 for tokens (§3.2.1: N*(C) = A C^alpha, alpha = 0.53, for the
optimal number of training tokens). The toy is far from those studies: 1,700
to 41,000 parameters (counting the embedding, which dominates at this
size), one 128,000-character book seen many times over at the larger
budgets, one learning rate for every width, three seeds. Chapter 14 says so.
"""

from __future__ import annotations

import math

import numpy as np

import experiments as X
import model as M

PART1 = dict(M.PART1_CONFIG)
WIDTHS = [6, 8, 12, 16, 24, 32, 48, 64]
BUDGETS = [3e9, 1e10, 3e10, 1e11]
MIN_STEPS = 100
MAX_STEPS = 40000
BATCH = 16
LR = 0.01


def config(width: int) -> dict:
    """Part 1's model at another width: 2 heads, FFN twice the width, 2 blocks."""
    return dict(PART1, d_model=width, d_ff=2 * width)


def n_params(width: int) -> int:
    return int(sum(v.size for v in M.init_params(config(width)).values()))


def steps_for(budget: float, width: int) -> int:
    return round(budget / (6 * n_params(width) * BATCH * PART1["seq_len"]))


def fit_parabola(xs: list[float], ys: list[float]) -> dict:
    """Least squares y = c0 + c1 x + c2 x^2 (normal equations, solved by
    Cramer's rule in the order the TypeScript port uses); the minimum at
    x* = -c1 / (2 c2) when c2 > 0."""
    s = [0.0] * 5
    t = [0.0] * 3
    for x, y in zip(xs, ys):
        p = 1.0
        for k in range(5):
            s[k] += p
            if k < 3:
                t[k] += p * y
            p *= x
    a = [[s[0], s[1], s[2]], [s[1], s[2], s[3]], [s[2], s[3], s[4]]]

    def det3(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))

    d = det3(a)
    c = []
    for j in range(3):
        mj = [row[:] for row in a]
        for i in range(3):
            mj[i][j] = t[i]
        c.append(det3(mj) / d)
    xmin = -c[1] / (2 * c[2]) if c[2] > 0 else None
    return {"c": c, "xMin": xmin, "yMin": (c[0] + c[1] * xmin + c[2] * xmin * xmin) if xmin is not None else None}


def fit_line(xs: list[float], ys: list[float]) -> dict:
    """Least squares y = a x + b."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) * (x - mx) for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx
    return {"slope": a, "intercept": my - a * mx}


def fit_points(ok: list[dict]) -> list[dict] | None:
    """The rule (fixed before the sweep ran): the lowest mean and its two
    neighbours in size. A budget whose lowest mean is the smallest or the
    largest size tried has no minimum inside the sizes, and no fit."""
    if len(ok) < 3:
        return None
    i = min(range(len(ok)), key=lambda j: ok[j]["mean"])
    if i == 0 or i == len(ok) - 1:
        return None
    return ok[i - 1:i + 2]


def sweep(ids) -> dict:
    rows = []
    for C in BUDGETS:
        pts = []
        for w in WIDTHS:
            steps = steps_for(C, w)
            if not MIN_STEPS <= steps <= MAX_STEPS:
                continue
            cfg = X.base_cfg(model=config(w), steps=steps,
                             schedule=dict(kind="cosine", lr=LR, warmup=min(50, steps // 10), min_ratio=0.1))
            runs = X.seeds(cfg, ids, eval_every=0)
            summ = X.summarise(runs)
            pts.append({"width": w, "params": n_params(w), "steps": steps,
                        "tokens": steps * BATCH * PART1["seq_len"],
                        "finals": summ["finals"], "mean": summ["mean"], "sd": summ["sd"],
                        "diverged": summ["diverged"]})
        ok = [p for p in pts if p["mean"] is not None]
        near = fit_points(ok)
        if near is None:
            rows.append({"budget": C, "points": pts, "fit": None, "fitWidths": None, "nOpt": None,
                         "dOpt": None, "interior": False})
            continue
        fit = fit_parabola([math.log10(p["params"]) for p in near], [p["mean"] for p in near])
        n_opt = 10 ** fit["xMin"] if fit["xMin"] is not None else None
        rows.append({"budget": C, "points": pts, "fit": fit, "fitWidths": [p["width"] for p in near],
                     "nOpt": n_opt, "dOpt": C / (6 * n_opt) if n_opt else None,
                     "interior": n_opt is not None and near[0]["params"] < n_opt < near[-1]["params"]})
    good = [r for r in rows if r["interior"]]
    line = fit_line([math.log10(r["budget"]) for r in good], [math.log10(r["nOpt"]) for r in good])
    return {"widths": WIDTHS, "budgets": BUDGETS, "batch": BATCH, "lr": LR,
            "seqLen": PART1["seq_len"], "rows": rows, "a": line["slope"], "b": 1 - line["slope"],
            "frontier": line, "published": {"chinchilla2": 0.49, "chinchilla1": 0.50, "chinchilla3": 0.46,
                                            "kaplan": 0.73, "llama3TokensAlpha": 0.53}}
