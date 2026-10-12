"""
optim.py — optimisers, gradient clipping and learning-rate schedules, the
reference for src/lib/training/optimizers.ts and schedules.ts.

Each optimiser follows PyTorch's definition (checks.py compares every step
with torch.optim on the same gradients):

  SGD          p <- p - lr g
  momentum     b <- mu b + g  (b = g on the first step);  p <- p - lr b
  Adam         m <- b1 m + (1 - b1) g;  v <- b2 v + (1 - b2) g^2
               m^ = m / (1 - b1^t);  v^ = v / (1 - b2^t)
               p <- p - lr m^ / (sqrt(v^) + eps)
               (+ L2: g <- g + wd p before the moments, the coupled decay)
  AdamW        p <- p (1 - lr wd), then the Adam step (decoupled decay,
               Loshchilov & Hutter, arXiv:1711.05101)

Weight decay applies to matrices only (the embedding, attention and FFN
weights), not to LayerNorm gains and shifts or biases, a common choice
(e.g. nanoGPT); `decays(name, value)` says which.
"""

from __future__ import annotations

import math

import numpy as np


def decays(value: np.ndarray) -> bool:
    """Weight decay applies to 2-D parameters (matrices) only."""
    return value.ndim == 2


def global_norm(grads: dict) -> float:
    """sqrt(sum of squares of every gradient), in parameter order."""
    total = 0.0
    for g in grads.values():
        total += float((g * g).sum())
    return math.sqrt(total)


def clip_by_global_norm(grads: dict, max_norm: float) -> tuple[float, float]:
    """torch.nn.utils.clip_grad_norm_: scale = min(1, max_norm / (norm + 1e-6))."""
    norm = global_norm(grads)
    scale = min(1.0, max_norm / (norm + 1e-6))
    if scale < 1.0:
        for k in grads:
            grads[k] = grads[k] * scale
    return norm, scale


class Optimizer:
    """One optimiser over a dict of parameters, stepping them in place."""

    def __init__(self, kind: str, params: dict, momentum=0.9, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.0, lr_mult=None):
        assert kind in ("sgd", "momentum", "adam", "adamw")
        self.kind = kind
        self.mu, self.b1, self.b2, self.eps, self.wd = momentum, beta1, beta2, eps, weight_decay
        self.t = 0
        # per-parameter learning-rate factors (muP, model.lr_multipliers); None = all 1
        self.lr_mult = lr_mult
        self.buf = {k: None for k in params}
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict, grads: dict, lr: float) -> None:
        self.t += 1
        base_lr = lr
        for k, p in params.items():
            lr = base_lr * self.lr_mult[k] if self.lr_mult else base_lr
            g = grads[k]
            wd = self.wd if decays(p) else 0.0
            if self.kind == "sgd":
                params[k] = p - lr * g
            elif self.kind == "momentum":
                b = g.copy() if self.buf[k] is None else self.mu * self.buf[k] + g
                self.buf[k] = b
                params[k] = p - lr * b
            else:
                if self.kind == "adamw":
                    p = p * (1 - lr * wd)
                elif wd:
                    g = g + wd * p
                self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
                self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
                mhat = self.m[k] / (1 - self.b1**self.t)
                vhat = self.v[k] / (1 - self.b2**self.t)
                params[k] = p - lr * mhat / (np.sqrt(vhat) + self.eps)


# ---------------------------------------------------------------------------
# Learning-rate schedules: lr(t) for step t = 0 .. total - 1.
# ---------------------------------------------------------------------------
def lr_at(t: int, sched: dict) -> float:
    """constant | cosine (linear warmup, cosine decay) | wsd (warmup, stable, linear decay)."""
    kind, lr, total = sched["kind"], sched["lr"], sched["total"]
    warm = sched.get("warmup", 0)
    lr_min = lr * sched.get("min_ratio", 0.1)
    if kind == "constant":
        return lr
    if t < warm:
        return lr * (t + 1) / warm
    if kind == "cosine":
        progress = (t - warm) / max(1, total - warm)
        return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * progress))
    if kind == "wsd":
        decay_start = total - math.floor(sched.get("decay_frac", 0.2) * total)
        if t < decay_start:
            return lr
        frac = (t - decay_start) / max(1, total - decay_start)
        return lr + (lr_min - lr) * frac
    raise ValueError(kind)
