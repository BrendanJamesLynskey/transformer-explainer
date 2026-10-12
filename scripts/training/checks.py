"""
checks.py — the trainer's correctness checks (run in CI by
`python scripts/training/make_fixtures.py --check`).

1. Initialiser: model.init_params equals scripts/reference.py's port of
   the site initialiser (the weights Part 1's animations were checked on).
2. Gradients: the hand-written backward pass (model.backward) against
   PyTorch autograd on the same forward written with torch ops, in float64,
   for every parameter. Tolerance: max |ours - torch| <= 1e-12 * (1 + max |torch|).
3. Gradients: against central finite differences (h = 1e-6) on a seeded
   sample of 24 entries of every parameter tensor. Tolerance:
   |fd - ours| <= 1e-8 + 1e-6 |ours| (1e-8 covers the central difference's
   round-off, about machine epsilon x loss / h).
4. Optimisers: 25 steps of each optimiser (SGD, momentum, Adam, Adam with
   L2, AdamW), with gradient clipping and a warmup + cosine schedule, from
   the same gradients, against torch.optim (float64).
   Tolerance: max |ours - torch| <= 1e-12 over every parameter.
5. Schedule: the cosine decay against torch's CosineAnnealingLR closed form.

Each check returns a dict of measured errors; make_fixtures.py writes them
to results.md and fails if any exceeds its tolerance.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

import model as M
import optim as O

GRAD_TOL = 1e-12
FD_TOL = 1e-6
FD_ABS = 1e-8
FD_H = 1e-6
OPT_TOL = 1e-12


# ---------------------------------------------------------------------------
# The same forward pass written with torch ops (float64), for autograd.
# ---------------------------------------------------------------------------
def torch_loss(tp: dict, config: dict, ids: np.ndarray, targets: np.ndarray):
    H = config["n_heads"]
    B, S = ids.shape
    D = config["d_model"]
    dk = D // H
    E = tp["tok_emb"]
    pe = torch.tensor(M.positional_encoding(S, D))
    x = E[torch.tensor(ids)] + pe

    def ln(v, g, b):
        mu = v.mean(-1, keepdim=True)
        var = v.var(-1, keepdim=True, unbiased=False)
        return (v - mu) / torch.sqrt(var + M.EPS) * g + b

    mask = torch.triu(torch.ones(S, S, dtype=torch.bool), 1)
    for i in range(config["n_blocks"]):
        p = f"block_{i}"
        a_in = ln(x, tp[f"{p}.ln1.gamma"], tp[f"{p}.ln1.beta"])
        Q = (a_in @ tp[f"{p}.attn.W_q"]).view(B, S, H, dk).transpose(1, 2)
        K = (a_in @ tp[f"{p}.attn.W_k"]).view(B, S, H, dk).transpose(1, 2)
        V = (a_in @ tp[f"{p}.attn.W_v"]).view(B, S, H, dk).transpose(1, 2)
        sc = (Q @ K.transpose(-1, -2)) / math.sqrt(dk)
        P = torch.softmax(sc.masked_fill(mask, float("-inf")), -1)
        C = (P @ V).transpose(1, 2).reshape(B, S, D)
        h = x + C @ tp[f"{p}.attn.W_o"]
        f_in = ln(h, tp[f"{p}.ln2.gamma"], tp[f"{p}.ln2.beta"])
        pre = f_in @ tp[f"{p}.ffn.W1"] + tp[f"{p}.ffn.b1"]
        act = torch.nn.functional.gelu(pre, approximate="tanh")
        x = h + act @ tp[f"{p}.ffn.W2"] + tp[f"{p}.ffn.b2"]
    xf = ln(x, tp["ln_final.gamma"], tp["ln_final.beta"])
    logits = xf @ E.T
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), torch.tensor(targets).reshape(-1)
    )


def perturbed(params: dict, seed: int) -> dict:
    """The site's init with LayerNorm gains/shifts and biases nudged off 1/0,
    so every gradient (including the bias and gain terms) is exercised."""
    rng = np.random.default_rng(seed)
    out = {}
    for k, v in params.items():
        if v.ndim == 1:
            base = 1.0 if k.endswith("gamma") else 0.0
            out[k] = base + 0.1 * rng.standard_normal(v.shape)
        else:
            out[k] = v * 10  # larger weights: attention is no longer uniform
    return out


def check_init(config: dict) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import reference  # scripts/reference.py

    ours = M.init_params(config)
    theirs = reference.init_weights_site(config)
    err = max(float(np.abs(ours[k] - theirs[k].numpy()).max()) for k in ours)
    return {"max_abs_err": err, "ok": err == 0.0}


def check_autograd(params: dict, config: dict, ids, targets) -> dict:
    loss, grads, _ = M.loss_and_grads(params, config, ids, targets)
    tp = {k: torch.tensor(v, requires_grad=True) for k, v in params.items()}
    tl = torch_loss(tp, config, ids, targets)
    tl.backward()
    worst, per = 0.0, {}
    for k in params:
        tg = tp[k].grad.numpy()
        e = float(np.abs(grads[k] - tg).max()) / (1 + float(np.abs(tg).max()))
        per[k] = e
        worst = max(worst, e)
    return {"loss_ours": float(loss), "loss_torch": float(tl.detach()), "loss_err": abs(float(loss) - float(tl.detach())),
            "max_scaled_err": worst, "per_param": per, "ok": worst <= GRAD_TOL}


def check_finite_differences(params: dict, config: dict, ids, targets, per_tensor=24, seed=0) -> dict:
    _, grads, _ = M.loss_and_grads(params, config, ids, targets)
    rng = np.random.default_rng(seed)
    worst, worst_abs, n = 0.0, 0.0, 0
    for k, v in params.items():
        flat = v.reshape(-1)
        picks = rng.choice(flat.size, size=min(per_tensor, flat.size), replace=False)
        for j in picks:
            orig = flat[j]
            flat[j] = orig + FD_H
            lp = M.loss_only(params, config, ids, targets)
            flat[j] = orig - FD_H
            lm = M.loss_only(params, config, ids, targets)
            flat[j] = orig
            fd = (lp - lm) / (2 * FD_H)
            ours = grads[k].reshape(-1)[j]
            err = abs(fd - ours)
            worst_abs = max(worst_abs, err)
            # allowed: FD_ABS (the round-off floor of a central difference,
            # ~ eps * loss / h ~ 1e-9) plus FD_TOL relative
            worst = max(worst, err / (FD_ABS + FD_TOL * abs(ours)))
            n += 1
    return {"entries": n, "max_abs_err": worst_abs, "max_ratio": worst, "ok": worst <= 1.0}


def check_optimisers(params: dict, config: dict, batches: list) -> dict:
    """Step ours and torch.optim on identical gradients (ours), compare the parameters."""
    results = {}
    steps = len(batches)
    sched = dict(kind="cosine", lr=0.01, total=steps, warmup=5)
    for kind, wd in (("sgd", 0.0), ("momentum", 0.0), ("adam", 0.0), ("adam_l2", 0.01), ("adamw", 0.1)):
        ours = {k: v.copy() for k, v in params.items()}
        okind = "adam" if kind == "adam_l2" else kind
        opt = O.Optimizer(okind, ours, weight_decay=wd)
        tp = {k: torch.tensor(v.copy(), requires_grad=False) for k, v in params.items()}
        decay = [t for k, t in tp.items() if params[k].ndim == 2]
        rest = [t for k, t in tp.items() if params[k].ndim != 2]
        groups = [{"params": decay, "weight_decay": wd}, {"params": rest, "weight_decay": 0.0}]
        if okind == "sgd":
            topt = torch.optim.SGD(groups, lr=0.01)
        elif okind == "momentum":
            topt = torch.optim.SGD(groups, lr=0.01, momentum=0.9)
        elif okind == "adam":
            topt = torch.optim.Adam(groups, lr=0.01, foreach=False)
        else:
            topt = torch.optim.AdamW(groups, lr=0.01, foreach=False)
        worst = 0.0
        for t, (x, y) in enumerate(batches):
            # both optimisers step from the same clipped gradients (taken at
            # our parameters), so any difference is the optimiser's arithmetic
            _, g, _ = M.loss_and_grads(ours, config, x, y)
            O.clip_by_global_norm(g, 1.0)
            for k, tensor in tp.items():
                tensor.grad = torch.tensor(g[k].copy())
            lr = O.lr_at(t, sched)
            for grp in topt.param_groups:
                grp["lr"] = lr
            opt.step(ours, g, lr)
            topt.step()
            worst = max(worst, max(float(np.abs(ours[k] - tp[k].numpy()).max()) for k in ours))
        results[kind] = {"steps": steps, "max_abs_err": worst, "ok": worst <= OPT_TOL}
    return results


def check_cosine_schedule(total=200, warmup=0, lr=0.01, min_ratio=0.1) -> dict:
    p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    opt = torch.optim.SGD([p], lr=lr)
    lr_min = lr * min_ratio
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=lr_min)
    worst = 0.0
    for t in range(total):
        # torch's closed form: eta_min + (base - eta_min)(1 + cos(pi t / T_max)) / 2
        theirs = sch._get_closed_form_lr()[0]
        ours = O.lr_at(t, dict(kind="cosine", lr=lr, total=total, warmup=warmup, min_ratio=min_ratio))
        worst = max(worst, abs(ours - theirs))
        opt.step()
        sch.step()
    return {"steps": total, "max_abs_err": worst, "ok": worst <= 1e-15}
