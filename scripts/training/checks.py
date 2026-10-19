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
4. Optimisers: 25 steps of each optimiser (SGD, momentum, Nesterov, Adam,
   Adam with L2, AdamW), with gradient clipping and a warmup + cosine
   schedule, from the same gradients, against torch.optim (float64).
   Tolerance: max |ours - torch| <= 1e-12 over every parameter.
4b. Nesterov on chapter 5's valley (brief 44): ours against
   torch.optim.SGD(momentum, nesterov=True) stepping on torch autograd's
   gradient of the same valley, and against the textbook form that takes
   the gradient at the look-ahead point (whose look-ahead point is the
   parameter PyTorch stores). Tolerance 1e-12.
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
def torch_loss(tp: dict, config: dict, ids: np.ndarray, targets: np.ndarray, with_z: bool = True):
    """The model (and its 29B variants: post-LN, RMSNorm, QK-norm, muP's
    scales, z-loss) written independently with torch ops, for autograd."""
    H = config["n_heads"]
    B, S = ids.shape
    D = config["d_model"]
    dk = D // H
    rms = config.get("norm_kind", "layer") == "rms"
    post = config.get("norm", "pre") == "post"
    E = tp["tok_emb"]
    pe = torch.tensor(M.positional_encoding(S, D))
    x = E[torch.tensor(ids)] + pe

    def ln(v, g, b):
        mu = v.mean(-1, keepdim=True)
        var = v.var(-1, keepdim=True, unbiased=False)
        return (v - mu) / torch.sqrt(var + M.EPS) * g + b

    def norm(v, prefix):
        if rms:
            return v / torch.sqrt((v * v).mean(-1, keepdim=True) + M.EPS) * tp[f"{prefix}.gamma"]
        return ln(v, tp[f"{prefix}.gamma"], tp[f"{prefix}.beta"])

    base = config.get("mup_base")
    scale = 1 / math.sqrt(dk) * math.sqrt((base // H) / dk) if base else 1 / math.sqrt(dk)
    mask = torch.triu(torch.ones(S, S, dtype=torch.bool), 1)

    def attn(a_in, p):
        Q = (a_in @ tp[f"{p}.attn.W_q"]).view(B, S, H, dk).transpose(1, 2)
        K = (a_in @ tp[f"{p}.attn.W_k"]).view(B, S, H, dk).transpose(1, 2)
        V = (a_in @ tp[f"{p}.attn.W_v"]).view(B, S, H, dk).transpose(1, 2)
        if config.get("qk_norm"):
            Q = ln(Q, tp[f"{p}.attn.q_norm.gamma"], tp[f"{p}.attn.q_norm.beta"])
            K = ln(K, tp[f"{p}.attn.k_norm.gamma"], tp[f"{p}.attn.k_norm.beta"])
        sc = (Q @ K.transpose(-1, -2)) * scale
        P = torch.softmax(sc.masked_fill(mask, float("-inf")), -1)
        C = (P @ V).transpose(1, 2).reshape(B, S, D)
        return C @ tp[f"{p}.attn.W_o"]

    def ffn(f_in, p):
        pre = f_in @ tp[f"{p}.ffn.W1"] + tp[f"{p}.ffn.b1"]
        act = torch.nn.functional.gelu(pre, approximate="tanh")
        return act @ tp[f"{p}.ffn.W2"] + tp[f"{p}.ffn.b2"]

    for i in range(config["n_blocks"]):
        p = f"block_{i}"
        if post:
            h = norm(x + attn(x, p), f"{p}.ln1")
            x = norm(h + ffn(h, p), f"{p}.ln2")
        else:
            h = x + attn(norm(x, f"{p}.ln1"), p)
            x = h + ffn(norm(h, f"{p}.ln2"), p)
    xf = norm(x, "ln_final")
    logits = xf @ E.T
    if base:
        logits = logits * (base / D)
    flat = logits.reshape(-1, logits.shape[-1])
    ce = torch.nn.functional.cross_entropy(flat, torch.tensor(targets).reshape(-1))
    if with_z and config.get("z_loss"):
        lse = torch.logsumexp(flat, -1)
        ce = ce + config["z_loss"] * (lse * lse).mean()
    return ce


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
    tl = torch_loss({k: v.detach() for k, v in tp.items()}, config, ids, targets, with_z=False)
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
            lp = M.objective(params, config, ids, targets)
            flat[j] = orig - FD_H
            lm = M.objective(params, config, ids, targets)
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
    for kind, wd in (("sgd", 0.0), ("momentum", 0.0), ("nesterov", 0.0), ("adam", 0.0), ("adam_l2", 0.01),
                     ("adamw", 0.1)):
        ours = {k: v.copy() for k, v in params.items()}
        okind = "adam" if kind == "adam_l2" else kind
        opt = O.Optimizer(okind, ours, weight_decay=wd)
        tp = {k: torch.tensor(v.copy(), requires_grad=False) for k, v in params.items()}
        decay = [t for k, t in tp.items() if params[k].ndim == 2]
        rest = [t for k, t in tp.items() if params[k].ndim != 2]
        groups = [{"params": decay, "weight_decay": wd}, {"params": rest, "weight_decay": 0.0}]
        if okind == "sgd":
            topt = torch.optim.SGD(groups, lr=0.01)
        elif okind in ("momentum", "nesterov"):
            topt = torch.optim.SGD(groups, lr=0.01, momentum=0.9, nesterov=okind == "nesterov")
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


def check_nesterov_valley(valley: dict, lr: float, mu: float) -> dict:
    """Chapter 5's race, Nesterov's line, three ways for `valley["steps"]` steps:
      ours       optim.Optimizer("nesterov"): b <- mu b + g; p <- p - lr (g + mu b)
      torch      torch.optim.SGD(lr, momentum=mu, nesterov=True), stepping on
                 torch autograd's gradient of the valley at its own parameters
      look-ahead the textbook form: the gradient is taken at the point momentum
                 is about to carry theta to, g = grad f(theta - lr mu b); then
                 b <- mu b + g; theta <- theta - lr b. Its look-ahead point
                 theta - lr mu b is what the other two store.
    """
    a, b_, xs, ys = valley["a"], valley["b"], valley["x_star"], valley["y_star"]

    def grad(th):
        return np.array([[a * (th[0] - xs), b_ * (th[1] - ys)]])

    ours = {"theta": np.array([valley["start"]], dtype=np.float64)}
    opt = O.Optimizer("nesterov", ours, momentum=mu)
    tp = torch.nn.Parameter(torch.tensor([valley["start"]], dtype=torch.float64))
    topt = torch.optim.SGD([tp], lr=lr, momentum=mu, nesterov=True)
    theta = np.array(valley["start"], dtype=np.float64)
    buf = np.zeros(2)
    worst_torch = worst_look = 0.0
    for _ in range(valley["steps"]):
        opt.step(ours, {"theta": grad(ours["theta"][0])}, lr)
        topt.zero_grad()
        loss = 0.5 * (a * (tp[0, 0] - xs) ** 2 + b_ * (tp[0, 1] - ys) ** 2)
        loss.backward()
        topt.step()
        g = grad(theta - lr * mu * buf)[0]
        buf = mu * buf + g
        theta = theta - lr * buf
        p = ours["theta"][0]
        worst_torch = max(worst_torch, float(np.abs(p - tp.detach().numpy()[0]).max()))
        worst_look = max(worst_look, float(np.abs(p - (theta - lr * mu * buf)).max()))
    return {"steps": valley["steps"], "lr": lr, "momentum": mu,
            "torch": {"max_abs_err": worst_torch}, "lookahead": {"max_abs_err": worst_look},
            "ok": max(worst_torch, worst_look) <= OPT_TOL}


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


def check_rounding(seed: int = 0, n: int = 20000) -> dict:
    """precision.round_array (vectorised) against precision.round_to (the
    Numerics site's bit-exact scalar code), in every format, with and
    without saturation: random values over 60 binades, and for the FP8
    formats every value, every midpoint between neighbours and overflow.
    Equal means the same double with the same sign (any NaN equals NaN)."""
    import precision as P

    rng = np.random.default_rng(seed)
    out, worst = {}, 0
    for fid in P.FORMAT_ORDER:
        f = P.INFO[fid]
        xs = list(rng.standard_normal(n) * np.exp(rng.uniform(-30, 30, n)))
        if f["bits"] == 8:
            vals = P.positive_values(fid)
            mids = [(a + b) / 2 for a, b in zip(vals, vals[1:])]
            xs += vals + mids + [-v for v in vals + mids]
        xs += [f["max"], f["max"] * 1.0625, f["max"] * 1.5, f["min_sub"] / 2, f["min_sub"] * 0.75,
               0.0, -0.0, math.inf, -math.inf]
        xs = np.array(xs)
        bad = 0
        for sat in (False, True):
            a = P.round_array(xs, fid, sat)
            b = np.array([P.round_to(float(v), fid, sat) for v in xs])
            nan = np.isnan(a) & np.isnan(b)
            same = nan | ((a == b) & (np.signbit(a) == np.signbit(b)))
            bad += int((~same).sum())
        out[fid] = {"values": int(2 * len(xs)), "mismatches": bad}
        worst = max(worst, bad)
    return {"formats": out, "max_mismatches": worst, "ok": worst == 0}
