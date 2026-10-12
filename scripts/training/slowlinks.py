"""
slowlinks.py — chapter 13, "Training over slow links" (brief 29C §2): Part
1's model trained by W simulated data-parallel workers, each on its own
share of every batch, synchronised four ways:

  dense    AdamW on the all-reduced (averaged) gradient: data-parallel
           training as in chapter 11. With 4 workers x 4 windows it is the
           default run (16 windows) up to the order of the additions.
  topk     top-k with error feedback (Deep Gradient Compression, Lin et al.
           2018, arXiv:1712.01887; error feedback / memory: Stich et al.
           2018, arXiv:1809.07599, Karimireddy et al. 2019, arXiv:1901.09847;
           1-bit SGD, Seide et al. 2014, did the same for 1-bit
           quantisation): each worker keeps a momentum u and an accumulator
           v (u <- mu u + g; v <- v + u), sends the largest 1/16 of v per
           tensor (values and indices), and keeps the rest (v and u are
           cleared where sent: "momentum factor masking"); the workers'
           sparse vectors are all-gathered and averaged; SGD steps with it.
  demo     DeMo (Peng et al. 2024, arXiv:2411.19870, Algorithm 1): each
           worker keeps its own momentum M <- beta M + g; each tensor is cut
           into chunks of up to 64 values, each chunk DCT-transformed
           (orthonormal DCT-II), the k largest coefficients per chunk (1/16)
           sent and subtracted from M (the momentum is the error feedback);
           the sent coefficients are all-gathered, averaged per position,
           inverse-transformed, and the step is lr * (sign(M*) + lambda p).
  diloco   DiLoCo (Douillard et al. 2023, arXiv:2311.08105, Algorithm 1):
           each worker runs AdamW alone for H steps; then the workers'
           parameter changes (the "pseudo-gradient") are all-reduced and an
           outer Nesterov momentum step (lr 0.7, momentum 0.9, the paper's
           setting) moves the shared weights.

Bytes are counted, not sent: every value on the wire at BF16 (2 bytes),
indices at the bytes a position needs (2 for a position in a tensor of
up to 65,536 values, 1 in a chunk of up to 256). The arithmetic is float64.
Ring collectives: an all-reduce costs each worker 2 (W-1)/W of the vector;
an all-gather of compressed payloads costs (W-1) payloads (each worker
forwards everyone else's), which is why compression saves less on the wire
than in the payload.

It is a toy: 5,376 parameters, 4 workers, one book, 2,000 steps, CPU. The
Compute in Transit results site (compute-in-transit-results.vercel.app)
measured DeMo and its fixed-band variants on a 15.8M-parameter GPT with 8
workers on TinyStories on GPUs; chapter 13 links it for the real numbers.
"""

from __future__ import annotations

import math
import time

import numpy as np

import data as D
import model as M
import optim as O

PART1 = dict(M.PART1_CONFIG)
WORKERS = 4
WINDOWS_PER_WORKER = 4
KEEP = 16           # keep 1 value (or coefficient) in KEEP
CHUNK = 64          # DeMo chunk length (the largest divisor of a tensor's size up to this)
BETA_DEMO = 0.999   # DeMo's compression decay (the reference code's default)
MU_TOPK = 0.9
OUTER_LR = 0.7      # DiLoCo's outer Nesterov learning rate and momentum
OUTER_MU = 0.9
WD = 0.01
STEPS = 2000
EVAL_EVERY = 50
VALUE_BYTES = 2


def base(**over) -> dict:
    cfg = dict(method="dense", workers=WORKERS, windows=WINDOWS_PER_WORKER, steps=STEPS,
               lr=0.01, warmup=50, min_ratio=0.1, clip=1.0, H=10, data_seed=1)
    cfg.update(over)
    return cfg


# ---------------------------------------------------------------------------
# DCT (orthonormal DCT-II) and chunking
# ---------------------------------------------------------------------------
def dct_matrix(n: int) -> np.ndarray:
    """D (n x n) with coefficients = D @ x and D @ D.T = I."""
    k = np.arange(n, dtype=np.float64)[:, None]
    i = np.arange(n, dtype=np.float64)[None, :]
    d = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * math.sqrt(2.0 / n)
    d[0] = d[0] / math.sqrt(2.0)
    return d


def chunk_len(size: int, target: int = CHUNK) -> int:
    """The largest divisor of `size` not above `target` (DeMo's rule)."""
    best = 1
    for d in range(1, min(size, target) + 1):
        if size % d == 0:
            best = d
    return best


def keep_count(m: int) -> int:
    return max(1, round(m / KEEP))


_DCT: dict[int, np.ndarray] = {}


def dct(n: int) -> np.ndarray:
    if n not in _DCT:
        _DCT[n] = dct_matrix(n)
    return _DCT[n]


def topk_idx(a: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest |a| per row, ties to the lower index
    (a stable sort on -|a|), returned in ascending index order."""
    order = np.argsort(-np.abs(a), axis=-1, kind="stable")[..., :k]
    return np.sort(order, axis=-1)


# ---------------------------------------------------------------------------
# Bytes per worker per step
# ---------------------------------------------------------------------------
def payload_bytes(method: str, shapes: dict) -> dict:
    """What one worker originates per step (payload) and sends in the ring
    collective (wire), and the dense all-reduce's wire for comparison."""
    P = sum(int(np.prod(s)) for s in shapes.values())
    W = WORKERS
    dense_wire = 2 * (W - 1) / W * P * VALUE_BYTES
    if method in ("dense", "diloco"):
        payload = P * VALUE_BYTES
        wire = dense_wire
    elif method == "topk":
        k = sum(keep_count(int(np.prod(s))) for s in shapes.values())
        payload = k * (VALUE_BYTES + 2)
        wire = (W - 1) * payload
    elif method == "demo":
        k = 0
        for s in shapes.values():
            n = int(np.prod(s))
            m = chunk_len(n)
            k += (n // m) * keep_count(m)
        payload = k * (VALUE_BYTES + 1)
        wire = (W - 1) * payload
    else:
        raise ValueError(method)
    return {"params": P, "payload": payload, "wire": wire, "denseWire": dense_wire}


# ---------------------------------------------------------------------------
# The simulated workers
# ---------------------------------------------------------------------------
def worker_grads(params, model, x, y, W, per):
    """Each worker's loss and gradient on its own `per` windows of the batch."""
    out = []
    for w in range(W):
        loss, g, _ = M.loss_and_grads(params, model, x[w * per:(w + 1) * per], y[w * per:(w + 1) * per])
        out.append((loss, g))
    return out


def mean_dict(ds: list[dict]) -> dict:
    """Element-wise mean of dicts of arrays, adding in worker order."""
    out = {}
    for k in ds[0]:
        acc = ds[0][k].copy()
        for d in ds[1:]:
            acc = acc + d[k]
        out[k] = acc / len(ds)
    return out


class DeMo:
    """Per-worker momenta; one step = compress, all-gather, sign update."""

    def __init__(self, params: dict, W: int, beta: float = BETA_DEMO):
        self.W, self.beta = W, beta
        self.M = [{k: np.zeros_like(v) for k, v in params.items()} for _ in range(W)]

    def sync(self, grads: list[dict], record: str | None = None) -> tuple[dict, dict | None]:
        """Returns the aggregated momentum M* per tensor (before the sign)
        and, for tensor `record`, every intermediate (the animation's
        states)."""
        agg = {}
        rec = None
        for name in grads[0]:
            shape = grads[0][name].shape
            n = int(np.prod(shape))
            m = chunk_len(n)
            k = keep_count(m)
            Dm = dct(m)
            total = np.zeros((n // m, m))
            count = np.zeros((n // m, m))
            if record == name:
                rec = {"name": name, "chunk": m, "k": k, "workers": []}
            for w in range(self.W):
                mom = self.beta * self.M[w][name] + grads[w][name]
                chunks = mom.reshape(n // m, m)
                coef = chunks @ Dm.T
                idx = topk_idx(coef, k)
                sent = np.take_along_axis(coef, idx, axis=-1)
                q = np.zeros_like(coef)
                np.put_along_axis(q, idx, sent, axis=-1)
                resid = chunks - q @ Dm
                self.M[w][name] = resid.reshape(shape)
                np.put_along_axis(total, idx, np.take_along_axis(total, idx, -1) + sent, -1)
                np.put_along_axis(count, idx, np.take_along_axis(count, idx, -1) + 1, -1)
                if record == name:
                    rec["workers"].append({"grad": grads[w][name].reshape(-1)[:m].tolist(),
                                           "momentum": chunks[0].tolist(), "coef": coef[0].tolist(),
                                           "idx": idx[0].tolist(), "sent": sent[0].tolist(),
                                           "residual": resid[0].tolist()})
            mean = np.where(count > 0, total / np.maximum(count, 1), 0.0)
            agg[name] = (mean @ Dm).reshape(shape)
            if record == name:
                rec["meanCoef"] = mean[0].tolist()
                rec["aggregate"] = agg[name].reshape(-1)[:m].tolist()
        return agg, rec


class TopK:
    """Deep Gradient Compression's sparse momentum SGD with error feedback."""

    def __init__(self, params: dict, W: int, mu: float = MU_TOPK):
        self.W, self.mu = W, mu
        self.u = [{k: np.zeros_like(v) for k, v in params.items()} for _ in range(W)]
        self.v = [{k: np.zeros_like(v) for k, v in params.items()} for _ in range(W)]

    def sync(self, grads: list[dict]) -> dict:
        agg = {}
        for name in grads[0]:
            shape = grads[0][name].shape
            n = int(np.prod(shape))
            k = keep_count(n)
            total = np.zeros(n)
            for w in range(self.W):
                u = self.mu * self.u[w][name] + grads[w][name]
                v = self.v[w][name] + u
                flat = v.reshape(-1)
                idx = topk_idx(flat[None, :], k)[0]
                total[idx] = total[idx] + flat[idx]
                mask = np.ones(n, dtype=bool)
                mask[idx] = False
                self.v[w][name] = (flat * mask).reshape(shape)
                self.u[w][name] = (u.reshape(-1) * mask).reshape(shape)
            agg[name] = (total / self.W).reshape(shape)
        return agg


def train_workers(cfg: dict, ids: list[int], keep_params: bool = False, eval_every: int = EVAL_EVERY) -> dict:
    """One run of `cfg["method"]` with W simulated workers (see the module
    docstring). Returns the validation curve, the training losses and the
    bytes per step."""
    model = PART1
    S = model["seq_len"]
    W, per = cfg["workers"], cfg["windows"]
    params = M.init_params(model)
    rng = M.mulberry32(cfg["data_seed"])
    train, val = D.split(ids)
    vx, vy = D.val_batch(val, S)
    sched = dict(kind="cosine", lr=cfg["lr"], warmup=cfg["warmup"], min_ratio=cfg["min_ratio"],
                 total=cfg["steps"])
    method = cfg["method"]
    if method == "dense":
        opt = O.Optimizer("adamw", params, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=WD)
    elif method == "diloco":
        local = [{k: v.copy() for k, v in params.items()} for _ in range(W)]
        opts = [O.Optimizer("adamw", local[w], beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=WD)
                for w in range(W)]
        outer_buf = None
    elif method == "demo":
        demo = DeMo(params, W)
    elif method == "topk":
        topk = TopK(params, W)
    else:
        raise ValueError(method)
    evals, losses = [], []
    t0 = time.time()
    diverged = None
    for t in range(cfg["steps"]):
        if t % eval_every == 0:
            evals.append({"step": t, "val": M.loss_only(params, model, vx, vy)})
        x, y = D.sample_batch(train, W * per, S, rng)
        lr = O.lr_at(t, sched)
        if method == "diloco":
            step_loss = 0.0
            for w in range(W):
                loss, g, _ = M.loss_and_grads(local[w], model, x[w * per:(w + 1) * per], y[w * per:(w + 1) * per])
                O.clip_by_global_norm(g, cfg["clip"])
                opts[w].step(local[w], g, lr)
                step_loss += loss / W
            if (t + 1) % cfg["H"] == 0:
                delta = mean_dict([{k: params[k] - local[w][k] for k in params} for w in range(W)])
                if outer_buf is None:
                    outer_buf = {k: v.copy() for k, v in delta.items()}
                else:
                    outer_buf = {k: OUTER_MU * outer_buf[k] + delta[k] for k in delta}
                for k in params:
                    params[k] = params[k] - OUTER_LR * (delta[k] + OUTER_MU * outer_buf[k])
                for w in range(W):
                    local[w] = {k: v.copy() for k, v in params.items()}
            losses.append(float(step_loss))
        else:
            wg = worker_grads(params, model, x, y, W, per)
            step_loss = sum(l for l, _ in wg) / W
            grads = [g for _, g in wg]
            if method == "dense":
                g = mean_dict(grads)
                O.clip_by_global_norm(g, cfg["clip"])
                opt.step(params, g, lr)
            else:
                for g in grads:
                    O.clip_by_global_norm(g, cfg["clip"])
                if method == "demo":
                    agg, _ = demo.sync(grads)
                    for k in params:
                        decay = WD if O.decays(params[k]) else 0.0
                        params[k] = params[k] - lr * (np.sign(agg[k]) + decay * params[k])
                else:
                    agg = topk.sync(grads)
                    for k in params:
                        decay = WD if O.decays(params[k]) else 0.0
                        params[k] = params[k] * (1 - lr * decay) - lr * agg[k]
            losses.append(float(step_loss))
        if not math.isfinite(losses[-1]) or losses[-1] > 12:
            diverged = t
            break
    if diverged is None:
        evals.append({"step": cfg["steps"], "val": M.loss_only(params, model, vx, vy)})
    out = {"evals": evals, "loss": losses[::10], "divergedAt": diverged, "seconds": time.time() - t0}
    if keep_params:
        out["params"] = params
    return out


def bytes_for(method: str, H: int = 1) -> dict:
    shapes = {k: v.shape for k, v in M.init_params(PART1).items()}
    b = payload_bytes(method, shapes)
    if method == "diloco":
        b = dict(b, payload=b["payload"] / H, wire=b["wire"] / H)
    return b


# ---------------------------------------------------------------------------
# The experiment (results.md §13)
# ---------------------------------------------------------------------------
SEEDS = [1, 2, 3]
# learning rates tried per method (each method is tuned: a sign update, a
# plain SGD step and Adam's normalised step want different rates). The
# three rates per method were placed by a one-seed pilot (2026-10-12) so the
# grid brackets the method's best: below them DeMo, top-k and DiLoCo stayed
# on chapter 1's plateau (DeMo 0.001: 2.95; top-k 0.03: 2.95; DiLoCo 0.003:
# 2.91 after 2,000 steps).
SWEEP = {
    "dense": [0.003, 0.01, 0.03],
    "topk": [0.3, 1.0, 3.0],
    "demo": [0.003, 0.01, 0.03],
    "diloco": [0.01, 0.03, 0.1],
}
DILOCO_H = [10, 50]
TARGET = 2.6


def steps_to_target(evals: list[dict], target: float = TARGET) -> int | None:
    hit = [e["step"] for e in evals if e["val"] <= target]
    return hit[0] if hit else None


def summarise(runs: list[dict]) -> dict:
    finals = [r["evals"][-1]["val"] if r["divergedAt"] is None else None for r in runs]
    ok = [f for f in finals if f is not None and math.isfinite(f)]
    hits = [steps_to_target(r["evals"]) if r["divergedAt"] is None else None for r in runs]
    steps = [e["step"] for e in max(runs, key=lambda r: len(r["evals"]))["evals"]]
    mean_curve = []
    for i in range(len(steps)):
        vals = [r["evals"][i]["val"] for r in runs if i < len(r["evals"])]
        mean_curve.append(float(np.mean(vals)) if len(vals) == len(runs) else None)
    return {
        "finals": finals, "mean": float(np.mean(ok)) if ok else None,
        "sd": float(np.std(ok, ddof=1)) if len(ok) > 1 else None,
        "diverged": len(runs) - len(ok),
        "stepsToTarget": hits,
        "meanStepsToTarget": float(np.mean(hits)) if all(h is not None for h in hits) else None,
        "evalSteps": steps, "meanEvals": mean_curve,
    }


def experiment(ids) -> dict:
    """Every method at each of its learning rates, three seeds; DiLoCo at
    H = 10 and 50. The row a method is quoted by is its best mean."""
    rows = []
    for method, lrs in SWEEP.items():
        Hs = DILOCO_H if method == "diloco" else [1]
        for H in Hs:
            per_lr = []
            for lr in lrs:
                runs = [train_workers(base(method=method, lr=lr, H=H, data_seed=s), ids) for s in SEEDS]
                per_lr.append({"lr": lr, **summarise(runs),
                               "losses": [r["loss"] for r in runs]})
            ok = [p for p in per_lr if p["mean"] is not None]
            best = min(ok, key=lambda p: p["mean"])
            name = f"{method} H={H}" if method == "diloco" else method
            b = bytes_for(method, H)
            rows.append({"method": method, "name": name, "H": H, "perLr": [
                {k: v for k, v in p.items() if k not in ("losses", "meanEvals", "evalSteps")}
                for p in per_lr], "bestLr": best["lr"], "mean": best["mean"], "sd": best["sd"],
                "evalSteps": best["evalSteps"], "meanEvals": best["meanEvals"],
                "stepsToTarget": best["stepsToTarget"], "meanStepsToTarget": best["meanStepsToTarget"],
                "bytes": b})
    return {"workers": WORKERS, "windowsPerWorker": WINDOWS_PER_WORKER, "keep": KEEP, "chunk": CHUNK,
            "betaDemo": BETA_DEMO, "muTopk": MU_TOPK, "outerLr": OUTER_LR, "outerMu": OUTER_MU,
            "steps": STEPS, "evalEvery": EVAL_EVERY, "target": TARGET, "seeds": SEEDS, "rows": rows}


# ---------------------------------------------------------------------------
# Chapter 13's animation: two DeMo syncs from the trained model
# ---------------------------------------------------------------------------
ANIM_TENSOR = "block_0.attn.W_q"
ANIM_SEED = 5


def sync_states(trained: dict, ids) -> list[dict]:
    """Two DeMo steps of 4 workers starting from the default run's trained
    weights with empty momenta: each worker's gradient on its own windows,
    its momentum, the DCT of the first chunk of one tensor, the top-k
    coefficients it sends and the residual it keeps, then the average over
    the workers and its inverse DCT. The second step shows the residual
    carried in the momentum (error feedback)."""
    model = PART1
    S = model["seq_len"]
    train, _ = D.split(ids)
    rng = M.mulberry32(ANIM_SEED)
    params = {k: v.copy() for k, v in trained.items()}
    demo = DeMo(params, WORKERS)
    out = []
    for step in range(2):
        x, y = D.sample_batch(train, WORKERS * WINDOWS_PER_WORKER, S, rng)
        grads = [g for _, g in worker_grads(params, model, x, y, WORKERS, WINDOWS_PER_WORKER)]
        _, rec = demo.sync(grads, record=ANIM_TENSOR)
        rec["step"] = step
        out.append(rec)
        # the update itself is not animated; keep the weights fixed so the
        # second step's gradients differ only by the batch
    return out
