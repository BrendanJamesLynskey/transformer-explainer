"""
experiments.py — the training loop with every knob chapters 6-10 turn
(brief 29B), and the measured experiments those chapters quote.

`train_ext` is make_fixtures.train_run (the site's trainer, step for step)
plus: model variants (model.py: post-LN, RMSNorm, QK-norm, z-loss, muP),
per-parameter learning rates (muP), emulated low precision (precision.py:
compute copies, activations and gradients rounded; FP32 master weights or
none; dynamic loss scaling), quantised gradients, divergence detection and
attention statistics. With none of them it is train_run exactly (checked:
test `same_as_train_run` in make_fixtures.build).

Every experiment runs three data seeds (1, 2, 3) per setting unless it says
otherwise, and its results are reported as mean +- sd over the seeds: at the
higher learning rates training is chaotic (29A measured single runs moving
by ~0.1 between CPUs), so a single run proves little.
"""

from __future__ import annotations

import math
import time

import numpy as np

import data as D
import model as M
import optim as O
import precision as P

PART1 = dict(M.PART1_CONFIG)
SEEDS = [1, 2, 3]
DIVERGED = 6.0  # a validation loss above this (ln 64 = 4.16) counts as diverged


def base_cfg(**over) -> dict:
    """The default run's settings (make_fixtures.DEFAULT_TRAIN), overridable."""
    cfg = dict(
        model=PART1, batch=16, steps=2000,
        optimizer=dict(kind="adamw", beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.01),
        schedule=dict(kind="cosine", lr=0.01, warmup=50, min_ratio=0.1),
        clip=1.0, data_seed=1,
    )
    cfg.update(over)
    return cfg


def attention_stats(caches, config) -> dict:
    """Largest |attention score| (before softmax) and the mean entropy of the
    attention rows, over every block, head and position (causal rows only)."""
    mx, ent, n = 0.0, 0.0, 0
    for b in caches["blocks"]:
        scores, Pm = b["attn"][11], b["attn"][4]
        S = scores.shape[-1]
        tril = np.tril(np.ones((S, S), dtype=bool))
        mx = max(mx, float(np.abs(scores[..., tril]).max()))
        with np.errstate(divide="ignore", invalid="ignore"):
            h = -np.where(Pm > 0, Pm * np.log(Pm), 0.0).sum(-1)
        ent += float(h[..., 1:].sum())  # position 0 always has entropy 0
        n += h[..., 1:].size
    return {"maxScore": mx, "attnEntropy": ent / n}


def train_ext(cfg: dict, ids: list[int], eval_every: int = 100, log_every: int = 10,
              stats_every: int = 0, keep_params: bool = False, snapshots=(),
              stop_at: float | None = None) -> dict:
    """One training run. Extra cfg keys (all optional):
      precision = dict(fwd="bf16", bwd="bf16", scaled=False, master=True,
                       loss_scale=None | "dynamic" | float)
      grad_bits = k       quantise each weight gradient to k-bit integers
      stop_on_diverge     stop when the training loss is not finite or > 2 * DIVERGED
    """
    model = cfg["model"]
    S = model["seq_len"]
    params = M.init_params(model)
    o = cfg["optimizer"]
    opt = O.Optimizer("adam" if o["kind"] == "adam_l2" else o["kind"], params,
                      momentum=o.get("momentum", 0.9), beta1=o.get("beta1", 0.9),
                      beta2=o.get("beta2", 0.999), eps=o.get("eps", 1e-8),
                      weight_decay=o.get("weight_decay", 0.0), lr_mult=M.lr_multipliers(model))
    rng = M.mulberry32(cfg["data_seed"])
    train, val = D.split(ids)
    vx, vy = D.val_batch(val, S)
    sched = dict(cfg["schedule"], total=cfg["steps"])
    prec = cfg.get("precision")
    cast = (P.Cast(prec["fwd"], prec["bwd"], prec.get("scaled", False),
                   prec.get("out", "same"), prec.get("out_bwd", "same")) if prec else M.NOCAST)
    store = prec["fwd"] if (prec and not prec.get("master", True)) else None
    if store:
        for k in params:
            params[k] = P.round_array(params[k], store)
    ls = prec.get("loss_scale") if prec else None
    scale = 65536.0 if ls == "dynamic" else ls
    good_steps, skipped = 0, 0
    upd_total, upd_lost = 0, 0
    steps, losses, norms, lrs, evals, stats, snaps = [], [], [], [], [], [], {}
    diverged_at = None
    t0 = time.time()
    for t in range(cfg["steps"]):
        if t in snapshots:
            snaps[t] = {k: v.copy() for k, v in params.items()}
        if eval_every and t % eval_every == 0:
            evals.append({"step": t, "val": M.loss_only(params, model, vx, vy, cast)})
            if stop_at is not None and evals[-1]["val"] <= stop_at:
                break
        if stats_every and t % stats_every == 0:
            _, caches = M.forward(params, model, vx, cast)
            stats.append(dict(step=t, **attention_stats(caches, model)))
        x, y = D.sample_batch(train, cfg["batch"], S, rng)
        loss, grads, _ = M.loss_and_grads(params, model, x, y, cast=cast, loss_scale=scale)
        if scale is not None:
            finite = all(np.isfinite(g).all() for g in grads.values())
            if not finite:
                # overflow: skip the step, halve the scale (torch.cuda.amp.GradScaler)
                skipped += 1
                scale = scale * 0.5
                good_steps = 0
                lr = O.lr_at(t, sched)
                if t % log_every == 0 or t == cfg["steps"] - 1:
                    steps.append(t), losses.append(float(loss)), norms.append(float("nan")), lrs.append(lr)
                continue
            for k in grads:
                grads[k] = grads[k] / scale
            if ls == "dynamic":
                good_steps += 1
                if good_steps == 2000:
                    scale, good_steps = scale * 2.0, 0
        if cfg.get("grad_bits"):
            for k in grads:
                grads[k] = P.quantise_int(grads[k], cfg["grad_bits"])
        norm = O.clip_by_global_norm(grads, cfg["clip"])[0] if cfg["clip"] else O.global_norm(grads)
        lr = O.lr_at(t, sched)
        if store:
            before = {k: v.copy() for k, v in params.items()}
        opt.step(params, grads, lr)
        if store:
            # the master copy is gone: the weights themselves are rounded,
            # and an update smaller than half the format's spacing is lost
            for k in params:
                moved = params[k] != before[k]
                params[k] = P.round_array(params[k], store)
                upd_total += int(moved.sum())
                upd_lost += int((moved & (params[k] == before[k])).sum())
        if t % log_every == 0 or t == cfg["steps"] - 1:
            steps.append(t)
            losses.append(float(loss))
            norms.append(norm)
            lrs.append(lr)
        if not math.isfinite(loss) or loss > 2 * DIVERGED:
            diverged_at = t
            if cfg.get("stop_on_diverge", True):
                break
    if diverged_at is None:
        evals.append({"step": cfg["steps"], "val": M.loss_only(params, model, vx, vy, cast)})
    out = {"steps": steps, "loss": losses, "gradNorm": norms, "lr": lrs, "evals": evals,
           "divergedAt": diverged_at, "seconds": time.time() - t0}
    if stats_every:
        out["stats"] = stats
    if prec:
        c = cast.counts
        out["precision"] = {"skipped": skipped, "finalScale": scale,
                            "updatesLost": upd_lost / upd_total if upd_total else None,
                            "gradFlushedToZero": c["g_zero"] / max(1, c["g_total"]),
                            "gradNonFinite": c["g_inf"] / max(1, c["g_total"])}
    if keep_params:
        out["params"] = params
    if snapshots:
        out["snapshots"] = snaps
    return out


def final_val(run: dict) -> float:
    """The final validation loss; a diverged run counts as +inf."""
    if run["divergedAt"] is not None:
        return math.inf
    v = run["evals"][-1]["val"]
    return v if math.isfinite(v) else math.inf


def summarise(runs: list[dict]) -> dict:
    """Mean and sd of the final validation loss over seeds (diverged runs
    excluded from the mean and counted)."""
    finals = [final_val(r) for r in runs]
    ok = [f for f in finals if math.isfinite(f) and f < DIVERGED]
    return {
        "finals": [f if math.isfinite(f) else None for f in finals],
        "diverged": len(finals) - len(ok),
        "mean": float(np.mean(ok)) if ok else None,
        "sd": float(np.std(ok, ddof=1)) if len(ok) > 1 else None,
    }


def seeds(cfg: dict, ids, seeds_=SEEDS, **kw) -> list[dict]:
    return [train_ext(dict(cfg, data_seed=s), ids, **kw) for s in seeds_]


def mean_curve(runs: list[dict], key: str = "evals") -> tuple[list[int], list[float | None]]:
    """Steps and the mean validation loss over the runs that reached each step."""
    steps = [e["step"] for e in max(runs, key=lambda r: len(r[key]))[key]]
    out = []
    for i in range(len(steps)):
        vals = [r[key][i]["val"] for r in runs if i < len(r[key]) and math.isfinite(r[key][i]["val"])]
        out.append(float(np.mean(vals)) if len(vals) == len(runs) else None)
    return steps, out


def slim(run: dict) -> dict:
    """What results.json keeps of a run: its validation curve, when it
    diverged, and (for the precision runs) the precision counters."""
    out = {"evals": run["evals"], "divergedAt": run["divergedAt"]}
    if "precision" in run:
        out["precision"] = run["precision"]
    return out


# ---------------------------------------------------------------------------
# Chapter 6: learning rate, schedules, batch size
# ---------------------------------------------------------------------------
LR_SWEEP = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def lr_sweep(ids) -> dict:
    """AdamW at seven peak learning rates, the default run otherwise
    (warmup 50 + cosine, 2,000 steps, clip 1.0), three seeds each."""
    rows = []
    for lr in LR_SWEEP:
        runs = seeds(base_cfg(schedule=dict(kind="cosine", lr=lr, warmup=50, min_ratio=0.1)), ids)
        rows.append({"lr": lr, "runs": [slim(r) for r in runs], **summarise(runs)})
    best = min((r for r in rows if r["mean"] is not None), key=lambda r: r["mean"])
    return {"rows": rows, "bestLr": best["lr"], "steps": 2000}


SCHEDULE_RUNS = [
    ("constant", dict(kind="constant", lr=0.01)),
    ("warmup + constant", dict(kind="wsd", lr=0.01, warmup=50, min_ratio=1.0, decay_frac=0.0)),
    ("warmup + cosine", dict(kind="cosine", lr=0.01, warmup=50, min_ratio=0.1)),
    ("WSD", dict(kind="wsd", lr=0.01, warmup=50, min_ratio=0.1, decay_frac=0.2)),
]


def schedules(ids) -> dict:
    """Four schedules at peak learning rate 0.01, 2,000 steps, three seeds;
    validation every 50 steps so the end of WSD's decay is visible."""
    rows = []
    for name, spec in SCHEDULE_RUNS:
        runs = seeds(base_cfg(schedule=spec), ids, eval_every=50)
        steps_, mean = mean_curve(runs)
        rows.append({"name": name, "spec": spec, "runs": [slim(r) for r in runs],
                     "evalSteps": steps_, "meanEvals": mean, **summarise(runs)})
    return {"rows": rows}


BATCHES = [4, 8, 16, 32, 64]
BATCH_LRS = [0.01, 0.03]
TARGET = 2.6


def steps_to(cfg: dict, ids, target: float, max_steps: int, every: int = 25) -> int | None:
    """Steps until the validation loss first reaches `target` (checked every
    `every` steps), or None within max_steps. Constant learning rate after a
    50-step warmup, so stopping early does not change the schedule."""
    run = train_ext(dict(cfg, steps=max_steps), ids, eval_every=every, stop_at=target)
    hit = [e["step"] for e in run["evals"] if e["val"] <= target]
    return hit[0] if hit else None


def batch_size(ids) -> dict:
    """Steps and tokens to reach validation loss TARGET at each batch size
    (each at the better of two learning rates), three seeds; then the
    McCandlish et al. (2018) fit S/S_min - 1 = (E/E_min - 1)^-1 (eq. 2.11), whose
    B_crit = E_min / S_min is the batch beyond which doubling the batch no
    longer halves the steps."""
    rows = []
    for B in BATCHES:
        per_lr = []
        for lr in BATCH_LRS:
            cfg = base_cfg(batch=B, schedule=dict(kind="wsd", lr=lr, warmup=50, min_ratio=1.0, decay_frac=0.0))
            cap = max(1500, (2000 * 16 * 3) // B)
            hits = [steps_to(dict(cfg, data_seed=s), ids, TARGET, cap) for s in SEEDS]
            ok = [h for h in hits if h is not None]
            per_lr.append({"lr": lr, "steps": hits,
                           "mean": float(np.mean(ok)) if len(ok) == len(hits) else None,
                           "sd": float(np.std(ok, ddof=1)) if len(ok) == len(hits) else None})
        done = [p for p in per_lr if p["mean"] is not None]
        best = min(done, key=lambda p: p["mean"]) if done else None
        rows.append({"batch": B, "perLr": per_lr, "best": best,
                     "steps": best["mean"] if best else None,
                     "windows": best["mean"] * B if best else None})
    fit = mccandlish_fit([(r["steps"], r["windows"]) for r in rows if r["steps"] is not None])
    return {"target": TARGET, "rows": rows, "fit": fit, "seqLen": PART1["seq_len"]}


def mccandlish_fit(points: list[tuple[float, float]]) -> dict:
    """Least squares (in log space) of S = S_min (1 + B_crit / B) over the
    measured (S, E = S B) points: the hyperbola (S/S_min - 1)(E/E_min - 1) = 1
    written per batch size. Grid search over B_crit, closed form for S_min."""
    best = None
    for k in range(0, 801):
        bc = 10 ** (-1 + k * 4 / 800)  # 0.1 .. 1000 windows
        # S = S_min (1 + bc / B): log S_min = mean(log S - log(1 + bc / B))
        logs = [math.log(s) - math.log(1 + bc / (e / s)) for s, e in points]
        lsm = float(np.mean(logs))
        err = 0.0
        for s, e in points:
            err += (math.log(s) - lsm - math.log(1 + bc / (e / s))) ** 2
        if best is None or err < best[0]:
            best = (err, bc, math.exp(lsm))
    err, bc, smin = best
    return {"bCrit": bc, "sMin": smin, "eMin": smin * bc, "rmsLogErr": math.sqrt(err / len(points))}


def noise_scale(ids, at=(0, 250, 500, 1000, 1500, 2000), n: int = 256) -> dict:
    """The simple gradient noise scale B_simple = tr(Sigma) / |G|^2
    (McCandlish et al. 2018, arXiv:1812.06162, eq. 2.9), measured along the
    default run from n single-window gradients g_i on held-out training
    windows: G ~ mean g_i, tr(Sigma) ~ sum_i |g_i - G|^2 / (n - 1), and
    |G|^2 ~ |mean g|^2 - tr(Sigma) / n (unbiased)."""
    model = PART1
    S = model["seq_len"]
    train, _ = D.split(ids)
    rng = M.mulberry32(99)
    xs, ys = D.sample_batch(train, n, S, rng)
    cfg = base_cfg()
    last = cfg["steps"]
    run = train_ext(cfg, ids, snapshots=set(at) - {last}, keep_params=True)
    snaps = dict(run["snapshots"])
    snaps[last] = run["params"]
    out = []
    for t in sorted(at):
        p = snaps[t]
        flat = []
        for i in range(n):
            _, g, _ = M.loss_and_grads(p, model, xs[i:i + 1], ys[i:i + 1])
            flat.append(np.concatenate([g[k].reshape(-1) for k in M.param_names(model)]))
        Gm = np.stack(flat)
        mean = Gm.mean(0)
        tr = float(((Gm - mean) ** 2).sum() / (n - 1))
        g2 = float((mean * mean).sum()) - tr / n
        out.append({"step": t, "trSigma": tr, "gNormSq": g2, "bSimple": tr / g2 if g2 > 0 else None})
    return {"windows": n, "points": out}


# ---------------------------------------------------------------------------
# Chapter 7: initialisation, normalisation, muP
# ---------------------------------------------------------------------------
DEPTH = 8
DEPTH_STDS = [0.02, 0.25]
DEPTH_TEXT = "alice was"


def depth_scale(std: float, norm: str) -> dict:
    """At initialisation, for an 8-block model with matrices drawn at `std`
    (0.02, the site's; 0.25 = 1/sqrt(16), Xavier-like): the RMS of the
    residual stream entering each block (and leaving the last), and the
    norm of each block's weight-matrix gradients, on the window "alice was".
    Chapter 7's animation draws these (anim_b.depth_states)."""
    cfg = dict(PART1, n_blocks=DEPTH, norm=norm)
    p = M.init_params(cfg)
    for k in p:
        if p[k].ndim == 2:
            p[k] = p[k] * (std / 0.02)
    ids = M.encode_chars(DEPTH_TEXT)
    x, y = np.array([ids[:-1]]), np.array([ids[1:]])
    logits, c = M.forward(p, cfg, x)
    _, g, _ = M.loss_and_grads(p, cfg, x, y)
    rms = [float(np.sqrt((b["x"] ** 2).mean())) for b in c["blocks"]]
    rms.append(float(np.sqrt((c["x_last"] ** 2).mean())))
    gn = []
    for i in range(DEPTH):
        s = 0.0
        for k in M.param_names(cfg):
            if k.startswith(f"block_{i}.") and M.is_hidden(k):
                s += float((g[k] * g[k]).sum())
        gn.append(math.sqrt(s))
    return {"std": std, "norm": norm, "residualRms": rms, "gradNorm": gn}


PRE_POST = dict(depth=6, lr=0.01, steps=1500)


def pre_post(ids) -> dict:
    """Pre-LN vs post-LN, 6 blocks, peak learning rate 0.01, with and
    without a 200-step warmup (cosine after it), 1,500 steps, three seeds."""
    rows = []
    for norm in ("pre", "post"):
        for warm in (0, 200):
            cfg = base_cfg(model=dict(PART1, n_blocks=PRE_POST["depth"], norm=norm), steps=PRE_POST["steps"],
                           schedule=dict(kind="cosine", lr=PRE_POST["lr"], warmup=warm, min_ratio=0.1))
            runs = seeds(cfg, ids)
            steps_, mean = mean_curve(runs)
            rows.append({"norm": norm, "warmup": warm, "runs": [slim(r) for r in runs],
                         "evalSteps": steps_, "meanEvals": mean, **summarise(runs)})
    return {**PRE_POST, "rows": rows}


def norm_kind(ids) -> dict:
    """LayerNorm (the default run) vs RMSNorm, otherwise the default run, three seeds."""
    rows = []
    for kind in ("layer", "rms"):
        model = PART1 if kind == "layer" else dict(PART1, norm_kind="rms")
        runs = seeds(base_cfg(model=model), ids)
        rows.append({"kind": kind, "params": int(sum(v.size for v in M.init_params(model).values())),
                     "runs": [slim(r) for r in runs], **summarise(runs)})
    return {"rows": rows}


COORD_WIDTHS = [16, 32, 64, 128, 256]
COORD = dict(steps=10, lr=0.01, batch=16)


def coord_check(ids) -> dict:
    """muP's coordinate check (Yang et al. 2022, arXiv:2203.03466, §D): train
    each width for 10 AdamW steps at the same learning rate (constant, no
    clipping, the same batches) and measure how far the output logits and
    the last block's residual stream moved, RMS per coordinate. Under the
    standard parameterisation the change grows with width (wider = a
    bigger effective step); under muP it stays about constant, which is
    why one learning rate can serve every width."""
    out = {"widths": COORD_WIDTHS, **COORD, "sp": [], "mup": []}
    train, _ = D.split(ids)
    xv, _ = D.sample_batch(train, 32, PART1["seq_len"], M.mulberry32(17))
    for param in ("sp", "mup"):
        for d in COORD_WIDTHS:
            model = dict(PART1, d_model=d, d_ff=2 * d)
            if param == "mup":
                model["mup_base"] = 16
            cfg = base_cfg(model=model, steps=COORD["steps"], batch=COORD["batch"], clip=None,
                           schedule=dict(kind="constant", lr=COORD["lr"]))
            p0 = M.init_params(model)
            l0, c0 = M.forward(p0, model, xv)
            run = train_ext(cfg, ids, eval_every=0, keep_params=True)
            l1, c1 = M.forward(run["params"], model, xv)
            out[param].append({
                "width": d,
                "logitsDelta": float(np.sqrt(((l1 - l0) ** 2).mean())),
                "residualDelta": float(np.sqrt(((c1["x_last"] - c0["x_last"]) ** 2).mean())),
            })
    return out


MUP_WIDTHS = [16, 64]
MUP_LRS = [0.01, 0.03, 0.1]
MUP_STEPS = 1500


def mup_transfer(ids) -> dict:
    """Validation loss after 1,500 steps for widths 16 and 64 at three
    learning rates, standard parameterisation vs muP (base width 16, so
    width 16 is the same model either way), three seeds."""
    rows = []
    for param in ("sp", "mup"):
        for d in MUP_WIDTHS:
            for lr in MUP_LRS:
                model = dict(PART1, d_model=d, d_ff=2 * d)
                if param == "mup":
                    model["mup_base"] = 16
                cfg = base_cfg(model=model, steps=MUP_STEPS,
                               schedule=dict(kind="cosine", lr=lr, warmup=50, min_ratio=0.1))
                runs = seeds(cfg, ids, eval_every=500)
                rows.append({"param": param, "width": d, "lr": lr,
                             "params": int(sum(v.size for v in M.init_params(model).values())),
                             "runs": [slim(r) for r in runs], **summarise(runs)})
    return {"widths": MUP_WIDTHS, "lrs": MUP_LRS, "steps": MUP_STEPS, "rows": rows}


# ---------------------------------------------------------------------------
# Chapter 8: stability
# ---------------------------------------------------------------------------
BAD = dict(at=1000, steps=1400, lr=0.5, char="@")


def bad_batch_run(cfg: dict, ids, skip: bool) -> dict:
    """Train with one corrupted batch at step BAD["at"] (every window the
    character '@' repeated: a batch unlike the corpus, standing in for the
    rare bad batches of real data). `skip`: drop any batch whose gradient
    norm exceeds 4 x the mean of the last 50 (a guard some large runs use)."""
    model = cfg["model"]
    S = model["seq_len"]
    params = M.init_params(model)
    o = cfg["optimizer"]
    opt = O.Optimizer(o["kind"], params, momentum=o.get("momentum", 0.9), beta2=o.get("beta2", 0.999),
                      weight_decay=o.get("weight_decay", 0.0))
    rng = M.mulberry32(cfg["data_seed"])
    train, val = D.split(ids)
    vx, vy = D.val_batch(val, S)
    sched = dict(cfg["schedule"], total=cfg["steps"])
    norms, evals, skipped = [], [], []
    for t in range(cfg["steps"]):
        x, y = D.sample_batch(train, cfg["batch"], S, rng)
        if t == BAD["at"]:
            x = np.full_like(x, M.CHAR_TO_ID[BAD["char"]])
            y = x.copy()
        loss, g, _ = M.loss_and_grads(params, model, x, y)
        n = O.global_norm(g)
        drop = False
        if skip and len(norms) >= 50:
            recent = 0.0
            for v in norms[-50:]:
                recent += v
            drop = n > 4 * recent / 50
        if drop:
            skipped.append(t)
        else:
            norms.append(n)
            if cfg["clip"]:
                O.clip_by_global_norm(g, cfg["clip"])
            opt.step(params, g, O.lr_at(t, sched))
        if t % 100 == 99 or BAD["at"] - 1 <= t <= BAD["at"] + 20 or t == cfg["steps"] - 1:
            v = M.loss_only(params, model, vx, vy)
            evals.append({"step": t + 1, "val": v if math.isfinite(v) else None})
            if not math.isfinite(v) or v > 1e6:
                break
    return {"evals": evals, "skipped": skipped}


BAD_CASES = [
    ("momentum, no clipping", dict(kind="momentum"), None, False),
    ("momentum, clip at 1.0", dict(kind="momentum"), 1.0, False),
    ("momentum, skip the batch", dict(kind="momentum"), None, True),
    ("AdamW (lr 0.01), no clipping", dict(kind="adamw", beta2=0.99, weight_decay=0.01), None, False),
]


def bad_batch(ids) -> dict:
    rows = []
    for name, opt, clip, skip in BAD_CASES:
        lr = BAD["lr"] if opt["kind"] == "momentum" else 0.01
        runs = []
        for s in SEEDS:
            cfg = base_cfg(optimizer=opt, clip=clip, steps=BAD["steps"], data_seed=s,
                           schedule=dict(kind="constant", lr=lr))
            runs.append(bad_batch_run(cfg, ids, skip))
        # the validation loss just before the bad step and just after it
        before = [[e for e in r["evals"] if e["step"] <= BAD["at"]][-1]["val"] for r in runs]
        after = [[e for e in r["evals"] if e["step"] == BAD["at"] + 1][0]["val"] for r in runs]
        final = [r["evals"][-1]["val"] if r["evals"][-1]["step"] == BAD["steps"] else None for r in runs]
        rows.append({"name": name, "lr": lr, "clip": clip, "skip": skip, "runs": runs,
                     "before": before, "after": after, "final": final})
    return {**BAD, "rows": rows}


ATTN_CASES = [
    ("baseline", {}),
    ("QK-norm", {"qk_norm": True}),
    ("z-loss 1e-3", {"z_loss": 1e-3}),
]
ATTN_LRS = [0.01, 0.1]


def attention_growth(ids) -> dict:
    """AdamW at peak learning rates 0.01 and 0.1 (warmup 50 + cosine,
    2,000 steps, clip 1.0), baseline vs QK-norm vs z-loss, three seeds,
    with the largest attention score and the attention entropy every 100
    steps on the validation windows (Wortsman et al. 2023, arXiv:2309.14322)."""
    rows = []
    for lr in ATTN_LRS:
        for name, extra in ATTN_CASES:
            cfg = base_cfg(model=dict(PART1, **extra), schedule=dict(kind="cosine", lr=lr, warmup=50, min_ratio=0.1))
            runs = seeds(cfg, ids, stats_every=100)
            k = len(runs[0]["stats"])
            stats = [{"step": runs[0]["stats"][i]["step"],
                      "maxScore": float(np.mean([r["stats"][i]["maxScore"] for r in runs])),
                      "attnEntropy": float(np.mean([r["stats"][i]["attnEntropy"] for r in runs]))}
                     for i in range(k)]
            rows.append({"name": name, "lr": lr, "runs": [slim(r) for r in runs], "meanStats": stats,
                         **summarise(runs)})
    return {"lrs": ATTN_LRS, "rows": rows}


SPIKE_SNAPSHOTS = [0, 100, 200, 400, 800, 2000]


def attention_snapshots(ids) -> dict:
    """The baseline run at lr 0.1, seed 1, with its weights at a few steps
    (public/training/attention-snapshots.json): chapter 8's animation runs
    the site's own forward pass on them."""
    cfg = base_cfg(schedule=dict(kind="cosine", lr=0.1, warmup=50, min_ratio=0.1))
    last = SPIKE_SNAPSHOTS[-1]
    run = train_ext(cfg, ids, stats_every=100, snapshots=set(SPIKE_SNAPSHOTS) - {last}, keep_params=True)
    snaps = dict(run["snapshots"])
    snaps[last] = run["params"]
    return {"steps": SPIKE_SNAPSHOTS, "params": {t: snaps[t] for t in SPIKE_SNAPSHOTS},
            "evals": run["evals"], "stats": run["stats"]}


# ---------------------------------------------------------------------------
# Chapter 9: mixed precision and quantised gradients
# ---------------------------------------------------------------------------
PRECISIONS = [
    ("float64 (reference)", None),
    ("FP32", dict(fwd="fp32", bwd="fp32")),
    ("BF16, FP32 master weights", dict(fwd="bf16", bwd="bf16")),
    ("BF16, no master copy", dict(fwd="bf16", bwd="bf16", master=False)),
    ("FP16, no loss scaling", dict(fwd="fp16", bwd="fp16")),
    ("FP16, dynamic loss scaling", dict(fwd="fp16", bwd="fp16", loss_scale="dynamic")),
    ("FP8 (E4M3 / E5M2), no scaling", dict(fwd="e4m3", bwd="e5m2", out="bf16", out_bwd="bf16")),
    ("FP8 (E4M3 / E5M2), per-tensor scaling", dict(fwd="e4m3", bwd="e5m2", scaled=True, out="bf16",
                                                   out_bwd="bf16")),
]


def precision_runs(ids) -> dict:
    """The default run in each emulated precision recipe, three seeds."""
    rows = []
    for name, prec in PRECISIONS:
        runs = seeds(base_cfg(precision=prec) if prec else base_cfg(), ids)
        rows.append({"name": name, "precision": prec, "runs": [slim(r) for r in runs], **summarise(runs),
                     "gradFlushedToZero": (float(np.mean([r["precision"]["gradFlushedToZero"] for r in runs]))
                                           if prec else 0.0),
                     "skipped": (float(np.mean([r["precision"]["skipped"] for r in runs])) if prec else 0.0),
                     "updatesLost": (float(np.mean([r["precision"]["updatesLost"] for r in runs]))
                                     if prec and not prec.get("master", True) else None)})
    return {"rows": rows}


GRAD_BITS = [8, 4, 2]


def grad_quant(ids) -> dict:
    """The default run with every weight gradient quantised to k-bit
    integers (symmetric, one scale per tensor, full scale = the tensor's
    largest magnitude), three seeds."""
    rows = []
    for bits in GRAD_BITS:
        runs = seeds(base_cfg(grad_bits=bits), ids)
        rows.append({"bits": bits, "levels": 2 * ((1 << (bits - 1)) - 1) + 1,
                     "runs": [slim(r) for r in runs], **summarise(runs)})
    return {"rows": rows}


# ---------------------------------------------------------------------------
# Chapter 10: memory
# ---------------------------------------------------------------------------
def memory_results(ids) -> dict:
    """The calculator's numbers for LLaMA 7B (checked: 6.7B parameters, as
    Touvron et al. 2023 Table 2) and Part 1's model, and the floats the
    NumPy model's forward pass actually keeps for the backward pass."""
    import memory as MEM

    train, _ = D.split(ids)
    x, _ = D.sample_batch(train, 16, PART1["seq_len"], M.mulberry32(1))
    _, caches = M.forward(M.init_params(PART1), PART1, x)
    saved = MEM.tiny_saved_floats(caches)
    mixed = dict(weightBytes=2, gradBytes=2, optimizer="adam", master=True)
    fp32 = dict(weightBytes=4, gradBytes=4, optimizer="adam", master=False)
    s, h, a = PART1["seq_len"], PART1["d_model"], PART1["n_heads"]
    by = {r: MEM.breakdown(MEM.LLAMA_7B, mixed, r) for r in ("none", "selective", "full")}
    return {
        "llama": {
            "arch": MEM.LLAMA_7B,
            "mixed": by,
            "fp32": MEM.breakdown(MEM.LLAMA_7B, fp32, "none"),
            # the same in GB (1e9 bytes), for the text
            "gb": {"weights": by["none"]["weights"] / 1e9, "static": by["none"]["static"] / 1e9,
                   "activations": by["none"]["activations"] / 1e9,
                   "peak": {r: by[r]["peak"] / 1e9 for r in by}},
        },
        "part1": {
            "arch": MEM.PART1_MEM,
            "mixed": MEM.breakdown(MEM.PART1_MEM, mixed, "none"),
            "savedFloatsPerSeqPerBlock": saved["perBlock"],
            # Korthikanti et al.'s count for the same shape, in values (bytes / 2)
            "korthikantiValuesPerSeqPerBlock": s * h * (34 + 5 * a * s / h) / 2,
        },
    }
