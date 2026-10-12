"""
jobs.py — chapter 15, "Running a real job": what the default run looks like
to the person watching it, and what saving and resuming it takes.

  monitor     the default run again, step for step, logging what a training
              dashboard shows: the loss, the gradient norm before clipping,
              the learning rate and the update-to-weight ratio
              |Delta theta| / |theta| (over all parameters and for one
              matrix). It is the default run: its final validation loss must
              equal results.json's.
  resume      the default run stopped at step 1,000 and resumed three ways:
                full       weights, AdamW's moments and step count, and the
                           data position (the random generator's draws):
                           must continue bit for bit as if never stopped
                weights    weights only: a fresh AdamW (moments zero, step 0)
                fresh data weights and optimiser, but the data generator
                           restarted from its seed, so the first 1,000 steps'
                           batches are drawn again
              three seeds each; the validation loss every 50 steps.
"""

from __future__ import annotations

import math

import numpy as np

import data as D
import model as M
import optim as O

PART1 = dict(M.PART1_CONFIG)
SEEDS = [1, 2, 3]
STOP = 1000


def default_cfg(seed: int = 1) -> dict:
    return dict(batch=16, steps=2000, lr=0.01, warmup=50, min_ratio=0.1, clip=1.0, data_seed=seed)


def new_opt(params):
    return O.Optimizer("adamw", params, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.01)


def skip_draws(rng, n: int) -> None:
    for _ in range(n):
        rng()


def run(cfg: dict, ids, start: int = 0, state: dict | None = None, stop: int | None = None,
        log_every: int = 10, eval_every: int = 50, monitor: bool = False) -> dict:
    """The default run's loop (make_fixtures.train_run) from step `start`,
    optionally from a saved `state` (params, opt, rng), until `stop`."""
    model = PART1
    S = model["seq_len"]
    train, val = D.split(ids)
    vx, vy = D.val_batch(val, S)
    sched = dict(kind="cosine", lr=cfg["lr"], warmup=cfg["warmup"], min_ratio=cfg["min_ratio"],
                 total=cfg["steps"])
    if state is None:
        params = M.init_params(model)
        opt = new_opt(params)
        rng = M.mulberry32(cfg["data_seed"])
    else:
        params, opt, rng = state["params"], state["opt"], state["rng"]
    stop = cfg["steps"] if stop is None else stop
    log = {"steps": [], "loss": [], "gradNorm": [], "lr": [], "updateRatio": [], "updateRatioW1": []}
    evals = []
    for t in range(start, stop):
        if eval_every and t % eval_every == 0:
            evals.append({"step": t, "val": M.loss_only(params, model, vx, vy)})
        x, y = D.sample_batch(train, cfg["batch"], S, rng)
        loss, grads, _ = M.loss_and_grads(params, model, x, y)
        norm, _ = O.clip_by_global_norm(grads, cfg["clip"])
        lr = O.lr_at(t, sched)
        before = {k: v.copy() for k, v in params.items()} if monitor else None
        opt.step(params, grads, lr)
        if monitor and (t % log_every == 0 or t == cfg["steps"] - 1):
            num = math.sqrt(sum(float(((params[k] - before[k]) ** 2).sum()) for k in params))
            den = math.sqrt(sum(float((before[k] ** 2).sum()) for k in params))
            w = "block_1.ffn.W1"
            r1 = float(np.linalg.norm(params[w] - before[w]) / np.linalg.norm(before[w]))
            log["steps"].append(t)
            log["loss"].append(float(loss))
            log["gradNorm"].append(norm)
            log["lr"].append(lr)
            log["updateRatio"].append(num / den)
            log["updateRatioW1"].append(r1)
    if stop == cfg["steps"]:
        evals.append({"step": cfg["steps"], "val": M.loss_only(params, model, vx, vy)})
    return {"evals": evals, "log": log if monitor else None,
            "state": {"params": params, "opt": opt, "rng": rng}}


def monitor(ids) -> dict:
    r = run(default_cfg(), ids, monitor=True)
    steps_to = {}
    for key, target in (("t28", 2.8), ("t26", 2.6), ("t24", 2.4)):  # keys without dots (R paths)
        hit = [e["step"] for e in r["evals"] if e["val"] <= target]
        steps_to[key] = hit[0] if hit else None
    return {"log": r["log"], "final": r["evals"][-1]["val"], "evals": r["evals"], "stepsTo": steps_to}


def copy_opt(opt) -> object:
    new = new_opt({k: v for k, v in opt.m.items()})
    new.t = opt.t
    new.m = {k: v.copy() for k, v in opt.m.items()}
    new.v = {k: v.copy() for k, v in opt.v.items()}
    return new


def resume(ids) -> dict:
    """Stop at STOP, resume three ways, three seeds each."""
    rows = {"full": [], "weights": [], "freshData": [], "uninterrupted": []}
    identical = True
    for seed in SEEDS:
        cfg = default_cfg(seed)
        whole = run(cfg, ids)
        rows["uninterrupted"].append(whole["evals"])
        first = run(cfg, ids, stop=STOP)
        saved = first["state"]
        params = {k: v.copy() for k, v in saved["params"].items()}
        # full: the generator's position is the number of draws so far
        rng = M.mulberry32(seed)
        skip_draws(rng, STOP * cfg["batch"])
        full = run(cfg, ids, start=STOP, state={"params": params, "opt": copy_opt(saved["opt"]), "rng": rng})
        same = (full["evals"][-1]["val"] == whole["evals"][-1]["val"]
                and all(np.array_equal(full["state"]["params"][k], whole["state"]["params"][k]) for k in params))
        identical = identical and same
        rows["full"].append(first["evals"] + full["evals"])
        # weights only
        p2 = {k: v.copy() for k, v in saved["params"].items()}
        rng2 = M.mulberry32(seed)
        skip_draws(rng2, STOP * cfg["batch"])
        w = run(cfg, ids, start=STOP, state={"params": p2, "opt": new_opt(p2), "rng": rng2})
        rows["weights"].append(first["evals"] + w["evals"])
        # fresh data: weights and optimiser, generator restarted
        p3 = {k: v.copy() for k, v in saved["params"].items()}
        fd = run(cfg, ids, start=STOP, state={"params": p3, "opt": copy_opt(saved["opt"]),
                                              "rng": M.mulberry32(seed)})
        rows["freshData"].append(first["evals"] + fd["evals"])

    def summ(curves):
        steps = [e["step"] for e in curves[0]]
        mean = [float(np.mean([c[i]["val"] for c in curves])) for i in range(len(steps))]
        finals = [c[-1]["val"] for c in curves]
        return {"evalSteps": steps, "meanEvals": mean, "finals": finals,
                "mean": float(np.mean(finals)), "sd": float(np.std(finals, ddof=1)),
                "afterResume": float(np.mean([c[steps.index(STOP + 50)]["val"] for c in curves]))}

    return {"stop": STOP, "seeds": SEEDS, "bitIdentical": bool(identical),
            **{k: summ(v) for k, v in rows.items()}}
