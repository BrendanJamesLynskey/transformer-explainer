"""
make_fixtures.py — Part 2's (Training) reference runs: the correctness
checks, the fixtures the TypeScript trainer must match, and the measured
results every number on the /training pages comes from.

    .venv-fixtures/bin/python scripts/training/make_fixtures.py          # write everything
    .venv-fixtures/bin/python scripts/training/make_fixtures.py --check  # CI: recompute, compare
    ... make_fixtures.py --groups ch6a,ch8            # recompute (or --check) some groups only

Groups (CI runs one job per group, in parallel): "base" is brief 29A's
checks, fixtures, default run and optimiser comparison; the rest are brief
29B's experiments (GROUPS below), each about 2-8 minutes on one CPU core.

Writes:
  tests/unit/fixtures/training/*.json   fixtures for tests/unit/training/*
  src/data/training/results.json        the measured numbers the pages show
  public/training/default-run-weights.json  the default run's final weights
  results.md                            the same, for people (README links it)

--check recomputes everything in memory, fails if a correctness check
fails, and compares the result with the committed files by the tolerances in
RULES: 1e-9 relative for every fixture (the chapter 3 and 4 fixtures are
rebuilt from the committed weights), looser for the long training runs,
whose last-bit differences between CPUs grow over thousands of steps (see
RULES); everything else exactly. results.md must be the rendering of
results.json.

Determinism: NumPy float64 with one BLAS thread (bit-reproducible on one
machine); the only NumPy randomness is the seeded perturbation in
checks.py; torch 2.11.0 CPU (pinned in scripts/training/requirements.txt).
Training at a high learning rate is chaotic, so a last-bit difference
between CPUs in a matrix product grows over a few hundred steps: --check
uses per-path tolerances (RULES).
"""

from __future__ import annotations

import os

# One BLAS thread (set before NumPy loads), so a run is bit-reproducible on
# one machine. Across CPUs the last bits differ; see RULES below.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import hashlib
import json
import math
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import anim  # noqa: E402
import anim_b as AB  # noqa: E402
import anim_c as AC  # noqa: E402
import results_c as RC  # noqa: E402
import slowlinks as SL  # noqa: E402
import checks as C  # noqa: E402
import experiments as X  # noqa: E402
import data as D  # noqa: E402
import model as M  # noqa: E402
import optim as O  # noqa: E402

ROOT = HERE.parents[1]
FIX = ROOT / "tests" / "unit" / "fixtures" / "training"
RESULTS_JSON = ROOT / "src" / "data" / "training" / "results.json"
RESULTS_MD = ROOT / "results.md"
CHECKPOINT = ROOT / "public" / "training" / "default-run-weights.json"
SNAPSHOTS = ROOT / "public" / "training" / "attention-snapshots.json"

PART1 = dict(M.PART1_CONFIG)

# The default run (src/lib/training/trainer.ts:DEFAULT_TRAIN).
DEFAULT_TRAIN = dict(
    model=PART1, batch=16, steps=2000,
    optimizer=dict(kind="adamw", beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.01),
    schedule=dict(kind="cosine", lr=0.01, warmup=50, min_ratio=0.1),
    clip=1.0, data_seed=1,
)
EVAL_EVERY = 100
SAMPLE_PROMPT = "alice "
SAMPLE_CHARS = 120
SAMPLE_TEMPERATURE = 0.8
SAMPLE_SEED = 7


def tolist(a):
    return np.asarray(a, dtype=np.float64).tolist()


def corpus_ids() -> list[int]:
    return M.encode_chars(D.load_corpus())


# ---------------------------------------------------------------------------
# Training runs
# ---------------------------------------------------------------------------
def make_optimizer(params, spec):
    kind = spec["kind"]
    return O.Optimizer(
        "adam" if kind == "adam_l2" else kind, params,
        momentum=spec.get("momentum", 0.9), beta1=spec.get("beta1", 0.9),
        beta2=spec.get("beta2", 0.999), eps=spec.get("eps", 1e-8),
        weight_decay=spec.get("weight_decay", 0.0),
    )


def train_run(cfg: dict, ids: list[int], eval_every=EVAL_EVERY, keep_params=False,
              log_every=1) -> dict:
    """The training loop of src/lib/training/trainer.ts, step for step."""
    model = cfg["model"]
    S = model["seq_len"]
    params = M.init_params(model)
    opt = make_optimizer(params, cfg["optimizer"])
    rng = M.mulberry32(cfg["data_seed"])
    train, val = D.split(ids)
    vx, vy = D.val_batch(val, S)
    sched = dict(cfg["schedule"], total=cfg["steps"])
    steps, losses, norms, lrs, evals = [], [], [], [], []
    t0 = time.time()
    for t in range(cfg["steps"]):
        if eval_every and t % eval_every == 0:
            evals.append({"step": t, "val": M.loss_only(params, model, vx, vy)})
        x, y = D.sample_batch(train, cfg["batch"], S, rng)
        loss, grads, _ = M.loss_and_grads(params, model, x, y)
        norm = O.clip_by_global_norm(grads, cfg["clip"])[0] if cfg["clip"] else float("nan")
        lr = O.lr_at(t, sched)
        opt.step(params, grads, lr)
        if t % log_every == 0 or t == cfg["steps"] - 1:
            steps.append(t)
            losses.append(float(loss))
            norms.append(norm)
            lrs.append(lr)
    evals.append({"step": cfg["steps"], "val": M.loss_only(params, model, vx, vy)})
    out = {"steps": steps, "loss": losses, "gradNorm": norms, "lr": lrs, "evals": evals,
           "seconds": time.time() - t0}
    if keep_params:
        out["params"] = params
    return out


def ts_softmax(x):
    m = max(x)
    e = [math.exp(v - m) for v in x]
    s = 0.0
    for v in e:
        s += v
    return [v / s for v in e]


def sample_from_probs(probs, rng) -> int:
    total = 0.0
    for p in probs:
        total += p
    u = rng() * total
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if u < acc:
            return i
    return len(probs) - 1


def sample_text(params, model, prompt_ids, n, temperature, seed) -> str:
    """Port of trainer.ts:sampleText (sampling.ts's temperature mode)."""
    rng = M.mulberry32(seed)
    ids = list(prompt_ids)
    out = ""
    for _ in range(n):
        ctx = np.array([ids[-model["seq_len"]:]], dtype=np.int64)
        logits, _ = M.forward(params, model, ctx)
        last = [v / temperature for v in logits[0, -1].tolist()]
        nxt = sample_from_probs(ts_softmax(last), rng)
        ids.append(nxt)
        out += M.ALPHABET[nxt]
    return out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def gradients_fixture(ids) -> tuple[dict, dict]:
    train, _ = D.split(ids)
    x, y = D.sample_batch(train, 3, PART1["seq_len"], M.mulberry32(7))
    p0 = M.init_params(PART1)
    pp = C.perturbed(p0, 0)
    cases = []
    record = []
    for name, params in (("init", p0), ("perturbed", pp)):
        rec = [] if name == "init" else None
        loss, grads, per = M.loss_and_grads(params, PART1, x, y, rec)
        case = {"name": name, "loss": float(loss), "perPosition": tolist(per),
                "grads": {k: tolist(v) for k, v in grads.items()}}
        if name == "perturbed":
            case["params"] = {k: tolist(v) for k, v in params.items()}
        else:
            record = [{"name": n, "value": tolist(v)} for n, v in rec]
        cases.append(case)
    checks = {
        "init": C.check_init(PART1),
        "autograd_init": C.check_autograd(p0, PART1, x, y),
        "autograd_perturbed": C.check_autograd(pp, PART1, x, y),
        "fd_init": C.check_finite_differences(p0, PART1, x, y),
        "fd_perturbed": C.check_finite_differences(pp, PART1, x, y),
    }
    fix = {"config": PART1, "inputs": x.tolist(), "targets": y.tolist(), "cases": cases,
           "record": record}
    return fix, checks


OPT_SPECS = [
    dict(kind="sgd"),
    dict(kind="momentum", momentum=0.9),
    dict(kind="adam"),
    dict(kind="adam_l2", weight_decay=0.01),
    dict(kind="adamw", weight_decay=0.1),
]


def optimizers_fixture(ids) -> tuple[dict, dict]:
    train, _ = D.split(ids)
    rng = M.mulberry32(3)
    batches = [D.sample_batch(train, 4, PART1["seq_len"], rng) for _ in range(10)]
    sched = dict(kind="cosine", lr=0.01, total=10, warmup=3, min_ratio=0.1)
    runs = []
    for spec in OPT_SPECS:
        params = M.init_params(PART1)
        opt = make_optimizer(params, spec)
        losses, norms, lrs = [], [], []
        for t, (x, y) in enumerate(batches):
            loss, grads, _ = M.loss_and_grads(params, PART1, x, y)
            norm, _ = O.clip_by_global_norm(grads, 1.0)
            lr = O.lr_at(t, sched)
            opt.step(params, grads, lr)
            losses.append(float(loss))
            norms.append(norm)
            lrs.append(lr)
        runs.append({"spec": spec, "loss": losses, "gradNorm": norms, "lr": lrs,
                     "params": {k: tolist(v) for k, v in params.items()}})
    fix = {"config": PART1, "schedule": sched, "clip": 1.0,
           "batches": [{"inputs": x.tolist(), "targets": y.tolist()} for x, y in batches],
           "runs": runs}
    torch_checks = C.check_optimisers(M.init_params(PART1), PART1,
                                      [D.sample_batch(train, 4, PART1["seq_len"], M.mulberry32(9))
                                       for _ in range(25)])
    torch_checks["cosine_schedule"] = C.check_cosine_schedule()
    return fix, torch_checks


SCHEDULES = [
    dict(kind="constant", lr=0.01, total=100),
    dict(kind="cosine", lr=0.01, total=100, warmup=10, min_ratio=0.1),
    dict(kind="cosine", lr=0.003, total=100, warmup=0, min_ratio=0.0),
    dict(kind="wsd", lr=0.01, total=100, warmup=10, min_ratio=0.1, decay_frac=0.2),
]


def schedules_fixture() -> dict:
    return {"schedules": [{"spec": s, "lr": [O.lr_at(t, s) for t in range(s["total"])]}
                          for s in SCHEDULES]}


BPE_MERGES = 64
BPE_EXCERPT_CHARS = 160


def bpe_fixture(text: str) -> dict:
    model = D.bpe_train(text, BPE_MERGES)
    excerpt = text[:BPE_EXCERPT_CHARS]
    enc = D.bpe_encode(text, model["merges"])
    return {"merges": model["merges"], "vocab": model["vocab"], "history": model["history"],
            "excerpt": excerpt, "excerptIds": D.bpe_encode(excerpt, model["merges"]),
            "corpusChars": len(text), "corpusTokens": len(enc)}


def train_fixture(ids) -> dict:
    cfg = dict(DEFAULT_TRAIN, steps=200)
    run = train_run(cfg, ids, eval_every=100, keep_params=True)
    params = run.pop("params")
    run.pop("seconds")
    prompt = M.encode_chars(SAMPLE_PROMPT)
    run["sample"] = sample_text(params, PART1, prompt, 40, SAMPLE_TEMPERATURE, SAMPLE_SEED)
    run["config"] = {"batch": cfg["batch"], "steps": cfg["steps"], "dataSeed": cfg["data_seed"]}
    run["tok_emb_row0"] = tolist(params["tok_emb"][0])
    run["ffn_W2_last"] = tolist(params["block_1.ffn.W2"][-1])
    return run


# ---------------------------------------------------------------------------
# Measured results (results.json / results.md)
# ---------------------------------------------------------------------------
COMPARE_SEEDS = [1, 2, 3]
OPT_COMPARE = {
    "sgd": [0.1, 0.3, 1.0],
    "momentum": [0.03, 0.1, 0.3],
    "adam": [0.003, 0.01, 0.03],
    "adamw": [0.003, 0.01, 0.03],
}


def results(ids, checks_g, checks_o, bpe) -> dict:
    text = D.load_corpus()
    train, val = D.split(ids)
    out: dict = {}
    out["corpus"] = {
        "title": "Alice's Adventures in Wonderland (Lewis Carroll, 1865)",
        "source": "Project Gutenberg eBook #11 (public domain)",
        "chars": len(text), "trainChars": len(train), "valChars": len(val),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "unigramEntropy": unigram_entropy(train),
        "bpeTokens": bpe["corpusTokens"], "bpeMerges": BPE_MERGES,
    }
    p0 = M.init_params(PART1)
    out["model"] = {"config": PART1, "params": int(sum(v.size for v in p0.values())),
                    "lnV": math.log(PART1["vocab_size"])}
    # the default run, logged every 10 steps
    run = train_run(DEFAULT_TRAIN, ids, keep_params=True, log_every=10)
    params = run.pop("params")
    prompt = M.encode_chars(SAMPLE_PROMPT)
    run["samples"] = {
        "prompt": SAMPLE_PROMPT,
        "init": sample_text(p0, PART1, prompt, SAMPLE_CHARS, SAMPLE_TEMPERATURE, SAMPLE_SEED),
        "trained": sample_text(params, PART1, prompt, SAMPLE_CHARS, SAMPLE_TEMPERATURE, SAMPLE_SEED),
    }
    run["config"] = {k: v for k, v in DEFAULT_TRAIN.items() if k != "model"}
    out["defaultRun"] = run
    out["defaultRunParams"] = params  # popped by build(): written as the checkpoint
    # optimiser comparison: each optimiser at three learning rates, constant LR,
    # 1,500 steps, three data seeds (the batches differ; the initial weights
    # are Part 1's in every run); mean and standard deviation of the final
    # validation loss over the seeds
    compare = []
    for kind, lrs in OPT_COMPARE.items():
        for lr in lrs:
            spec = {"kind": kind, "beta2": 0.99}
            if kind == "adamw":
                spec["weight_decay"] = 0.01
            runs = []
            for seed in COMPARE_SEEDS:
                cfg = dict(DEFAULT_TRAIN, steps=1500, optimizer=spec, data_seed=seed,
                           schedule=dict(kind="constant", lr=lr))
                r = train_run(cfg, ids, log_every=10)
                runs.append({"seed": seed, "evals": r["evals"], "final": r["evals"][-1]["val"],
                             "seconds": r["seconds"]})
            finals = np.array([r["final"] for r in runs])
            compare.append({"kind": kind, "lr": lr, "runs": runs, "final": float(finals.mean()),
                            "sd": float(finals.std(ddof=1)),
                            "evalSteps": [e["step"] for e in runs[0]["evals"]],
                            "meanEvals": [float(np.mean([r["evals"][i]["val"] for r in runs]))
                                          for i in range(len(runs[0]["evals"]))]})
    out["optimizerCompare"] = compare
    S, B, steps = PART1["seq_len"], DEFAULT_TRAIN["batch"], DEFAULT_TRAIN["steps"]
    n_params = out["model"]["params"]
    final_val = run["evals"][-1]["val"]
    out["derived"] = {
        "predictionsPerStep": B * S,
        "tokensSeen": B * S * steps,
        "epochs": B * S * steps / len(train),
        "trainTokensPerParam": len(train) / n_params,
        "charsPerBpeToken": len(text) / bpe["corpusTokens"],
        "valPerplexity": math.exp(final_val),
        "initPerplexity": math.exp(run["evals"][0]["val"]),
        "unigramPerplexity": math.exp(out["corpus"]["unigramEntropy"]),
    }
    out["checks"] = {"gradients": strip(checks_g), "optimizers": checks_o}
    out["environment"] = {"python": platform.python_version(), "numpy": np.__version__,
                          "torch": torch.__version__, "machine": platform.machine()}
    return out


def unigram_entropy(ids) -> float:
    counts = np.bincount(np.array(ids), minlength=64).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def strip(checks: dict) -> dict:
    out = {}
    for k, v in checks.items():
        out[k] = {kk: vv for kk, vv in v.items() if kk != "per_param"}
    return out


# ---------------------------------------------------------------------------
# Brief 29B's experiments (chapters 6-10), by group
# ---------------------------------------------------------------------------
GROUPS = {
    "ch6a": ["lrSweep", "schedules"],
    "ch6b": ["batchSize", "noiseScale"],
    "ch7a": ["depthScale", "prePost", "normKind", "coordCheck"],
    "ch7b": ["mupTransfer"],
    "ch8": ["badBatch", "attnGrowth", "attnRun"],
    "ch9": ["precision"],
    "ch10": ["gradQuant", "memory", "gradRounding"],
    # brief 29C (chapters 11-16): results_c.py
    **RC.GROUPS_C,
}
ALL_GROUPS = ["base"] + list(GROUPS)


def results_b(ids, groups) -> tuple[dict, dict | None]:
    """The results.json keys of the 29B groups asked for, and (group ch8)
    the high-learning-rate run's weight snapshots."""
    out: dict = {}
    snap = None
    if "ch6a" in groups:
        out["lrSweep"] = X.lr_sweep(ids)
        out["schedules"] = X.schedules(ids)
    if "ch6b" in groups:
        out["batchSize"] = X.batch_size(ids)
        out["noiseScale"] = X.noise_scale(ids)
    if "ch7a" in groups:
        out["depthScale"] = [{"std": s, "pre": X.depth_scale(s, "pre"), "post": X.depth_scale(s, "post")}
                             for s in X.DEPTH_STDS]
        out["prePost"] = X.pre_post(ids)
        out["normKind"] = X.norm_kind(ids)
        out["coordCheck"] = X.coord_check(ids)
    if "ch7b" in groups:
        out["mupTransfer"] = X.mup_transfer(ids)
    if "ch8" in groups:
        out["badBatch"] = X.bad_batch(ids)
        out["attnGrowth"] = X.attention_growth(ids)
        snap = X.attention_snapshots(ids)
        out["attnRun"] = {"steps": snap["steps"], "evals": snap["evals"], "stats": snap["stats"]}
    if "ch9" in groups:
        out["precision"] = X.precision_runs(ids)
    if "ch10" in groups:
        out["gradQuant"] = X.grad_quant(ids)
        out["memory"] = X.memory_results(ids)
    return out, snap


def post_ln_fixture(ids) -> dict:
    """Post-LN (chapter 7): loss and gradients of a 4-block post-LN model at
    initialisation and perturbed, for src/lib/training/model-grad.ts. The
    reference's post-LN gradients are checked against autograd here too."""
    cfg = dict(PART1, n_blocks=4, norm="post")
    train, _ = D.split(ids)
    x, y = D.sample_batch(train, 3, PART1["seq_len"], M.mulberry32(11))
    p0 = M.init_params(cfg)
    pp = C.perturbed(p0, 1)
    cases = []
    for name, params in (("init", p0), ("perturbed", pp)):
        a = C.check_autograd(params, cfg, x, y)
        if not a["ok"]:
            sys.exit(f"post-LN autograd check failed: {a['max_scaled_err']}")
        loss, grads, _ = M.loss_and_grads(params, cfg, x, y)
        case = {"name": name, "loss": float(loss), "grads": {k: tolist(v) for k, v in grads.items()}}
        if name == "perturbed":
            case["params"] = {k: tolist(v) for k, v in params.items()}
        cases.append(case)
    return {"config": cfg, "inputs": x.tolist(), "targets": y.tolist(), "cases": cases}


def rounding_fixture(n: int = 400, seed: int = 3) -> dict:
    """Values and their rounding to every format (precision.round_to, the
    Numerics site's code) for src/lib/training/rounding.ts: random values
    over 50 binades, every FP8 value and midpoint, the edges."""
    import precision as Pr

    rng = np.random.default_rng(seed)
    xs = [float(v) for v in rng.standard_normal(n) * np.exp(rng.uniform(-25, 25, n))]
    for fid in ("e4m3", "e5m2"):
        vals = Pr.positive_values(fid)
        xs += vals + [(a + b) / 2 for a, b in zip(vals, vals[1:])]
    xs += [65504.0, 65520.0, 1e6, -1e6, 2.0 ** -24, 2.0 ** -25, 3 * 2.0 ** -26, 0.0]
    out = {}
    for fid in Pr.FORMAT_ORDER:
        out[fid] = [Pr.round_to(x, fid) for x in xs]
        out[fid + "_sat"] = [Pr.round_to(x, fid, True) for x in xs]
    def enc(v: float):  # JSON has no infinities or NaN
        if v != v:
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v

    return {"xs": xs, "rounded": {k: [enc(v) for v in vs] for k, vs in out.items()}}


def snapshot_file(snap: dict) -> dict:
    return {
        "note": "Part 1's model during the baseline run at peak learning rate 0.1 (results.md §8), "
                "9 significant digits; chapter 8's animation runs the site's forward pass on them.",
        "config": PART1, "steps": snap["steps"],
        "params": {str(t): {k: tolist(v) for k, v in AB.snapshot_round(snap["params"][t]).items()}
                   for t in snap["steps"]},
    }


def load_snapshots(payload: dict) -> dict:
    return {"steps": payload["steps"],
            "params": {int(t): {k: np.array(v, dtype=np.float64) for k, v in p.items()}
                       for t, p in payload["params"].items()}}


# ---------------------------------------------------------------------------
# Writing and checking
# ---------------------------------------------------------------------------
def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return None if math.isnan(f) else f
    return x


def dump(obj) -> str:
    return json.dumps(jsonable(obj), allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"


VOLATILE = {"seconds", "environment"}

# How --check compares a recomputed number with the committed one, by path
# (first match wins; the default is 1e-9 relative). The fixtures are short
# computations and agree across machines to the last few bits. The long runs
# do not: BLAS and SIMD paths differ between CPUs in the last bit, and
# training amplifies that, slowly for the default run, fast for the
# high-learning-rate runs of the optimiser comparison (chaotic). Measured
# errors of the correctness checks are machine-dependent too; their
# tolerances are enforced by the checks themselves.
RULES = [
    (r"\.checks\..*\.(max_abs_err|max_scaled_err|max_ratio|loss_err|loss_ours|loss_torch)$", ("skip", 0)),
    # chaotic at the higher learning rates: a CPU's last-bit differences
    # moved single runs' final losses by about 0.1 and points on their
    # curves by up to 0.2 (measured in CI, 2026-10-12), so this is a sanity
    # check, and the pages quote the mean and spread over three seeds
    # the curves: where a run leaves the plateau shifts, so skip them
    (r"optimizerCompare\[\d+\]\.(meanEvals\[\d+\]|runs\[\d+\]\.evals\[\d+\]\.val)$", ("skip", 0)),
    (r"optimizerCompare\[\d+\]\.", ("abs", 0.25)),
    (r"results\.json\.(defaultRun|derived)\.", ("rel", 1e-6)),
    # brief 29B. The same story for its sweeps, only more so: whether a run
    # leaves chapter 1's plateau, and when, depends on last bits, so on
    # another CPU a single seed's final loss moved by up to 0.35, a step
    # count by a factor of 2, and one run at learning rate 1.0 diverged
    # (measured in CI, 2026-10-12). So per-seed values, curves, spreads and
    # divergence flags are not compared; the means over seeds are sanity-
    # checked; and the claims the chapters make are re-checked on the
    # recomputed numbers (CLAIMS below). The deterministic parts (depth
    # scale, the 10-step coordinate check, the noise scale along the default
    # run, memory, rounding) are checked tightly.
    (r"\.(lrSweep|schedules|prePost|normKind|mupTransfer|attnGrowth|precision|gradQuant|badBatch|batchSize)\."
     r"(.*\.)?(runs|finals|perLr|stats|meanStats|evals|meanEvals|steps|before|after|final)(\[\d+\])?(\.|$)", ("skip", 0)),
    (r"\.(sd|diverged|divergedAt|skipped|gradFlushedToZero|updatesLost|bestLr)$", ("skip", 0)),
    (r"\.(lrSweep|schedules|prePost|normKind|mupTransfer|attnGrowth|precision|gradQuant)\..*\.mean$", ("abs", 0.35)),
    (r"\.batchSize\.(rows|fit)", ("skip", 0)),
    (r"\.attnRun\.", ("skip", 0)),
    (r"attention-snapshots\.json\.params\.", ("skip", 0)),
    (r"\.noiseScale\.", ("rel", 1e-4)),
    (r"\.coordCheck\.", ("rel", 1e-7)),
    (r"default-run-weights\.json\.params\.", ("abs", 1e-6)),
    # brief 29C. Slow links and the IsoFLOP sweep are as chaotic as 29B's
    # sweeps (which seeds leave the plateau, and when): per-seed values,
    # curves, step counts and the fits through them are not compared; the
    # means are sanity-checked; the claims are re-checked (CLAIMS).
    (r"\.slowLinks\..*\.(finals|stepsToTarget|meanStepsToTarget|meanEvals|evalSteps)(\[\d+\])?$", ("skip", 0)),
    (r"\.slowLinks\.rows\[\d+\]\.perLr\[\d+\]\.mean$", ("skip", 0)),
    (r"\.slowLinks\.rows\[\d+\]\.mean$", ("abs", 0.35)),
    (r"\.isoflop\.rows\[\d+\]\.points\[\d+\]\.(finals|sd|diverged)", ("skip", 0)),
    (r"\.isoflop\.rows\[\d+\]\.points\[\d+\]\.mean$", ("abs", 0.35)),
    (r"\.isoflop\.(rows\[\d+\]\.(fit|fitWidths|nOpt|dOpt|interior)|a|b|frontier)", ("skip", 0)),
    # the default run again (monitor) and the resume experiment: the
    # default run's own drift (29A: ~5e-9), a little more after a resume
    (r"\.monitor\.", ("rel", 1e-5)),
    (r"\.resume\.", ("rel", 1e-4)),
]


def _rows(r, key, **match):
    return [x for x in r[key]["rows"] if all(x.get(k) == v for k, v in match.items())]


def _mean(r, key, **match):
    return _rows(r, key, **match)[0]["mean"]


# What chapters 6-10 claim about the chaotic experiments, re-checked on the
# recomputed numbers in CI (each claim is robust to the CPU drift above:
# the margins are several times the spread over seeds). A claim whose
# experiment group was not recomputed is skipped.
CLAIMS = [
    ("lrSweep", "the best peak learning rate is 0.01 or 0.03",
     lambda r: r["lrSweep"]["bestLr"] in (0.01, 0.03)),
    ("lrSweep", "lr 0.001 is still on the plateau (> 2.9) and 1.0 learns far less than 0.01",
     lambda r: _mean(r, "lrSweep", lr=0.001) > 2.9
     and (_mean(r, "lrSweep", lr=1.0) is None or _mean(r, "lrSweep", lr=1.0) > _mean(r, "lrSweep", lr=0.01) + 0.3)),
    ("schedules", "every schedule leaves the plateau (mean < 2.6)",
     lambda r: all(x["mean"] < 2.6 for x in r["schedules"]["rows"])),
    ("batchSize", "64 windows need under half the steps of 4 windows, and B_crit is between 2 and 100 windows",
     lambda r: r["batchSize"]["rows"][-1]["steps"] < 0.5 * r["batchSize"]["rows"][0]["steps"]
     and 2 < r["batchSize"]["fit"]["bCrit"] < 100),
    ("noiseScale", "the noise scale grows over the run",
     lambda r: r["noiseScale"]["points"][-1]["bSimple"] > 10 * r["noiseScale"]["points"][0]["bSimple"]),
    ("prePost", "post-LN without warmup stays on the plateau (> 2.9); pre-LN without warmup trains (< 2.6)",
     lambda r: _mean(r, "prePost", norm="post", warmup=0) > 2.9 and _mean(r, "prePost", norm="pre", warmup=0) < 2.6),
    ("normKind", "RMSNorm trains as well as LayerNorm (within 0.2)",
     lambda r: _mean(r, "normKind", kind="rms") < _mean(r, "normKind", kind="layer") + 0.2),
    ("coordCheck", "the logits move > 2.5x more at width 256 than 16 under SP, and within 10% under muP",
     lambda r: r["coordCheck"]["sp"][-1]["logitsDelta"] > 2.5 * r["coordCheck"]["sp"][0]["logitsDelta"]
     and abs(r["coordCheck"]["mup"][-1]["logitsDelta"] / r["coordCheck"]["mup"][0]["logitsDelta"] - 1) < 0.1),
    ("mupTransfer", "at width 64, muP's best mean beats the standard parameterisation's best",
     lambda r: min(x["mean"] for x in _rows(r, "mupTransfer", param="mup", width=64))
     < min(x["mean"] for x in _rows(r, "mupTransfer", param="sp", width=64))),
    ("badBatch", "unclipped momentum diverges after the bad batch in every seed; clipped and skipped runs finish below 2.6",
     lambda r: all(v is None for v in r["badBatch"]["rows"][0]["final"])
     and all(v is not None and v < 2.6 for row in r["badBatch"]["rows"][1:3] for v in row["final"])),
    ("attnGrowth", "at lr 0.1 the baseline's largest score exceeds 1,000 and QK-norm's stays under 100; QK-norm's loss is lower by > 0.2",
     lambda r: _rows(r, "attnGrowth", lr=0.1, name="baseline")[0]["meanStats"][-1]["maxScore"] > 1000
     and _rows(r, "attnGrowth", lr=0.1, name="QK-norm")[0]["meanStats"][-1]["maxScore"] < 100
     and _mean(r, "attnGrowth", lr=0.1, name="QK-norm") < _mean(r, "attnGrowth", lr=0.1, name="baseline") - 0.2),
    ("precision", "FP32, BF16 with master weights and FP16 with loss scaling end within 0.15 of float64; unscaled FP8 stays on the plateau",
     lambda r: all(abs(_mean(r, "precision", name=n) - _mean(r, "precision", name="float64 (reference)")) < 0.15
                   for n in ("FP32", "BF16, FP32 master weights", "FP16, dynamic loss scaling"))
     and _mean(r, "precision", name="FP8 (E4M3 / E5M2), no scaling") > 2.85),
    ("gradQuant", "2-bit gradients end worse than 8-bit ones",
     lambda r: _mean(r, "gradQuant", bits=2) > _mean(r, "gradQuant", bits=8) + 0.2),
]


def _committed() -> dict:
    return json.loads(RESULTS_JSON.read_text())


def _sl(r, name):
    return next(x for x in r["slowLinks"]["rows"] if x["name"] == name)


def _sl_lr(r, name, lr):
    return next(p for p in _sl(r, name)["perLr"] if p["lr"] == lr)


# brief 29C's claims (chapters 11-16)
CLAIMS += [
    ("systems", "every held-out published run is predicted within 20% and at or below its published throughput",
     lambda r: all(-0.2 < x["error"] <= 1e-9 for x in r["systems"]["validation"]["rows"] if not x["fitted"])),
    ("systems", "the simulated best checkpoint interval is within a factor of 2 of Young's",
     lambda r: 0.5 < r["systems"]["reliability"]["bestTau"] / r["systems"]["reliability"]["young"] < 2),
    ("pipeline", "the simulated bubbles equal the closed forms",
     lambda r: r["pipeline"]["maxDiff"] < 1e-12),
    ("slowLinks", "dense with 4 workers x 4 windows (lr 0.01, seed 1) is the default run, to 1e-6",
     lambda r: abs(_sl_lr(r, "dense", 0.01)["finals"][0] - _committed()["defaultRun"]["evals"][-1]["val"]) < 1e-6),
    ("slowLinks", "the compressors send >= 4x fewer wire bytes than dense, DiLoCo H=10 10x fewer",
     lambda r: all(_sl(r, n)["bytes"]["denseWire"] / _sl(r, n)["bytes"]["wire"] >= 4 for n in ("topk", "demo"))
     and abs(_sl(r, "diloco H=10")["bytes"]["denseWire"] / _sl(r, "diloco H=10")["bytes"]["wire"] - 10) < 1e-9),
    ("slowLinks", "top-k and DeMo reach 2.6 in fewer mean steps than dense and end below dense + 0.1",
     lambda r: all(_sl(r, n)["meanStepsToTarget"] is not None
                   and _sl(r, n)["meanStepsToTarget"] < _sl(r, "dense")["meanStepsToTarget"]
                   and _sl(r, n)["mean"] < _sl(r, "dense")["mean"] + 0.1 for n in ("topk", "demo"))),
    ("slowLinks", "DiLoCo ends above dense by > 0.15 at both H; top-k at lr 3.0 diverges in every seed",
     lambda r: all(_sl(r, n)["mean"] > _sl(r, "dense")["mean"] + 0.15 for n in ("diloco H=10", "diloco H=50"))
     and _sl_lr(r, "topk", 3.0)["diverged"] == 3),
    ("isoflop", "the smallest budget has no interior minimum, the other three do, N_opt grows with the budget, 0.15 < a < 0.8",
     lambda r: not r["isoflop"]["rows"][0]["interior"] and all(x["interior"] for x in r["isoflop"]["rows"][1:])
     and all(a["nOpt"] < b["nOpt"] for a, b in zip(r["isoflop"]["rows"][1:], r["isoflop"]["rows"][2:]))
     and 0.15 < r["isoflop"]["a"] < 0.8),
    ("monitor", "the monitored run is the default run (final loss to 1e-9 relative); the update ratio falls > 10x",
     lambda r: abs(r["monitor"]["final"] / _committed()["defaultRun"]["evals"][-1]["val"] - 1) < 1e-9
     and r["monitor"]["log"]["updateRatio"][-1] < 0.1 * max(r["monitor"]["log"]["updateRatio"])),
    ("resume", "a full resume is bit-identical; a weights-only one ends within the spread over seeds",
     lambda r: r["resume"]["bitIdentical"]
     and abs(r["resume"]["weights"]["mean"] - r["resume"]["uninterrupted"]["mean"])
     < max(r["resume"]["weights"]["sd"], r["resume"]["uninterrupted"]["sd"])),
]


def check_claims(r: dict) -> list[str]:
    """The CLAIMS whose experiment is in r that do not hold."""
    bad = []
    for key, text, ok in CLAIMS:
        if key in r:
            try:
                good = bool(ok(r))
            except (TypeError, KeyError, IndexError):
                good = False
            print(f"  claim {'OK  ' if good else 'FAIL'} {key}: {text}")
            if not good:
                bad.append(f"claim does not hold: {key}: {text}")
    return bad


def rule_for(path: str):
    for rx, rule in RULES:
        if re.search(rx, path):
            return rx, rule
    return "default", ("rel", 1e-9)


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def compare(a, b, path="", problems=None, stats=None):
    """Numbers by RULES (with a 1e-12 absolute floor), the rest exactly.
    `stats[rule] = largest difference seen`, printed by --check."""
    if problems is None:
        problems = []
    if rule_for(path)[1][0] == "skip" and path.split(".")[-1].split("[")[0] not in ("checks",):
        # chaotic values (RULES): not compared at all, even in shape
        if stats is not None and _num(a) and _num(b):
            name = rule_for(path)[0]
            stats[name] = max(stats.get(name, 0.0), abs(a - b))
        return problems
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            problems.append(f"{path}: keys differ {sorted(set(a) ^ set(b))}")
        for k in a:
            if k in b and k not in VOLATILE:
                compare(a[k], b[k], f"{path}.{k}", problems, stats)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            problems.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]", problems, stats)
    elif (isinstance(a, float) or isinstance(b, float)
          or (_num(a) and _num(b) and rule_for(path)[0] != "default")
          or ((a is None) != (b is None) and rule_for(path)[1][0] == "skip")):
        name, (kind, tol) = rule_for(path)
        if a is None or b is None:
            if kind != "skip":
                problems.append(f"{path}: {a} != {b}")
            return problems
        d = abs(a - b)
        if stats is not None:
            stats[name] = max(stats.get(name, 0.0), d if kind == "abs" else d / max(1e-300, max(abs(a), abs(b))))
        if kind == "skip":
            return problems
        limit = 1e-12 + (tol if kind == "abs" else tol * max(abs(a), abs(b)))
        if d > limit:
            problems.append(f"{path}: {a} != {b}")
    elif a != b:
        problems.append(f"{path}: {a!r} != {b!r}")
    return problems


def build(ids, text, groups, check: bool) -> tuple[dict, dict]:
    """The output files of the groups asked for. results.json's payload
    holds only those groups' keys (main merges or compares them).

    With `check`, what is derived from a long, chaotic run is rebuilt from
    the committed files instead of this machine's recomputation, so it is
    checked exactly whatever the run's drift: chapters 3, 4 and 9's
    fixtures from the committed default-run weights, chapter 6's from the
    committed schedule runs, chapter 8's from the committed snapshots."""
    files: dict = {}
    checks: dict = {}
    res: dict = {}
    committed = json.loads(RESULTS_JSON.read_text()) if RESULTS_JSON.exists() else {}
    trained = None
    if "base" in groups:
        grads_fix, checks_g = gradients_fixture(ids)
        opt_fix, checks_o = optimizers_fixture(ids)
        checks_o["rounding"] = C.check_rounding()
        bpe = bpe_fixture(text)
        files.update({
            FIX / "gradients.json": grads_fix,
            FIX / "optimizers.json": opt_fix,
            FIX / "schedules.json": schedules_fixture(),
            FIX / "bpe.json": bpe,
            FIX / "train_run.json": train_fixture(ids),
        })
        for name, payload in anim.fixtures(ids, text).items():
            files[FIX / name] = payload
        res = results(ids, checks_g, checks_o, bpe)
        trained = res.pop("defaultRunParams")
        files[CHECKPOINT] = {
            "note": "Part 1's model after the default training run (results.md), trained by "
                    "scripts/training/make_fixtures.py; chapter 1 trains the same run in the browser.",
            "config": PART1, "steps": DEFAULT_TRAIN["steps"],
            "params": {k: tolist(v) for k, v in trained.items()},
        }
        checks = {"gradients": checks_g, "optimizers": checks_o}
    if check or trained is None:
        ck = json.loads(CHECKPOINT.read_text())
        trained = {k: np.array(v, dtype=np.float64) for k, v in ck["params"].items()}
    if "base" in groups:
        for name, payload in anim.window_fixtures(trained).items():
            files[FIX / name] = payload
    rb, snap = results_b(ids, groups)
    res.update(rb)
    rc = RC.results_c(ids, groups, trained)
    res.update(rc)
    files[RESULTS_JSON] = res
    if snap is not None:
        files[SNAPSHOTS] = snapshot_file(snap)
    # chapters 6-10's animation fixtures
    if "ch6a" in groups:
        sched = committed["schedules"] if check else rb["schedules"]
        files[FIX / "anim_schedules.json"] = {"states": AB.schedule_states(sched)}
    if "ch7a" in groups:
        files[FIX / "gradients_post.json"] = post_ln_fixture(ids)
        files[FIX / "anim_depth.json"] = {"depth": X.DEPTH, "text": X.DEPTH_TEXT,
                                         "stds": {str(s): AB.depth_states(s) for s in X.DEPTH_STDS}}
    if "ch8" in groups:
        payload = json.loads(SNAPSHOTS.read_text()) if check else jsonable(files[SNAPSHOTS])
        run = committed["attnRun"] if check else rb["attnRun"]
        sn = load_snapshots(payload)
        sn["evals"] = run["evals"]
        files[FIX / "anim_attention.json"] = {"text": AB.ATTN_TEXT, "block": AB.ATTN_BLOCK,
                                             "head": AB.ATTN_HEAD, "states": AB.attention_states(sn)}
    if "ch10" in groups:
        files[FIX / "rounding.json"] = rounding_fixture()
        res["gradRounding"] = [{k: st[k] for k in ("fmt", "scaleExp", "nonzero", "zero", "overflow", "meanRelErr")}
                               for st in AB.rounding_states(trained)]
        files[FIX / "anim_rounding.json"] = {"text": AB.GRAD_TEXT, "lo": AB.HIST_LO, "hi": AB.HIST_HI,
                                            "states": AB.rounding_states(trained)}
        files[FIX / "anim_memory.json"] = {"recipe": AB.MIXED, "cases": [
            {"arch": a, "recompute": r, "states": AB.memory_states(a, r)}
            for a in (AB.MEM.LLAMA_7B, AB.MEM.PART1_MEM) for r in ("none", "selective", "full")]}
    # chapters 11-16 (brief 29C)
    if "sys" in groups:
        files[FIX / "systems.json"] = RC.systems_fixture(rc["systems"])
        files[FIX / "anim_ring.json"] = AC.ring_fixture(trained, ids)
        files[FIX / "anim_pipeline.json"] = AC.pipeline_fixture()
        files[FIX / "anim_goodput.json"] = AC.goodput_fixture()
    if "ch13" in groups:
        files[FIX / "slowlinks.json"] = RC.slowlinks_fixture(ids)
        files[FIX / "anim_demo.json"] = {"tensor": SL.ANIM_TENSOR, "seed": SL.ANIM_SEED,
                                         "states": SL.sync_states(trained, ids)}
    if "ch14" in groups:
        iso = committed["isoflop"] if check else rc["isoflop"]
        files[FIX / "anim_isoflop.json"] = {"states": AC.isoflop_states(iso)}
    return files, checks


def all_ok(checks: dict) -> list[str]:
    bad = []
    for group, items in checks.items():
        for k, v in items.items():
            if not v.get("ok"):
                bad.append(f"{group}.{k}")
    return bad


def render_committed_md() -> str:
    import results_md

    return results_md.render(json.loads(RESULTS_JSON.read_text()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--groups", default=",".join(ALL_GROUPS),
                    help=f"comma-separated subset of {ALL_GROUPS}")
    args = ap.parse_args()
    groups = [g for g in args.groups.split(",") if g]
    unknown = set(groups) - set(ALL_GROUPS)
    if unknown:
        sys.exit(f"unknown groups {sorted(unknown)}; choose from {ALL_GROUPS}")
    torch.set_default_dtype(torch.float64)
    text = D.load_corpus()
    ids = M.encode_chars(text)
    t0 = time.time()
    files, checks = build(ids, text, groups, args.check)
    print(f"groups {groups}: computed in {time.time() - t0:.0f} s")
    bad = all_ok(checks)
    import results_md  # noqa: E402

    if bad:
        sys.exit(f"correctness checks FAILED: {bad}")
    if args.check:
        problems: list[str] = []
        stats: dict[str, float] = {}
        for path, payload in files.items():
            committed = json.loads(path.read_text())
            if path == RESULTS_JSON:
                missing = set(payload) - set(committed)
                if missing:
                    problems.append(f"results.json lacks {sorted(missing)}")
                committed = {k: committed[k] for k in payload if k in committed}
            compare(jsonable(payload), committed, str(path.relative_to(ROOT)), problems, stats)
        print("largest difference from the committed files, by rule (abs, or rel for 'rel' rules):")
        for rx, (kind, tol) in RULES + [("default", ("rel", 1e-9))]:
            if rx in stats:
                print(f"  {stats[rx]:.3e}  ({kind} {tol})  {rx}")
        print("the chapters' claims, on the recomputed numbers:")
        problems += check_claims(jsonable(files[RESULTS_JSON]))
        committed_md = render_committed_md()
        if RESULTS_MD.read_text() != committed_md:
            problems.append("results.md is not the rendering of the committed results.json")
        if problems:
            print("\n".join(problems[:20]))
            by: dict[str, int] = {}
            for pr in problems:
                k = re.sub(r"\[\d+\]", "[]", pr.split(":")[0])
                by[k] = by.get(k, 0) + 1
            print("\n".join(f"{n:6d}  {k}" for k, n in sorted(by.items(), key=lambda kv: -kv[1])))
            sys.exit(f"{len(problems)} differences from the committed files")
        print(f"check OK: {len(files)} files agree; all correctness checks pass")
        return
    # write: merge the groups computed into the committed results.json
    res = files.pop(RESULTS_JSON)
    if set(groups) != set(ALL_GROUPS) and RESULTS_JSON.exists():
        merged = json.loads(RESULTS_JSON.read_text())
        merged.update(jsonable(res))
        res = merged
    files[RESULTS_JSON] = res
    bad = check_claims(jsonable(res))
    if bad:
        print("WARNING: " + "; ".join(bad))
    for path, payload in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dump(payload))
        print(f"wrote {path.relative_to(ROOT)}")
    RESULTS_MD.write_text(results_md.render(jsonable(res)))
    print(f"wrote {RESULTS_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
