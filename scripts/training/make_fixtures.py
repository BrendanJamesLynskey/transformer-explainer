"""
make_fixtures.py — Part 2's (Training) reference runs: the correctness
checks, the fixtures the TypeScript trainer must match, and the measured
results every number on the /training pages comes from.

    .venv-fixtures/bin/python scripts/training/make_fixtures.py          # write everything
    .venv-fixtures/bin/python scripts/training/make_fixtures.py --check  # CI: recompute, compare

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
import checks as C  # noqa: E402
import data as D  # noqa: E402
import model as M  # noqa: E402
import optim as O  # noqa: E402

ROOT = HERE.parents[1]
FIX = ROOT / "tests" / "unit" / "fixtures" / "training"
RESULTS_JSON = ROOT / "src" / "data" / "training" / "results.json"
RESULTS_MD = ROOT / "results.md"
CHECKPOINT = ROOT / "public" / "training" / "default-run-weights.json"

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
    # moved single runs' final losses by up to 0.12 (measured in CI,
    # 2026-10-12), so this is a sanity check, and the pages quote the mean
    # and spread over three seeds, not single runs
    (r"optimizerCompare\[\d+\]\.", ("abs", 0.25)),
    (r"results\.json\.(defaultRun|derived)\.", ("rel", 1e-6)),
    (r"default-run-weights\.json\.params\.", ("abs", 1e-6)),
]


def rule_for(path: str):
    for rx, rule in RULES:
        if re.search(rx, path):
            return rx, rule
    return "default", ("rel", 1e-9)


def compare(a, b, path="", problems=None, stats=None):
    """Numbers by RULES (with a 1e-12 absolute floor), the rest exactly.
    `stats[rule] = largest difference seen`, printed by --check."""
    if problems is None:
        problems = []
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
    elif isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            problems.append(f"{path}: {a} != {b}")
            return problems
        name, (kind, tol) = rule_for(path)
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


def build(ids, text, window_params=None) -> tuple[dict, dict]:
    """Every output file. `window_params`: the weights chapters 3 and 4 use
    (default: this run's final weights; --check passes the committed ones,
    so those fixtures are checked exactly whatever the long run's drift)."""
    grads_fix, checks_g = gradients_fixture(ids)
    opt_fix, checks_o = optimizers_fixture(ids)
    bpe = bpe_fixture(text)
    files = {
        FIX / "gradients.json": grads_fix,
        FIX / "optimizers.json": opt_fix,
        FIX / "schedules.json": schedules_fixture(),
        FIX / "bpe.json": bpe,
        FIX / "train_run.json": train_fixture(ids),
    }
    for name, payload in anim.fixtures(ids, text).items():
        files[FIX / name] = payload
    res = results(ids, checks_g, checks_o, bpe)
    trained = res.pop("defaultRunParams")
    files[RESULTS_JSON] = res
    files[CHECKPOINT] = {
        "note": "Part 1's model after the default training run (results.md), trained by "
                "scripts/training/make_fixtures.py; chapter 1 trains the same run in the browser.",
        "config": PART1, "steps": DEFAULT_TRAIN["steps"],
        "params": {k: tolist(v) for k, v in trained.items()},
    }
    for name, payload in anim.window_fixtures(trained if window_params is None else window_params).items():
        files[FIX / name] = payload
    return files, {"gradients": checks_g, "optimizers": checks_o}


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
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    text = D.load_corpus()
    ids = M.encode_chars(text)
    window_params = None
    if args.check:
        ck = json.loads(CHECKPOINT.read_text())
        window_params = {k: np.array(v, dtype=np.float64) for k, v in ck["params"].items()}
    files, checks = build(ids, text, window_params)
    bad = all_ok(checks)
    import results_md  # noqa: E402

    md = results_md.render(files[RESULTS_JSON])
    if bad:
        sys.exit(f"correctness checks FAILED: {bad}")
    if args.check:
        problems: list[str] = []
        stats: dict[str, float] = {}
        for path, payload in files.items():
            committed = json.loads(path.read_text())
            compare(jsonable(payload), committed, str(path.relative_to(ROOT)), problems, stats)
        print("largest difference from the committed files, by rule (abs, or rel for 'rel' rules):")
        for rx, (kind, tol) in RULES + [("default", ("rel", 1e-9))]:
            if rx in stats:
                print(f"  {stats[rx]:.3e}  ({kind} {tol})  {rx}")
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
    for path, payload in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dump(payload))
        print(f"wrote {path.relative_to(ROOT)}")
    RESULTS_MD.write_text(md)
    print(f"wrote {RESULTS_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
