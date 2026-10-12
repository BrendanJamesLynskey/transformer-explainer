"""
results_md.py — renders src/data/training/results.json as results.md.

Every number on the /training pages is read from results.json (written by
make_fixtures.py); results.md shows the same numbers for a reader of the
repository. `make_fixtures.py --check` also requires that results.md is
exactly this rendering of the committed results.json.
"""

from __future__ import annotations


def f(x, d=4):
    return f"{x:.{d}f}"


def e(x):
    return f"{x:.1e}"


def render(r: dict) -> str:
    c, m, run = r["corpus"], r["model"], r["defaultRun"]
    cg, co = r["checks"]["gradients"], r["checks"]["optimizers"]
    env = r["environment"]
    cfg = run["config"]
    L = []
    w = L.append
    w("# Training results (Part 2)")
    w("")
    w("Every number on the `/training` pages comes from this file's source,")
    w("`src/data/training/results.json`, written by")
    w("`scripts/training/make_fixtures.py` (NumPy float64 reference, seeded).")
    w("Regenerate with `.venv-fixtures/bin/python scripts/training/make_fixtures.py`;")
    w("CI runs it with `--check` and fails on any difference.")
    w("")
    w(f"Recorded with Python {env['python']}, NumPy {env['numpy']}, PyTorch {env['torch']} "
      f"(CPU), {env['machine']}.")
    w("")
    w("## 1. Corpus")
    w("")
    w("| | |")
    w("|---|---|")
    w(f"| Text | {c['title']} |")
    w(f"| Source | {c['source']}; see `public/training/CORPUS.md` |")
    w(f"| Characters | {c['chars']:,} (train {c['trainChars']:,}, validation {c['valChars']:,}) |")
    w(f"| SHA-256 of `public/training/corpus.txt` | `{c['sha256']}` |")
    w(f"| Unigram entropy of the training characters | {f(c['unigramEntropy'])} nats |")
    w(f"| BPE tokens after {c['bpeMerges']} merges | {c['bpeTokens']:,} "
      f"({f(c['chars'] / c['bpeTokens'], 3)} characters per token) |")
    w("")
    w("## 2. Model")
    w("")
    k = m["config"]
    w(f"Part 1's model: seq_len {k['seq_len']}, d_model {k['d_model']}, {k['n_heads']} heads, "
      f"d_ff {k['d_ff']}, {k['n_blocks']} blocks, vocabulary {k['vocab_size']}, seed {k['seed']}; "
      f"{m['params']:,} parameters. A uniform guess scores ln {k['vocab_size']} = {f(m['lnV'])}.")
    w("")
    w("## 3. Correctness checks")
    w("")
    w("| Check | Measured | Tolerance |")
    w("|---|---|---|")
    w(f"| Initialiser = `scripts/reference.py` port | max abs err {cg['init']['max_abs_err']} | exact |")
    for key, label in (("autograd_init", "at initialisation"), ("autograd_perturbed", "perturbed weights")):
        a = cg[key]
        w(f"| Backward = PyTorch autograd, {label} | max err / (1 + max abs grad) {e(a['max_scaled_err'])}; "
          f"loss diff {e(a['loss_err'])} | 1e-12 |")
    for key, label in (("fd_init", "at initialisation"), ("fd_perturbed", "perturbed weights")):
        a = cg[key]
        w(f"| Backward = central differences (h = 1e-6), {label}, {a['entries']} entries | "
          f"max abs err {e(a['max_abs_err'])} | 1e-8 + 1e-6 abs grad |")
    for key in ("sgd", "momentum", "adam", "adam_l2", "adamw"):
        a = co[key]
        w(f"| `{key}` = torch.optim over {a['steps']} steps | max abs err {e(a['max_abs_err'])} | 1e-12 |")
    a = co["cosine_schedule"]
    w(f"| Cosine schedule = CosineAnnealingLR | max abs err {e(a['max_abs_err'])} | 1e-15 |")
    w("")
    w("## 4. The default run (chapter 1)")
    w("")
    o, sc = cfg["optimizer"], cfg["schedule"]
    w(f"{cfg['steps']:,} steps of {cfg['batch']} windows of {k['seq_len']} characters; "
      f"{o['kind']} (beta1 {o['beta1']}, beta2 {o['beta2']}, weight decay {o['weight_decay']}); "
      f"learning rate {sc['lr']}, {sc['warmup']} warmup steps, cosine to {sc['min_ratio']} of it; "
      f"clip at global norm {cfg['clip']}; data seed {cfg['data_seed']}. "
      f"{f(run['seconds'], 1)} s on the machine above (NumPy).")
    w("")
    w("| Step | Validation loss | Training loss (this step's batch) |")
    w("|---:|---:|---:|")
    by_step = dict(zip(run["steps"], run["loss"]))
    for ev in run["evals"]:
        tl = by_step.get(ev["step"])
        w(f"| {ev['step']:,} | {f(ev['val'])} | {f(tl) if tl is not None else '–'} |")
    w("")
    w(f"Samples ({run['samples'].get('prompt', 'alice ')!r} + 120 characters, temperature 0.8, seed 7):")
    w("")
    w(f"- at initialisation: `{run['samples']['init']}`")
    w(f"- after {cfg['steps']:,} steps: `{run['samples']['trained']}`")
    w("")
    w("## 5. Optimisers on the real model (chapter 5)")
    w("")
    w("Each optimiser at three constant learning rates, 1,500 steps of 16 windows, clip 1.0,")
    w("beta2 0.99 for Adam and AdamW, AdamW weight decay 0.01; three data seeds (1, 2, 3) per")
    w("setting. Mean ± standard deviation of the final validation loss over the seeds.")
    w("Single runs at the higher learning rates are chaotic: recomputed on another CPU, a run's")
    w("final loss moved by up to 0.12 (last-bit differences in the matrix products, amplified),")
    w("so only differences well beyond the spread mean anything.")
    w("")
    w("| Optimiser | Learning rate | Final validation loss (mean ± sd) | Seeds 1, 2, 3 | Best for this optimiser |")
    w("|---|---:|---:|---|:---:|")
    best = {}
    for row in r["optimizerCompare"]:
        if row["kind"] not in best or row["final"] < best[row["kind"]]["final"]:
            best[row["kind"]] = row
    for row in r["optimizerCompare"]:
        mark = "yes" if best[row["kind"]] is row else ""
        seeds = ", ".join(f(x["final"], 3) for x in row["runs"])
        w(f"| {row['kind']} | {row['lr']} | {f(row['final'], 3)} ± {f(row['sd'], 3)} | {seeds} | {mark} |")
    w("")
    return "\n".join(L)
