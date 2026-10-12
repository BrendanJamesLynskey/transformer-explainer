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
    if "rounding" in co:
        a = co["rounding"]
        n = sum(v["values"] for v in a["formats"].values())
        w(f"| Vectorised rounding (`precision.round_array`) = the Numerics site's scalar code, "
          f"{', '.join(a['formats'])}, {n:,} values | mismatches {a['max_mismatches']} | exact |")
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
    w("final loss moved by about 0.1 (last-bit differences in the matrix products, amplified),")
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
    render_b(r, w)
    render_c(r, w)
    return "\n".join(L)


def ms(row, d=3) -> str:
    """mean ± sd over seeds, with a note of diverged runs."""
    if row.get("mean") is None:
        return "diverged" if row.get("diverged") else "–"
    sd = f" ± {f(row['sd'], d)}" if row.get("sd") is not None else ""
    div = f" ({row['diverged']} of {len(row['finals'])} diverged)" if row.get("diverged") else ""
    return f"{f(row['mean'], d)}{sd}{div}"


def finals(row, d=3) -> str:
    return ", ".join("diverged" if x is None else f(x, d) for x in row["finals"])


def render_b(r: dict, w) -> None:
    """Brief 29B's sections (chapters 6-10), each only if its key is present."""
    if "lrSweep" in r:
        w("## Chapters 6-10: how to read these numbers")
        w("")
        w("Every setting below ran with three data seeds; the pages quote the mean and the standard")
        w("deviation. Most of these runs are chaotic: whether and when a run leaves the plateau of")
        w("chapter 1 depends on the last bits of its arithmetic, so recomputing them on another CPU")
        w("moved single seeds' final losses by up to about 0.35, step counts by up to a factor of 2,")
        w("and made one run at learning rate 1.0 diverge. CI recomputes every group, sanity-checks")
        w("the means, and re-checks each claim the chapters make (`CLAIMS` in `make_fixtures.py`);")
        w("the deterministic parts (scale through depth, the coordinate check, the noise scale,")
        w("memory, rounding) are checked to 1e-7 or better.")
        w("")
        x = r["lrSweep"]
        w("## 6. Learning rate, schedules and batch size (chapter 6)")
        w("")
        w(f"### 6a. Learning-rate sweep")
        w("")
        w(f"AdamW (beta2 0.99, weight decay 0.01), warmup 50 steps then cosine to 0.1 of the peak,")
        w(f"{x['steps']:,} steps of 16 windows, clip 1.0, three data seeds per peak learning rate.")
        w("A run counts as diverged if its loss stops being finite or exceeds 12 nats.")
        w("")
        w("| Peak learning rate | Final validation loss (mean ± sd) | Seeds 1, 2, 3 |")
        w("|---:|---:|---|")
        for row in x["rows"]:
            w(f"| {row['lr']} | {ms(row)} | {finals(row)} |")
        w("")
        w(f"Best peak learning rate (lowest mean): {x['bestLr']}.")
        w("")
    if "schedules" in r:
        w("### 6b. Schedules")
        w("")
        w("AdamW at peak learning rate 0.01, 2,000 steps, three seeds; validation every 50 steps.")
        w("")
        w("| Schedule | Final validation loss (mean ± sd) | Seeds 1, 2, 3 |")
        w("|---|---:|---|")
        for row in r["schedules"]["rows"]:
            w(f"| {row['name']} | {ms(row)} | {finals(row)} |")
        w("")
    if "batchSize" in r:
        x = r["batchSize"]
        w("### 6c. Batch size: steps to a target loss")
        w("")
        w(f"Steps until the validation loss first reaches {x['target']} (checked every 25 steps), at each")
        w("batch size (windows of 8 characters), AdamW at a constant learning rate after a 50-step warmup;")
        w("the better of two learning rates (lower mean steps), three seeds.")
        w("")
        w("| Batch (windows) | Learning rate | Steps (mean ± sd) | Seeds | Windows processed |")
        w("|---:|---:|---:|---|---:|")
        for row in x["rows"]:
            b = row["best"]
            if b is None:
                w(f"| {row['batch']} | – | not reached | – | – |")
                continue
            seeds = ", ".join(str(v) for v in b["steps"])
            w(f"| {row['batch']} | {b['lr']} | {f(b['mean'], 0)} ± {f(b['sd'], 0)} | {seeds} | {row['windows']:,.0f} |")
        fit = x["fit"]
        w("")
        w(f"Fit of S = S_min (1 + B_crit / B) (McCandlish et al. 2018): B_crit = {f(fit['bCrit'], 1)} windows, "
          f"S_min = {f(fit['sMin'], 0)} steps, E_min = {f(fit['eMin'], 0)} windows "
          f"(RMS error {f(fit['rmsLogErr'], 3)} in log steps).")
        w("")
    if "noiseScale" in r:
        x = r["noiseScale"]
        w("### 6d. Gradient noise scale along the default run")
        w("")
        w(f"B_simple = tr(Sigma) / |G|^2 from {x['windows']} single-window gradients (McCandlish et al. 2018, eq. 2.9).")
        w("")
        w("| Step | tr(Sigma) | abs(G)^2 | B_simple (windows) |")
        w("|---:|---:|---:|---:|")
        for p in x["points"]:
            b = "–" if p["bSimple"] is None else f(p["bSimple"], 1)
            w(f"| {p['step']:,} | {e(p['trSigma'])} | {e(p['gNormSq'])} | {b} |")
        w("")
    if "depthScale" in r:
        w("## 7. Initialisation, normalisation and muP (chapter 7)")
        w("")
        w("### 7a. Scale through depth at initialisation")
        w("")
        w("An 8-block version of the model on \"alice was\": RMS of the residual stream entering each")
        w("block (and leaving the last), and the norm of each block's weight-matrix gradients.")
        w("")
        for by in r["depthScale"]:
            for norm in ("pre", "post"):
                d = by[norm]
                w(f"- std {by['std']}, {norm}-LN: residual RMS "
                  + ", ".join(f(v, 2) for v in d["residualRms"]) + "; gradient norms "
                  + ", ".join(f"{v:.3g}" for v in d["gradNorm"]))
        w("")
    if "prePost" in r:
        x = r["prePost"]
        w("### 7b. Pre-LN vs post-LN")
        w("")
        w(f"{x['depth']} blocks, AdamW at peak learning rate {x['lr']}, cosine, {x['steps']:,} steps, three seeds.")
        w("")
        w("| Norm placement | Warmup steps | Final validation loss (mean ± sd) | Seeds 1, 2, 3 |")
        w("|---|---:|---:|---|")
        for row in x["rows"]:
            w(f"| {row['norm']}-LN | {row['warmup']} | {ms(row)} | {finals(row)} |")
        w("")
    if "normKind" in r:
        w("### 7c. LayerNorm vs RMSNorm")
        w("")
        w("The default run (2 blocks) with each, three seeds.")
        w("")
        w("| Norm | Parameters | Final validation loss (mean ± sd) | Seeds 1, 2, 3 |")
        w("|---|---:|---:|---|")
        for row in r["normKind"]["rows"]:
            name = "LayerNorm" if row["kind"] == "layer" else "RMSNorm"
            w(f"| {name} | {row['params']:,} | {ms(row)} | {finals(row)} |")
        w("")
    if "coordCheck" in r:
        x = r["coordCheck"]
        w("### 7d. muP coordinate check")
        w("")
        w(f"RMS change of the logits and of the last residual stream after {x['steps']} AdamW steps at")
        w(f"learning rate {x['lr']} (constant, no clipping), on 32 fixed windows.")
        w("")
        w("| Width | SP: logits | SP: residual | muP: logits | muP: residual |")
        w("|---:|---:|---:|---:|---:|")
        for a, b in zip(x["sp"], x["mup"]):
            w(f"| {a['width']} | {e(a['logitsDelta'])} | {e(a['residualDelta'])} | "
              f"{e(b['logitsDelta'])} | {e(b['residualDelta'])} |")
        w("")
    if "mupTransfer" in r:
        x = r["mupTransfer"]
        w("### 7e. Learning-rate transfer across width")
        w("")
        w(f"Final validation loss after {x['steps']:,} steps (warmup 50 + cosine), three seeds;")
        w("muP's base width is 16, so width 16 is the same model either way.")
        w("")
        w("| Parameterisation | Width | Parameters | Peak learning rate | Final validation loss (mean ± sd) | Seeds 1, 2, 3 |")
        w("|---|---:|---:|---:|---:|---|")
        for row in x["rows"]:
            w(f"| {'muP' if row['param'] == 'mup' else 'standard'} | {row['width']} | {row['params']:,} | "
              f"{row['lr']} | {ms(row)} | {finals(row)} |")
        w("")
    if "badBatch" in r:
        x = r["badBatch"]
        w("## 8. Stability (chapter 8)")
        w("")
        w(f"### 8a. One bad batch")
        w("")
        w(f"At step {x['at']:,} every window of the batch is the character '{x['char']}' repeated. "
          f"{x['steps']:,} steps, constant learning rate, three seeds.")
        w("")
        w("| Run | Learning rate | Validation loss before | Just after | At the end |")
        w("|---|---:|---|---|---|")
        for row in x["rows"]:
            fm = lambda v: "diverged" if v is None else (f"{v:.3g}" if v > 100 else f(v, 3))  # noqa: E731
            w(f"| {row['name']} | {row['lr']} | {', '.join(fm(v) for v in row['before'])} | "
              f"{', '.join(fm(v) for v in row['after'])} | {', '.join(fm(v) for v in row['final'])} |")
        w("")
    if "attnGrowth" in r:
        x = r["attnGrowth"]
        w("### 8b. Attention-logit growth at a high learning rate")
        w("")
        w("AdamW, warmup 50 + cosine, 2,000 steps, clip 1.0, three seeds. Largest attention score and")
        w("mean attention entropy on the validation windows at the end (mean over seeds).")
        w("")
        w("| Peak learning rate | Run | Final validation loss (mean ± sd) | Largest score | Entropy (nats) |")
        w("|---:|---|---:|---:|---:|")
        for row in x["rows"]:
            last = row["meanStats"][-1]
            w(f"| {row['lr']} | {row['name']} | {ms(row)} | {last['maxScore']:,.1f} | {f(last['attnEntropy'], 2)} |")
        w("")
    if "precision" in r:
        w("## 9. Mixed precision and quantised gradients (chapter 9)")
        w("")
        w("### 9a. Emulated precision")
        w("")
        w("The default run with weights' compute copies, activations at every matrix product and")
        w("gradients rounded to the formats (`scripts/training/precision.py`; float64 stands in for FP32")
        w("in the elementwise operations and the master weights). Three seeds.")
        w("")
        w("| Recipe | Final validation loss (mean ± sd) | Seeds 1, 2, 3 | Gradient values flushed to zero | Steps skipped |")
        w("|---|---:|---|---:|---:|")
        for row in r["precision"]["rows"]:
            w(f"| {row['name']} | {ms(row)} | {finals(row)} | {100 * row['gradFlushedToZero']:.2f}% | {row['skipped']:.1f} |")
        w("")
    if "gradQuant" in r:
        w("### 9b. Quantised gradients")
        w("")
        w("Every weight gradient quantised to k-bit integers (one scale per tensor, full scale = its")
        w("largest magnitude) before clipping and the optimiser step; three seeds.")
        w("")
        w("| Bits | Levels | Final validation loss (mean ± sd) | Seeds 1, 2, 3 |")
        w("|---:|---:|---:|---|")
        for row in r["gradQuant"]["rows"]:
            w(f"| {row['bits']} | {row['levels']} | {ms(row)} | {finals(row)} |")
        w("")
    if "memory" in r:
        x = r["memory"]
        L7 = x["llama"]
        a = L7["arch"]
        w("## 10. Memory (chapter 10)")
        w("")
        w(f"{a['name']} (Touvron et al. 2023, Table 2): {L7['mixed']['none']['params']:,} parameters by the")
        w(f"calculator; sequence {a['s']:,}, micro-batch {a['b']}. Bytes:")
        w("")
        w("| Recipe | Weights | Gradients | Optimiser | Master | Activations | Logits | Peak |")
        w("|---|---:|---:|---:|---:|---:|---:|---:|")
        gb = lambda v: f"{v / 1e9:.1f} GB"  # noqa: E731
        rows = [("mixed precision Adam, no recomputation", L7["mixed"]["none"]),
                ("mixed precision Adam, selective recomputation", L7["mixed"]["selective"]),
                ("mixed precision Adam, full recomputation", L7["mixed"]["full"]),
                ("FP32 Adam, no recomputation", L7["fp32"])]
        for name, b in rows:
            w(f"| {name} | {gb(b['weights'])} | {gb(b['grads'])} | {gb(b['optimizer'])} | {gb(b['master'])} | "
              f"{gb(b['activations'])} | {gb(b['logits'])} | {gb(b['peak'])} |")
        p = x["part1"]
        w("")
        w(f"Part 1's model keeps {', '.join(f(v, 0) for v in p['savedFloatsPerSeqPerBlock'])} values per sequence per block")
        w(f"for its backward pass (counted in the NumPy forward pass); Korthikanti et al.'s formula for the same shape")
        w(f"gives {f(p['korthikantiValuesPerSeqPerBlock'], 0)}.")
        w("")


def ms_time(sec: float) -> str:
    if sec < 1e-3:
        return f"{sec * 1e6:.1f} us"
    if sec < 1:
        return f"{sec * 1e3:.1f} ms"
    return f"{sec:.2f} s"


def render_c(r: dict, w) -> None:
    """Brief 29C's sections (chapters 11-16), each only if its key is present."""
    if "systems" in r:
        x = r["systems"]
        v = x["validation"]
        w("## 11. The training-systems model (chapters 11, 12 and 15)")
        w("")
        w("`scripts/training/systems.py` (ported to `src/lib/training/systems.ts`, matched to its fixtures):")
        w("collectives on an alpha-beta link model, ZeRO and per-device memory, a discrete-event simulation")
        w("of pipeline schedules, a step-time model, and failures and checkpoints. Nothing here trains; it")
        w("is arithmetic and seeded simulation.")
        w("")
        w("### 11a. Validation against published runs")
        w("")
        w(f"One constant is fitted: eta = {f(v['eta'], 3)}, the fraction of peak the matrix products reach,")
        w("on Megatron-LM's 1.7B run (no tensor or pipeline parallelism). Every other row is a prediction.")
        w("Published numbers: Narayanan et al. 2021 (arXiv:2104.04473) Table 1 (A100, 312 TFLOP/s peak,")
        w("hardware FLOP/s with full recomputation by their eq. 3) and Grattafiori et al. 2024 (Llama 3,")
        w("arXiv:2407.21783) Table 4 (H100, BF16 MFU). Assumptions where the papers are silent:")
        w("")
        for k, text in v["assumptions"].items():
            w(f"- {k}: {text}")
        w("")
        w("| Run | GPUs | Layout (t, p, d, cp, v) | Micro-batches | Published TFLOP/s per GPU | Model | Error |")
        w("|---|---:|---|---:|---:|---:|---:|")
        for row in v["rows"]:
            lay = row["layout"]
            tag = " (fitted)" if row["fitted"] else ""
            src = "Megatron" if row["source"] == "megatron" else "Llama 3 405B"
            w(f"| {src} {row['label']}{tag} | {row['gpus']:,} | {lay['t']}, {lay['p']}, {lay['d']}, {lay.get('cp', 1)}, "
              f"{lay.get('v', 1)} | {row['microBatches']} | {row['published'] / 1e12:.0f} ({row['publishedPct']}%) | "
              f"{row['predicted'] / 1e12:.1f} ({row['predictedPct']:.1f}%) | {100 * row['error']:+.1f}% |")
        w("")
        mx = v["maxAbsErrorBySource"]
        w(f"**Validation band:** every held-out row within {100 * v['maxAbsError']:.1f}% of the published")
        w(f"throughput (Megatron rows {100 * mx['megatron']:.1f}%, Llama 3 rows {100 * mx['llama3']:.1f}%; mean absolute")
        w(f"error {100 * v['meanAbsError']:.1f}%). Every prediction is at or below the published number: the model")
        w("overlaps nothing, and one eta cannot follow the matrix products getting more efficient as the")
        w("models widen (Megatron's per-GPU throughput rises from 137 to 163 TFLOP/s; the model's stays near 130).")
        w("")
        w("### 11b. Collectives, ZeRO and data parallelism")
        w("")
        c = x["collectives"]
        w(f"LLaMA 7B's gradient ({c['params']:,} parameters, {c['bytes'] / 1e9:.2f} GB in BF16) on H100s:")
        w("")
        w("| Where | Collective | Time | Bytes each GPU sends |")
        w("|---|---|---:|---:|")
        names = {"allreduce": "ring all-reduce", "tree": "binary-tree all-reduce (not pipelined)",
                 "reducescatter": "reduce-scatter", "allgather": "all-gather"}
        for row in c["rows"]:
            w(f"| {row['where']} | {names[row['kind']]} | {ms_time(row['time'])} | {row['wire'] / 1e9:.2f} GB |")
        w("")
        w("ZeRO (Rajbhandari et al. 2020, arXiv:1910.02054): bytes per GPU for the same model's states")
        w("(2 + 2 + 12 bytes per parameter), and what each GPU sends per step:")
        w("")
        w("| Data-parallel GPUs | Stage | Weights | Gradients | Optimiser | Total | Sent per step |")
        w("|---:|---:|---:|---:|---:|---:|---:|")
        for row in x["zero"]["rows"]:
            w(f"| {row['d']} | {row['stage']} | {row['weights'] / 1e9:.2f} GB | {row['grads'] / 1e9:.2f} GB | "
              f"{row['optimizer'] / 1e9:.2f} GB | {row['total'] / 1e9:.2f} GB | {row['wire'] / 1e9:.2f} GB |")
        d = x["dpExample"]
        w("")
        w(f"One data-parallel step of LLaMA 7B on {d['layout']['d']} H100s, {d['tokens']:,} tokens, eta as fitted:")
        w(f"compute {ms_time(d['compute'])}, gradient all-reduce {ms_time(d['data'])}; step {ms_time(d['time'])} with")
        w(f"nothing overlapped (MFU {100 * d['mfu']:.1f}%), {ms_time(d['overlapped'])} with the all-reduce hidden behind")
        w(f"the backward pass (MFU {100 * d['mfuOverlapped']:.1f}%).")
        w("")
        w("### 11c. Failures and checkpoints (chapter 15)")
        w("")
        rel = x["reliability"]
        ll = rel["llama"]
        w(f"Llama 3 405B: {ll['interruptions']} interruptions in {ll['days']} days, {ll['unexpected']} unexpected")
        w(f"(Grattafiori et al. 2024 §3.3.4), on up to {ll['gpus']:,} GPUs: a failure every {ll['mtbfHours']:.2f} hours")
        w(f"for the job, about one per GPU every {ll['gpuMtbfYears']:.1f} years. Checkpoint time delta = {rel['delta']:.1f} s")
        w(f"({rel['deltaNote']}); restart R = {rel['restart']:.0f} s ({rel['restartNote']}). Goodput over {ll['days']}")
        w(f"simulated days, mean of {len(rel['seeds'])} seeds, against the first-order formula:")
        w("")
        w("| Checkpoint every | Simulated goodput | First-order formula |")
        w("|---:|---:|---:|")
        for row in rel["rows"]:
            w(f"| {row['tau']:,} s | {100 * row['simulated']:.2f}% | {100 * row['formula']:.2f}% |")
        w("")
        w(f"Best simulated interval {rel['bestTau']:,} s; Young's tau* = sqrt(2 delta M) = {rel['young']:.0f} s")
        w("(the grid steps by a factor of 1.41).")
        w("")
        w("### 11d. Stragglers and queueing (chapter 13)")
        w("")
        w("Eight workers, each step 1 + 0.1 x Exp(1) time units (seeded); the mean time per step when")
        w("everyone waits for the slowest, syncing every step or every H steps:")
        w("")
        w("| Sync every | Time per step | Mean worker's step |")
        w("|---:|---:|---:|")
        for row in x["stragglers"]:
            per = row["syncEveryStep"] if row["H"] == 1 else row["syncEveryH"]
            w(f"| {row['H']} | {f(per, 3)} | {f(row['meanWorker'], 3)} |")
        w("")
        w("A sync message's mean wait on a link carrying background transfers (each 1 time unit, Poisson")
        w("arrivals): the M/D/1 formula W = load / (2 (1 - load)) against a seeded simulation:")
        w("")
        w("| Load | Formula | Simulated |")
        w("|---:|---:|---:|")
        for row in x["queue"]:
            w(f"| {row['load']} | {f(row['formula'], 3)} | {f(row['simulated'], 3)} |")
        w("")
    if "pipeline" in r:
        w("## 12. Pipeline schedules (chapter 12)")
        w("")
        w("A discrete-event simulation of every forward and backward on every device (backward = 2 x forward),")
        w("against the closed forms of Narayanan et al. (2021) §2.2: bubble = (p - 1) / m, interleaved (p - 1) / (v m).")
        w("")
        w("| Schedule | Devices p | Micro-batches m | Chunks v | Simulated bubble | Closed form |")
        w("|---|---:|---:|---:|---:|---:|")
        for row in r["pipeline"]["rows"]:
            w(f"| {row['kind']} | {row['p']} | {row['m']} | {row['v']} | {f(row['bubble'])} | {f(row['closedForm'])} |")
        w("")
        w(f"Largest difference: {r['pipeline']['maxDiff']:.1e}.")
        w("")
    if "slowLinks" in r:
        x = r["slowLinks"]
        w("## 13. Training over slow links (chapter 13)")
        w("")
        w(f"Part 1's model, {x['workers']} simulated workers with {x['windowsPerWorker']} windows each per step (the default")
        w(f"run's 16), {x['steps']:,} steps, warmup 50 + cosine to 0.1 of the peak, each worker's gradient clipped")
        w(f"to 1.0, weight decay 0.01 on matrices; three seeds. Compressors keep 1 in {x['keep']}: top-k per tensor")
        w(f"(momentum {x['muTopk']}), DeMo per DCT chunk of up to {x['chunk']} (beta {x['betaDemo']}, sign update);")
        w(f"DiLoCo's outer step is Nesterov momentum (lr {x['outerLr']}, momentum {x['outerMu']}). Each method's")
        w("learning rate is the best mean of three tried. Bytes per worker per step, BF16 values; ring")
        w("collectives (all-reduce for dense and DiLoCo, all-gather for the compressors).")
        w("")
        w(f"| Method | Best lr | Final validation loss (mean ± sd) | Steps to {x['target']} (seeds) | Payload | Wire | Wire vs dense |")
        w("|---|---:|---:|---|---:|---:|---:|")
        for row in x["rows"]:
            b = row["bytes"]
            hits = ", ".join("not reached" if h is None else str(h) for h in row["stepsToTarget"])
            w(f"| {row['name']} | {row['bestLr']} | {ms(row)} | {hits} | {b['payload']:,.0f} B | {b['wire']:,.0f} B | "
              f"{b['denseWire'] / b['wire']:.1f}x less |")
        w("")
        w("Every learning rate tried (mean ± sd over seeds):")
        w("")
        w("| Method | lr | Final validation loss |")
        w("|---|---:|---:|")
        for row in x["rows"]:
            for p_ in row["perLr"]:
                w(f"| {row['name']} | {p_['lr']} | {ms(p_)} |")
        w("")
    if "isoflop" in r:
        x = r["isoflop"]
        w("## 14. A toy IsoFLOP sweep (chapter 14)")
        w("")
        w(f"Part 1's model at widths {', '.join(str(v) for v in x['widths'])} (2 heads, FFN twice the width, 2 blocks),")
        w(f"each trained for exactly the tokens a budget C = 6 N D allows ({x['batch']} windows of {x['seqLen']} per step,")
        w(f"AdamW peak lr {x['lr']}, warmup then cosine to 10% over the run); three seeds. N counts every parameter.")
        w("A parabola in log10 N through each budget's lowest mean and its two neighbours gives N_opt;")
        w("a line through log N_opt against log C gives the exponent a.")
        w("")
        w("| Budget C (FLOPs) | Width | N | Steps | Tokens | Final validation loss (mean ± sd) |")
        w("|---:|---:|---:|---:|---:|---:|")
        for row in x["rows"]:
            for p_ in row["points"]:
                w(f"| {row['budget']:.0e} | {p_['width']} | {p_['params']:,} | {p_['steps']:,} | {p_['tokens']:,} | {ms(p_)} |")
        w("")
        w("| Budget | N_opt | D_opt (tokens) | Minimum inside the sizes tried |")
        w("|---:|---:|---:|---|")
        for row in x["rows"]:
            n_opt = f"{row['nOpt']:,.0f}" if row["nOpt"] else "–"
            d_opt = f"{row['dOpt']:,.0f}" if row["dOpt"] else "–"
            w(f"| {row['budget']:.0e} | {n_opt} | {d_opt} | {'yes' if row['interior'] else 'no'} |")
        w("")
        pub = x["published"]
        w(f"Fitted exponent a = {f(x['a'], 2)} (N_opt ~ C^a; b = 1 - a = {f(x['b'], 2)}), from the budgets with a minimum")
        w(f"inside the sizes tried. Published: Chinchilla {pub['chinchilla1']} / {pub['chinchilla2']} / {pub['chinchilla3']}")
        w(f"(Hoffmann et al. 2022, Table 2, approaches 1-3), Kaplan et al. 2020 {pub['kaplan']}; Llama 3 fitted")
        w(f"alpha = {pub['llama3TokensAlpha']} for the optimal number of tokens.")
        w("")
    if "monitor" in r:
        x = r["monitor"]
        lg = x["log"]
        w("## 15. Running the job (chapter 15)")
        w("")
        w(f"The default run, step for step, as a dashboard sees it (final validation loss {f(x['final'])}, the")
        w("default run's). Every 10th step:")
        w("")
        w("| Step | Training loss | Gradient norm (before clipping) | Learning rate | Update / weight (all) | Update / weight (block 1 FFN W1) |")
        w("|---:|---:|---:|---:|---:|---:|")
        for i in range(0, len(lg["steps"]), 20):
            w(f"| {lg['steps'][i]} | {f(lg['loss'][i], 3)} | {f(lg['gradNorm'][i], 3)} | {lg['lr'][i]:.5f} | "
              f"{lg['updateRatio'][i]:.2e} | {lg['updateRatioW1'][i]:.2e} |")
        w("")
    if "resume" in r:
        x = r["resume"]
        w(f"### Stopping at step {x['stop']:,} and resuming (three seeds)")
        w("")
        w(f"Resuming with the weights, AdamW's state and the data position continues bit for bit: "
          f"**{'yes' if x['bitIdentical'] else 'NO'}** (every seed).")
        w("")
        w(f"| Resumed with | Validation loss 50 steps after resuming | Final validation loss (mean ± sd) |")
        w("|---|---:|---:|")
        for key, label in (("uninterrupted", "never stopped"), ("full", "weights + optimiser + data position"),
                           ("weights", "weights only (fresh AdamW)"), ("freshData", "weights + optimiser, data restarted")):
            row = x[key]
            w(f"| {label} | {f(row['afterResume'], 3)} | {f(row['mean'], 3)} ± {f(row['sd'], 3)} |")
        w("")
    if "sft" in r:
        x = r["sft"]
        p = x["pref"]
        w("## 16. After pretraining (chapter 16)")
        w("")
        w(f"The trained model on a prompt and a response: `{x['prompt']}` + `{x['response']}`. Mean loss on the")
        w(f"response's characters (the SFT loss, prompt masked): {f(x['sftLoss'], 3)} nats; on every character")
        w(f"after the first: {f(x['unmaskedLoss'], 3)}. Log probability of two continuations of `{p['prompt']}`:")
        w(f"`{p['chosen']}` {f(p['chosenLogp'], 2)}, `{p['rejected']}` {f(p['rejectedLogp'], 2)}.")
        w("")
