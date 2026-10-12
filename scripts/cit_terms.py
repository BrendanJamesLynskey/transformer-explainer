"""
cit_terms.py — the training concepts the Compute in Transit results site
(https://compute-in-transit-results.vercel.app) relies on, for
cit_concept_audit.py.

Taken from the site's own text (its seven pages, fetched 2026-10-12 and read
in full): every training concept a reader meets there, with a phrase from the
page that uses it (`evidence`, checked against the live site by
`cit_concept_audit.py --refresh`), mapped to the Part 2 chapter that explains
it (`chapter`, a slug in TRAINING_SECTIONS) and an anchor in that chapter's
MDX (`<span id="t-…" />`). Terms specific to the experiment itself (fixed
bands, the emulated optics, ENOB, bit planes, the photonic module) are
explained on the results site and are not listed.

Each term: key, name, chapter slug, anchor id, alias regexes (the first is
the canonical one; matched case-insensitively in the anchored paragraph),
and (page, evidence phrase).
"""

TERMS = [
    # ── chapter 1: the loop ──────────────────────────────────────────────
    ("training-step", "Training step", "01-training-loop", "t-training-step",
     [r"training step", r"\bsteps?\b"], ("experiment", "3000 steps")),
    ("loss-curve", "Loss curve", "01-training-loop", "t-loss-curve",
     [r"loss curves?"], ("home", "Loss curves for every configuration")),
    ("validation-loss", "Validation loss (and evaluation interval)", "01-training-loop", "t-validation-loss",
     [r"validation loss", r"evaluated every"], ("home", "final validation losses")),
    ("seeds", "Seeds, mean ± sd, single-seed results", "01-training-loop", "t-seeds",
     [r"seeds?", r"mean ± standard deviation"], ("method", "5 configurations have three seeds")),
    ("parameters", "Parameters (model size)", "01-training-loop", "t-parameters",
     [r"parameters?", r"weights"], ("method", "15,833,472 parameters")),
    # ── chapter 2: data and tokens ───────────────────────────────────────
    ("bpe", "BPE tokeniser and vocabulary", "02-data-and-tokens", "t-bpe",
     [r"\bBPE\b", r"byte-pair"], ("method", "byte-level BPE, 4096 tokens")),
    ("context-length", "Context length", "02-data-and-tokens", "t-context-length",
     [r"context length", r"context"], ("method", "context 256")),
    ("batch", "Sequences and tokens per step (the batch)", "02-data-and-tokens", "t-context-length",
     [r"\bbatch\b", r"sequences"], ("method", "8 sequences of 256 tokens per worker per step")),
    ("dataset", "Training corpus (TinyStories)", "02-data-and-tokens", "t-dataset",
     [r"TinyStories", r"corpus|corpora"], ("notebook", "tinystories-gpu")),
    # ── chapter 3: the objective ─────────────────────────────────────────
    ("cross-entropy", "Next-token cross-entropy loss", "03-objective", "t-cross-entropy",
     [r"cross-entropy", r"minus the log"], ("home", "final loss")),
    ("next-token", "Next-token prediction (teacher forcing)", "03-objective", "t-next-token",
     [r"next-token prediction", r"teacher forcing"], ("method", "trained small GPTs")),
    # ── chapter 4: backprop ──────────────────────────────────────────────
    ("gradient", "Gradient (one number per parameter)", "04-backprop", "t-gradient",
     [r"gradient"], ("energy", "the step's gradient")),
    ("backprop", "Backpropagation", "04-backprop", "t-backprop",
     [r"backpropagation", r"backward pass"], ("energy", "the step's gradient")),
    # ── chapter 5: optimisers ────────────────────────────────────────────
    ("learning-rate", "Learning rate", "05-optimisers", "t-sgd",
     [r"learning rate"], ("method", "Learning rates came from a short sweep")),
    ("momentum", "Momentum (the optimiser's momentum buffer)", "05-optimisers", "t-momentum",
     [r"momentum"], ("home", "DeMo compresses each worker's momentum")),
    ("adam", "Adam: first and second moments, bias correction", "05-optimisers", "t-adam",
     [r"\bAdam\b", r"bias correction"], ("home", "AdamW")),
    ("adamw", "AdamW: decoupled weight decay", "05-optimisers", "t-adamw",
     [r"AdamW", r"weight decay"], ("home", "AdamW, the dense baseline")),
    # ── chapter 6: learning rate, batch size, schedules (29B) ────────────
    ("lr-sweep", "Learning-rate sweep, tuned per method", "06-learning-rate-and-schedules", "t-lr-sweep",
     [r"learning-rate sweep", r"sweep"], ("method", "Learning rates came from a short sweep on AdamW")),
    # ── chapter 9: precision (29B) ───────────────────────────────────────
    ("bf16", "BF16 (2-byte values)", "09-mixed-precision", "t-bf16",
     [r"\bbf16\b", r"bfloat16"], ("method", "values in BF16 (2 bytes)")),
    ("quantisation", "Quantised values, full scale and clipping", "09-mixed-precision", "t-quantisation",
     [r"quantis", r"quantiz"], ("optics", "each value quantised to P bit planes against a full scale")),
    # ── chapter 11: data parallelism (29C) ───────────────────────────────
    ("data-parallel", "Data parallelism and workers", "11-data-parallelism", "t-data-parallel",
     [r"data parallel"], ("method", "simulated data parallelism")),
    ("all-reduce", "Ring all-reduce", "11-data-parallelism", "t-all-reduce",
     [r"all-reduce"], ("experiment", "AdamW all-reduces the dense gradient")),
    ("all-gather", "All-gather (and why its bytes grow with the workers)", "11-data-parallelism", "t-all-gather",
     [r"all-gather"], ("experiment", "an all-gather's wire bytes grow with the number of workers")),
    ("wire-bytes", "Payload vs bytes on the wire per step", "11-data-parallelism", "t-wire-bytes",
     [r"on the wire", r"wire bytes"], ("method", "Wire = bytes a worker sends in a ring collective")),
    ("step-time", "Step time = compute + communication; overlap", "11-data-parallelism", "t-step-time",
     [r"step time", r"overlap"], ("method", "compute and communication do not overlap")),
    ("mfu", "MFU (model FLOPs utilisation)", "11-data-parallelism", "t-mfu",
     [r"\bMFU\b"], ("method", "an MFU of 0.40")),
    # ── chapter 13: training over slow links (29C) ───────────────────────
    ("top-k", "Top-k sparsification", "13-slow-links", "t-top-k",
     [r"top-k"], ("home", "keeping the top-k coefficients of a chunked DCT")),
    ("error-feedback", "Error feedback (the residual)", "13-slow-links", "t-error-feedback",
     [r"error feedback"], ("optics", "The cause was error feedback")),
    ("compressed-allgather", "Why compressed vectors are all-gathered, not all-reduced", "13-slow-links", "t-compressed-allgather",
     [r"all-gather"], ("experiment", "the DeMo variants all-gather their compressed values")),
    ("demo", "DeMo (DCT of momentum chunks)", "13-slow-links", "t-demo",
     [r"DeMo", r"\bDCT\b"], ("home", "DeMo compresses each worker's momentum")),
    ("diloco", "DiLoCo / syncing every H steps", "13-slow-links", "t-diloco",
     [r"DiLoCo", r"every H steps"], ("transit", "Syncing every H steps (DiLoCo-style)")),
    ("wan", "WAN / inter-data-centre bandwidth", "13-slow-links", "t-wan",
     [r"\bWAN\b", r"inter-?(DC|data-centre)"], ("transit", "100 GbE WAN")),
    ("queueing", "Queueing on a shared link, stragglers", "13-slow-links", "t-queueing",
     [r"queue", r"straggler"], ("transit", "A shared WAN: queueing")),
    # ── chapter 14: scaling (29C) ────────────────────────────────────────
    ("extrapolation", "Extrapolating to larger models", "14-scaling-laws", "t-extrapolation",
     [r"extrapolat"], ("transit", "Extrapolations: bigger models, rarer syncs")),
    # ── chapter 15: running a real job (29C) ─────────────────────────────
    ("checkpoint", "Checkpoints: saving and resuming a run", "15-running-a-real-job", "t-checkpoint",
     [r"checkpoint"], ("notebook", "an interrupted run resumes from its last checkpoint")),
    ("steps-to-loss", "Steps (and time) to reach a target loss", "15-running-a-real-job", "t-steps-to-loss",
     [r"steps to (reach )?a (target )?loss", r"time to"], ("experiment", "Steps to reach a loss")),
]
