"""
notebooks_build.py — writes the Colab notebooks in notebooks/ (brief 29B):
the GPU-scale versions of chapters 7-10's CPU experiments. One shared core
(data, model, training loop, checkpoint/resume, run metadata) and one
experiment cell per notebook, so the four stay consistent.

    python3 scripts/training/notebooks_build.py          # (re)write notebooks/*.ipynb
    python3 scripts/training/notebook_smoke.py           # run each with SMOKE=1 on the CPU

Each notebook: PyTorch only; seeded; checkpoints to Google Drive (or
./runs) so a dropped Colab session resumes; writes one results JSON with
the GPU, driver, torch and CUDA versions, seeds and wall time; SMOKE=1
runs a tiny version on the CPU in seconds (CI runs that).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks"

TESTED_TORCH = "2.11.0"

SETUP = r'''
import json, math, os, platform, subprocess, sys, time, urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SMOKE = os.environ.get("SMOKE") == "1"   # tiny CPU run (CI and a local check)
TESTED_TORCH = "%s"                        # the version this notebook was written and smoke-tested with
if torch.__version__.split("+")[0] != TESTED_TORCH:
    print(f"note: tested with torch {TESTED_TORCH}, running {torch.__version__} (recorded in the results)")

try:  # Colab: keep checkpoints and results on Drive, so a dropped session resumes
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive")
    BASE = Path("/content/drive/MyDrive/transformer-training-runs")
except Exception:
    BASE = Path(os.environ.get("RUNS_DIR", "runs"))
BASE.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() and not SMOKE else "cpu"
print("device:", DEVICE, "| results and checkpoints in", BASE)
''' % TESTED_TORCH

DATA = r'''
# Data: public-domain books from Project Gutenberg (the gutenberg.pglaf.org
# mirror), byte-level tokens (vocabulary 256), the last 5% for validation.
# Small (about 5 MB): fine for comparing settings, not for a good model.
# All are public domain in the USA; the Project Gutenberg header, footer
# and licence are stripped, as for the site's own corpus (CORPUS.md).
BOOKS = [11, 1342, 2701, 84, 1661, 98, 1080, 345]  # Alice, Pride and Prejudice, Moby Dick, Frankenstein, Sherlock Holmes, A Tale of Two Cities, A Modest Proposal, Dracula


def strip_gutenberg(text: str) -> str:
    a = text.find("*** START OF")
    b = text.find("*** END OF")
    if a != -1:
        text = text[text.find("\n", a) + 1:]
    if b != -1:
        text = text[: text.find("*** END OF")]
    return text


def load_corpus() -> bytes:
    if SMOKE:  # no network: the site's own corpus if it is here, else a stub
        local = Path("public/training/corpus.txt")
        return (local.read_text() if local.exists() else "alice was beginning to get very tired. " * 400).encode()
    cache = BASE / "corpus.bin"
    if cache.exists():
        return cache.read_bytes()
    parts = []
    for n in BOOKS:
        url = f"https://gutenberg.pglaf.org/cache/epub/{n}/pg{n}.txt"
        with urllib.request.urlopen(url, timeout=60) as r:
            parts.append(strip_gutenberg(r.read().decode("utf-8", "ignore")))
    data = "\n\n".join(parts).encode("utf-8")
    cache.write_bytes(data)
    return data


CORPUS = torch.tensor(list(load_corpus()), dtype=torch.long)
cut = int(len(CORPUS) * 0.95)
TRAIN, VAL = CORPUS[:cut], CORPUS[cut:]
print(f"corpus: {len(CORPUS):,} bytes (train {len(TRAIN):,}, validation {len(VAL):,})")


def batch(split, B, S, gen):
    src = TRAIN if split == "train" else VAL
    ix = torch.randint(len(src) - S - 1, (B,), generator=gen)
    x = torch.stack([src[i:i + S] for i in ix])
    y = torch.stack([src[i + 1:i + S + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)
'''

MODEL = r'''
# A small GPT (pre-LN by default), with the knobs the chapters measure:
# post-LN, QK-norm, z-loss and muP (Yang et al. 2022: hidden weights at
# std 1/sqrt(fan_in) relative to the base width, Adam LR / m for them,
# attention scores 1/d_k, logits / m), and optional activation
# checkpointing per block.
class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        d, h = c["d"], c["heads"]
        self.c = c
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.up = nn.Linear(d, 4 * d)
        self.down = nn.Linear(4 * d, d)
        if c.get("qk_norm"):
            self.qn, self.kn = nn.LayerNorm(d // h), nn.LayerNorm(d // h)
        self.max_score = 0.0

    def attn(self, x):
        B, S, D = x.shape
        h = self.c["heads"]
        q, k, v = self.qkv(x).view(B, S, 3, h, D // h).permute(2, 0, 3, 1, 4)
        if self.c.get("qk_norm"):
            q, k = self.qn(q), self.kn(k)
        dk = D // h
        scale = 1 / dk * math.sqrt(self.c["base"] // h) if self.c.get("mup") else 1 / math.sqrt(dk)
        s = (q @ k.transpose(-1, -2)) * scale
        self.max_score = max(self.max_score, float(s.detach().abs().max()))
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=x.device), 1)
        p = torch.softmax(s.masked_fill(mask, float("-inf")).float(), -1).to(v.dtype)
        return self.o((p @ v).transpose(1, 2).reshape(B, S, D))

    def ffn(self, x):
        return self.down(F.gelu(self.up(x), approximate="tanh"))

    def forward(self, x):
        if self.c.get("post"):
            x = self.ln1(x + self.attn(x))
            return self.ln2(x + self.ffn(x))
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(256, c["d"])
        self.pos = nn.Embedding(c["seq"], c["d"])
        self.blocks = nn.ModuleList(Block(c) for _ in range(c["layers"]))
        self.lnf = nn.LayerNorm(c["d"])
        m = c["d"] / c["base"] if c.get("mup") else 1.0
        for name, p in self.named_parameters():
            if p.dim() == 2:
                hidden = "blocks" in name
                std = 0.02 / math.sqrt(m) if (hidden and c.get("mup")) else 0.02
                nn.init.normal_(p, 0, std)
            elif name.endswith("bias"):
                nn.init.zeros_(p)
        self.m = m

    def forward(self, x, y):
        S = x.shape[1]
        h = self.emb(x) + self.pos(torch.arange(S, device=x.device))
        for b in self.blocks:
            if self.c.get("checkpoint") and self.training:
                h = torch.utils.checkpoint.checkpoint(b, h, use_reentrant=False)
            else:
                h = b(h)
        logits = self.lnf(h) @ self.emb.weight.T
        if self.c.get("mup"):
            logits = logits / self.m
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, 256), y.reshape(-1))
        if self.c.get("z_loss"):
            lse = torch.logsumexp(logits, -1)
            return loss, loss + self.c["z_loss"] * (lse * lse).mean()
        return loss, loss


def param_groups(model, lr, wd):
    """AdamW groups: decay on matrices; muP divides the hidden matrices' LR by m."""
    groups = []
    for name, p in model.named_parameters():
        hidden = p.dim() == 2 and "blocks" in name
        g_lr = lr / model.m if (hidden and model.c.get("mup")) else lr
        groups.append({"params": [p], "lr": g_lr, "base_lr": g_lr, "weight_decay": wd if p.dim() == 2 else 0.0})
    return groups
'''

LOOP = r'''
def lr_factor(t, total, warmup):
    if t < warmup:
        return (t + 1) / warmup
    return 0.1 + 0.45 * (1 + math.cos(math.pi * (t - warmup) / max(1, total - warmup)))


AMP = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def run(name, c, lr, steps, B, seed, precision="fp32", scaler=False, wd=0.1, warmup=None, eval_every=None,
        fake_fp8=False):
    """One training run; resumes from its checkpoint; returns its log."""
    path = BASE / f"{name}.pt"
    warmup = warmup if warmup is not None else max(1, steps // 20)
    eval_every = eval_every or max(1, steps // 20)
    torch.manual_seed(seed)
    model = GPT(c).to(DEVICE)
    opt = torch.optim.AdamW(param_groups(model, lr, wd), lr=lr, betas=(0.9, 0.95))
    gs = torch.amp.GradScaler("cuda", enabled=scaler and DEVICE == "cuda")
    gen = torch.Generator().manual_seed(seed)
    log = {"name": name, "config": c, "lr": lr, "steps": steps, "batch": B, "seed": seed,
           "precision": precision, "scaler": scaler, "fakeFp8": fake_fp8, "evals": [], "maxScore": [],
           "skipped": 0}
    t0, start = time.time(), 0
    if path.exists():
        ck = torch.load(path, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        gen.set_state(ck["gen"]); log = ck["log"]; start = ck["step"]
        print(f"{name}: resumed at step {start}")
    if fake_fp8:
        install_fake_fp8(model)
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tokens = 0
    for t in range(start, steps):
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * lr_factor(t, steps, warmup)
        x, y = batch("train", B, c["seq"], gen)
        with torch.autocast(DEVICE, dtype=AMP[precision], enabled=AMP[precision] is not None):
            loss, total = model(x, y)
        opt.zero_grad(set_to_none=True)
        gs.scale(total).backward()
        gs.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        before = gs.get_scale() if gs.is_enabled() else None
        gs.step(opt); gs.update()
        if before is not None and gs.get_scale() < before:
            log["skipped"] += 1
        tokens += B * c["seq"]
        if not math.isfinite(loss.item()):
            log["divergedAt"] = t
            break
        if (t + 1) % eval_every == 0 or t + 1 == steps:
            model.eval()
            with torch.no_grad(), torch.autocast(DEVICE, dtype=AMP[precision], enabled=AMP[precision] is not None):
                vg = torch.Generator().manual_seed(1234)
                vl = sum(model(*batch("val", B, c["seq"], vg))[0].item() for _ in range(4)) / 4
            model.train()
            log["evals"].append({"step": t + 1, "val": vl, "train": loss.item()})
            log["maxScore"].append({"step": t + 1, "value": max(b.max_score for b in model.blocks)})
            for b in model.blocks:
                b.max_score = 0.0
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "gen": gen.get_state(),
                        "log": log, "step": t + 1}, path)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        log["peakMemoryBytes"] = torch.cuda.max_memory_allocated()
    log["seconds"] = time.time() - t0
    log["tokensPerSecond"] = tokens / max(1e-9, log["seconds"])
    log["params"] = sum(p.numel() for p in model.parameters())
    return log


def install_fake_fp8(model):
    """FP8 emulation on any GPU: each Linear's input and weight are rounded
    through torch.float8_e4m3fn with a per-tensor scale (amax), as the FP8
    recipes do; the matrix product itself runs in the autocast precision."""
    def q(t):
        s = 448.0 / t.detach().abs().amax().clamp(min=1e-12)
        return (t * s).to(torch.float8_e4m3fn).to(t.dtype) / s
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            mod.forward = (lambda m: lambda x: F.linear(q(x), q(m.weight), m.bias))(mod)


def metadata():
    meta = {"torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
            "device": DEVICE, "smoke": SMOKE, "testedTorch": TESTED_TORCH,
            "date": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    if DEVICE == "cuda":
        meta["gpu"] = torch.cuda.get_device_name(0)
        try:
            meta["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                            capture_output=True, text=True).stdout.strip()
        except Exception:
            meta["driver"] = None
    return meta


def save(results, fname):
    out = {"metadata": metadata(), **results}
    p = BASE / fname
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)
    return p
'''

NOTEBOOKS = {
    "07_mup_width_sweep": {
        "title": "Chapter 7 on a GPU: muP vs the standard parameterisation, a width sweep",
        "intro": """The CPU experiment (results.md §7d-e) trains widths 16 and 64. Here: widths 128-1024
(about 1M-50M parameters), seven learning rates each, standard parameterisation vs muP (base
width 128). The claim to test: with muP the best learning rate stays put as the model widens; with
the standard parameterisation it moves (Yang et al. 2022, arXiv:2203.03466).

**GPU:** an L4 or A100. **Expected time:** about 1 h on an A100, 2-3 h on an L4 (56 runs of
2,000 steps). Runs checkpoint to Drive and resume, so a dropped session costs a few minutes.

**Output:** `mup_width_sweep.json` in `MyDrive/transformer-training-runs/`: put it in the site
repository as `notebooks/results/mup_width_sweep.json`.""",
        "experiment": r'''
WIDTHS = [32, 64] if SMOKE else [128, 256, 512, 1024]
LRS = [2.0 ** k for k in ([-8, -6] if SMOKE else range(-12, -5))]
STEPS, B, SEQ = (20, 4, 32) if SMOKE else (2000, 32, 256)
rows = []
for param in ("sp", "mup"):
    for d in WIDTHS:
        for lr in LRS:
            c = dict(d=d, heads=4, layers=2 if SMOKE else 4, seq=SEQ, base=WIDTHS[0], mup=param == "mup")
            log = run(f"mup-{param}-{d}-{lr:g}", c, lr, STEPS, B, seed=1, precision="fp32" if SMOKE else "bf16")
            rows.append({"param": param, "width": d, "lr": lr, "final": log["evals"][-1]["val"] if log["evals"] else None,
                         "divergedAt": log.get("divergedAt"), "params": log["params"], "evals": log["evals"]})
            print(param, d, lr, rows[-1]["final"])
save({"experiment": "mup_width_sweep", "steps": STEPS, "batch": B, "seq": SEQ, "rows": rows}, "mup_width_sweep.json")
''',
    },
    "08_attention_logit_growth": {
        "title": "Chapter 8 on a GPU: attention-logit growth and QK-norm at larger width",
        "intro": """The CPU experiment (results.md §8b) shows attention scores growing to 10^4-10^5 at a high
learning rate, and QK-norm keeping them small. Here the same comparison on a 6-layer, width-512
model (about 20M parameters): baseline, QK-norm and z-loss at four learning rates, two seeds,
logging the largest attention score (Wortsman et al. 2023, arXiv:2309.14322).

**GPU:** an L4 or A100. **Expected time:** about 45 min on an A100 (24 runs of 3,000 steps).

**Output:** `attention_logit_growth.json` → `notebooks/results/attention_logit_growth.json`.""",
        "experiment": r'''
LRS = [3e-3, 3e-2] if SMOKE else [1e-3, 3e-3, 1e-2, 3e-2]
SEEDS = [1] if SMOKE else [1, 2]
STEPS, B = (20, 4) if SMOKE else (3000, 32)
c0 = dict(d=64 if SMOKE else 512, heads=4 if SMOKE else 8, layers=2 if SMOKE else 6, seq=32 if SMOKE else 256, base=64)
rows = []
for name, extra in (("baseline", {}), ("QK-norm", {"qk_norm": True}), ("z-loss 1e-4", {"z_loss": 1e-4})):
    for lr in LRS:
        for seed in SEEDS:
            log = run(f"attn-{name}-{lr:g}-{seed}", dict(c0, **extra), lr, STEPS, B, seed,
                      precision="fp32" if SMOKE else "bf16")
            rows.append({"name": name, "lr": lr, "seed": seed, "evals": log["evals"], "maxScore": log["maxScore"],
                         "divergedAt": log.get("divergedAt")})
            print(name, lr, seed, log["evals"][-1] if log["evals"] else "diverged")
save({"experiment": "attention_logit_growth", "steps": STEPS, "batch": B, "rows": rows}, "attention_logit_growth.json")
''',
    },
    "09_mixed_precision": {
        "title": "Chapter 9 on a GPU: FP32 vs BF16 vs FP16 (with and without loss scaling) vs FP8",
        "intro": """The CPU experiment (results.md §9) emulates the formats by rounding in float64. Here the real
thing on a GPU: the same 8-layer, width-768 model (about 60M parameters) trained in FP32, BF16
autocast, FP16 autocast with and without a GradScaler, and an FP8 (E4M3) emulation with
per-tensor scaling of every Linear's inputs (a fake-quant: the products still run in BF16).
Measured: loss curves, tokens per second and peak memory.

**GPU:** an A100 (BF16 and FP16 tensor cores); an L4 also works (FP8 there is still emulated).
**Expected time:** about 40 min on an A100 (5 runs of 3,000 steps).

**Output:** `mixed_precision.json` → `notebooks/results/mixed_precision.json`.""",
        "experiment": r'''
STEPS, B = (20, 4) if SMOKE else (3000, 32)
c = dict(d=64 if SMOKE else 768, heads=4 if SMOKE else 12, layers=2 if SMOKE else 8, seq=32 if SMOKE else 256, base=64)
RECIPES = [("FP32", "fp32", False, False), ("BF16 autocast", "bf16", False, False),
           ("FP16 autocast + GradScaler", "fp16", True, False), ("FP16 autocast, no scaling", "fp16", False, False),
           ("FP8 E4M3 (emulated, per-tensor scaled) + BF16", "bf16", False, True)]
if SMOKE:  # CPU autocast has no FP16 GradScaler; FP8 casts need torch >= 2.1
    RECIPES = [r for r in RECIPES if r[1] != "fp16"]
rows = []
for name, prec, scaler, fp8 in RECIPES:
    log = run(f"prec-{name}", c, 6e-4, STEPS, B, seed=1, precision=prec, scaler=scaler, fake_fp8=fp8)
    rows.append({"name": name, "evals": log["evals"], "tokensPerSecond": log["tokensPerSecond"],
                 "peakMemoryBytes": log.get("peakMemoryBytes"), "skipped": log["skipped"],
                 "divergedAt": log.get("divergedAt")})
    print(name, log["evals"][-1] if log["evals"] else "diverged", round(log["tokensPerSecond"]))
save({"experiment": "mixed_precision", "steps": STEPS, "batch": B, "rows": rows}, "mixed_precision.json")
''',
    },
    "10_activation_checkpointing": {
        "title": "Chapter 10 on a GPU: activation checkpointing, memory against time",
        "intro": """Chapter 10's calculator predicts activation memory from Korthikanti et al. (2022). Here it is
measured: peak memory (`torch.cuda.max_memory_allocated`) and step time for a 12-layer, width-768
model at sequence lengths 512-2048, with and without checkpointing each block
(`torch.utils.checkpoint`), BF16 autocast, 20 timed steps each.

**GPU:** any (an L4 or a T4 is enough). **Expected time:** about 10 min.

**Output:** `activation_checkpointing.json` → `notebooks/results/activation_checkpointing.json`.""",
        "experiment": r'''
SEQS = [32, 64] if SMOKE else [512, 1024, 2048]
B = 2 if SMOKE else 8
STEPS = 3 if SMOKE else 20
rows = []
for seq in SEQS:
    for ckpt in (False, True):
        c = dict(d=64 if SMOKE else 768, heads=4 if SMOKE else 12, layers=2 if SMOKE else 12, seq=seq, base=64,
                 checkpoint=ckpt)
        name = f"ckpt-{seq}-{ckpt}"
        p = BASE / f"{name}.pt"
        if p.exists():
            p.unlink()  # a timing run: always from scratch
        log = run(name, c, 3e-4, STEPS, B, seed=1, precision="fp32" if SMOKE else "bf16", eval_every=STEPS)
        rows.append({"seq": seq, "checkpoint": ckpt, "params": log["params"], "batch": B,
                     "peakMemoryBytes": log.get("peakMemoryBytes"), "secondsPerStep": log["seconds"] / STEPS})
        print(seq, ckpt, rows[-1])
save({"experiment": "activation_checkpointing", "rows": rows}, "activation_checkpointing.json")
''',
    },
    # brief 29C (chapters 14-15)
    "14_isoflop_sweep": {
        "title": "Chapter 14 on a GPU: an IsoFLOP sweep",
        "intro": """The CPU sweep (results.md §14) uses Part 1's model at 1,000-70,000 parameters, where an
optimisation plateau decides most of the outcome. Here the same method (Hoffmann et al. 2022,
arXiv:2203.15556, Approach 2) on GPTs of about 1M-40M parameters: four compute budgets
C = 6 N D, six widths each, every run trained for exactly the tokens its budget allows with its
cosine schedule stretched to that length; a parabola in log N through each budget's three best
sizes gives N_opt(C), and a line through those gives the exponent a in N_opt ~ C^a. Still a toy
next to Chinchilla (70M-16B parameters): the corpus is about 5 MB of public-domain books, so the
larger budgets see it several times over.

**GPU:** an A100 (an L4 works, about 3x slower). **Expected time:** about 1 h on an A100 (24 runs).

**Output:** `isoflop_sweep.json` → `notebooks/results/isoflop_sweep.json`.""",
        "experiment": r'''
BUDGETS = [2e9, 6e9] if SMOKE else [3e14, 1e15, 3e15, 1e16]
WIDTHS = [32, 64, 96] if SMOKE else [128, 192, 256, 384, 512, 768]
B, SEQ, LAYERS = (4, 32, 2) if SMOKE else (32, 256, 6)


def n_params(d):
    c = dict(d=d, heads=max(1, d // 64), layers=LAYERS, seq=SEQ, base=64)
    return sum(p.numel() for p in GPT(c).parameters()), c


rows = []
for C in BUDGETS:
    for d in WIDTHS:
        N, c = n_params(d)
        steps = round(C / (6 * N * B * SEQ))
        if steps < 50:
            continue
        log = run(f"iso-{C:g}-{d}", c, 6e-4, steps, B, seed=1, precision="fp32" if SMOKE else "bf16")
        rows.append({"budget": C, "width": d, "params": N, "steps": steps, "tokens": steps * B * SEQ,
                     "final": log["evals"][-1]["val"] if log["evals"] else None,
                     "tokensPerSecond": log["tokensPerSecond"]})
        print(C, d, N, steps, rows[-1]["final"])


def fit_parabola(xs, ys):
    import numpy as np
    c2, c1, c0 = np.polyfit(xs, ys, 2)
    return (-c1 / (2 * c2)) if c2 > 0 else None


frontier = []
for C in BUDGETS:
    pts = [r for r in rows if r["budget"] == C and r["final"] is not None]
    if len(pts) < 3:
        continue
    i = min(range(len(pts)), key=lambda j: pts[j]["final"])
    if 0 < i < len(pts) - 1:
        near = pts[i - 1:i + 2]
        x = fit_parabola([math.log10(p["params"]) for p in near], [p["final"] for p in near])
        if x is not None:
            frontier.append({"budget": C, "nOpt": 10 ** x})
a = None
if len(frontier) >= 2:
    import numpy as np
    a = float(np.polyfit([math.log10(f["budget"]) for f in frontier], [math.log10(f["nOpt"]) for f in frontier], 1)[0])
print("frontier", frontier, "a =", a)
save({"experiment": "isoflop_sweep", "batch": B, "seq": SEQ, "layers": LAYERS, "rows": rows,
      "frontier": frontier, "a": a}, "isoflop_sweep.json")
''',
    },
    "15_throughput_mfu": {
        "title": "Chapter 15 on a GPU: throughput, MFU and an exact resume",
        "intro": """Two things a real job measures. Throughput and MFU: a 12-layer, width-768 GPT (about 85M
parameters with its embedding) at sequence 1,024, BF16 autocast, 30 timed steps after 5 warm-up
steps; MFU counts 6N + 12 L H Q T FLOPs per token (Chowdhery et al. 2022, arXiv:2204.02311,
App. B) against the GPU's dense BF16 peak, which you should check against your GPU's datasheet
(the table below is the vendor's figures for common Colab GPUs, halved where the datasheet quotes
sparsity). And a resume check: a run stopped and resumed from its checkpoint (weights, optimiser,
data generator) against the same run uninterrupted.

**GPU:** any. **Expected time:** about 10 min.

**Output:** `throughput_mfu.json` → `notebooks/results/throughput_mfu.json`.""",
        "experiment": r'''
# dense BF16 (FP16 for the T4) tensor-core peaks, FLOP/s: vendor datasheets
PEAKS = {"A100": 312e12, "H100": 989e12, "L4": 121e12, "T4": 65e12}
c = dict(d=64 if SMOKE else 768, heads=4 if SMOKE else 12, layers=2 if SMOKE else 12, seq=32 if SMOKE else 1024,
         base=64)
B = 2 if SMOKE else 16
WARM, TIMED = (1, 3) if SMOKE else (5, 30)
torch.manual_seed(1)
model = GPT(c).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
gen = torch.Generator().manual_seed(1)
prec = None if (SMOKE or DEVICE != "cuda") else torch.bfloat16
for t in range(WARM + TIMED):
    if t == WARM:
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
    x, y = batch("train", B, c["seq"], gen)
    with torch.autocast(DEVICE, dtype=prec, enabled=prec is not None):
        loss, _ = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
if DEVICE == "cuda":
    torch.cuda.synchronize()
secs = (time.time() - t0) / TIMED
N = sum(p.numel() for p in model.parameters())
flops_per_token = 6 * N + 12 * c["layers"] * c["d"] * c["seq"]
tps = B * c["seq"] / secs
name = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"
peak = next((v for k, v in PEAKS.items() if k in name), None)
mfu = tps * flops_per_token / peak if peak else None
print(f"{name}: {tps:,.0f} tokens/s, {secs * 1000:.1f} ms/step, MFU {mfu}")

# resume check: 2 x K steps straight, against K steps + checkpoint + K steps
K = 3 if SMOKE else 20


def train_steps(model, opt, gen, n):
    for _ in range(n):
        x, y = batch("train", B, c["seq"], gen)
        with torch.autocast(DEVICE, dtype=prec, enabled=prec is not None):
            loss, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return loss.item()


def fresh():
    torch.manual_seed(2)
    m = GPT(c).to(DEVICE)
    return m, torch.optim.AdamW(m.parameters(), lr=3e-4), torch.Generator().manual_seed(2)


m1, o1, g1 = fresh()
straight = train_steps(m1, o1, g1, 2 * K)
m2, o2, g2 = fresh()
train_steps(m2, o2, g2, K)
ck = BASE / "resume_check.pt"
torch.save({"model": m2.state_dict(), "opt": o2.state_dict(), "gen": g2.get_state()}, ck)
m3, o3, g3 = fresh()
state = torch.load(ck, weights_only=False)
m3.load_state_dict(state["model"]); o3.load_state_dict(state["opt"]); g3.set_state(state["gen"])
resumed = train_steps(m3, o3, g3, K)
diff = max((a - b).abs().max().item() for a, b in zip(m1.parameters(), m3.parameters()))
print("resume: straight", straight, "resumed", resumed, "max weight difference", diff)
save({"experiment": "throughput_mfu", "rows": [{"gpu": name, "params": N, "batch": B, "seq": c["seq"],
      "secondsPerStep": secs, "tokensPerSecond": tps, "flopsPerToken": flops_per_token, "peak": peak,
      "mfu": mfu, "resumeLossStraight": straight, "resumeLossResumed": resumed, "resumeMaxWeightDiff": diff}]},
     "throughput_mfu.json")
''',
    },
}


def cell(kind: str, src: str) -> dict:
    lines = src.strip("\n").split("\n")
    c = {"cell_type": kind, "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def build() -> None:
    OUT.mkdir(exist_ok=True)
    for name, nb in NOTEBOOKS.items():
        badge = (f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
                 f"(https://colab.research.google.com/github/BrendanJamesLynskey/transformer-explainer/blob/main/notebooks/{name}.ipynb)")
        cells = [
            cell("markdown", f"# {nb['title']}\n\n{badge}\n\n{nb['intro']}\n\n"
                 "Part of *Transformer Decoder Explained*, Part 2 (Training). Generated by "
                 "`scripts/training/notebooks_build.py`; `SMOKE=1` runs a tiny version on a CPU in seconds."),
            cell("code", SETUP),
            cell("code", DATA),
            cell("code", MODEL),
            cell("code", LOOP),
            cell("code", nb["experiment"]),
        ]
        doc = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                            "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                            "language_info": {"name": "python"}},
               "nbformat": 4, "nbformat_minor": 5}
        for i, c in enumerate(doc["cells"]):
            c["id"] = f"{name[:2]}-{i}"
        (OUT / f"{name}.ipynb").write_text(json.dumps(doc, indent=1) + "\n")
        print("wrote", OUT / f"{name}.ipynb")


if __name__ == "__main__":
    build()
