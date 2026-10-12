"""
systems.py — the training-systems model of chapters 11, 12 and 15 (brief
29C): what one training step costs a cluster in memory, time and
communication, and what failures cost a long run. The reference for
src/lib/training/systems.ts (matched to fixtures in CI).

Everything here is closed-form arithmetic or a seeded discrete-event
simulation; nothing trains. The pieces:

  collectives   ring (and non-pipelined binary-tree) all-reduce,
                reduce-scatter, all-gather and all-to-all on an alpha-beta
                link model (time = latency * hops + bytes / bandwidth), with
                the bytes each rank sends ("on the wire")
  ring_states   a ring all-reduce actually run on N vectors, chunk by
                chunk (reduce-scatter, then all-gather), for chapter 11's
                animation: its result is checked equal to the sum
  zero_memory   bytes per device for data parallelism and ZeRO stages 1-3
                (Rajbhandari et al. 2020, arXiv:1910.02054, Figure 1: 2 + 2 +
                K bytes per parameter, K = 12 for mixed-precision Adam)
  device_memory weights, gradients, optimiser states and activations per
                GPU under tensor (t), pipeline (p), data (d, with a ZeRO
                stage), context (cp) and expert (ep) parallelism, with
                Korthikanti et al. (2022, arXiv:2205.05198) Table 2 for the
                activations and their (1 + (p-1)/(p v)) factor for the
                interleaved schedule
  pipeline      a discrete-event simulation of the GPipe, 1F1B and
                interleaved 1F1B schedules (Huang et al. 2019,
                arXiv:1811.06965; Narayanan et al. 2021, arXiv:2104.04473):
                every forward and backward of every micro-batch on every
                stage, so the bubble is measured, then checked against the
                closed forms (p - 1)/m and (p - 1)/(v m)
  step_time     compute (FLOPs / (peak * eta)) + the exposed communication
                (tensor-parallel all-reduces, context-parallel all-gathers,
                pipeline sends, the data-parallel gradient sync) + the
                pipeline bubble; throughput per GPU, MFU and HFU
  validation    step_time on the published configurations of Megatron-LM
                (Narayanan et al. 2021, Table 1, A100) and Llama 3 405B
                (Grattafiori et al. 2024, arXiv:2407.21783, Table 4, H100):
                one constant, eta, fitted on Megatron's smallest run (no
                tensor or pipeline parallelism), every other row predicted
  reliability   failures (exponential, a mean time between failures M),
                checkpoints every tau costing delta, restarts costing R:
                goodput by seeded simulation, against Young's (1974)
                tau* = sqrt(2 delta M)
  stragglers    synchronous steps wait for the slowest worker: a seeded
                simulation of per-worker step times, syncing every step or
                every H steps
  queueing      a shared WAN link (background transfers arriving at
                random): the M/D/1 waiting time, checked by simulation

Units: bytes, seconds, FLOP. Every hardware number is in HARDWARE with its
source; every assumption the validation makes is in ASSUMPTIONS.
"""

from __future__ import annotations

import math

import model as M

# ---------------------------------------------------------------------------
# Hardware. Peaks are dense BF16/FP16 matrix FLOP/s; link bandwidths are per
# GPU per direction.
# ---------------------------------------------------------------------------
HARDWARE = {
    "a100": dict(
        name="A100 80GB (DGX A100 nodes)", peak=312e12,
        nvlink=300e9, net=25e9, nvlink_alpha=5e-6, net_alpha=10e-6, per_node=8, hbm=80e9,
        source="Narayanan et al. 2021 §5 (312 TFLOP/s; eight 200 Gb/s HDR InfiniBand cards per "
               "8-GPU node, so 25 GB/s per GPU); NVLink 600 GB/s total per GPU (A100 datasheet), "
               "300 GB/s each way",
    ),
    "h100": dict(
        name="H100 80GB (8-GPU servers)", peak=989e12,
        nvlink=450e9, net=50e9, nvlink_alpha=5e-6, net_alpha=10e-6, per_node=8, hbm=80e9,
        source="989 TFLOP/s dense BF16 and NVLink 900 GB/s total (H100 SXM datasheet); "
               "400 Gb/s per GPU (Grattafiori et al. 2024 §3.3.1), so 50 GB/s",
    ),
}

ASSUMPTIONS = {
    "microBatch": "1 sequence per micro-batch (neither paper gives it for these runs)",
    "interleave": "v = layers per pipeline rank (one layer per model chunk), the most "
                  "interleaving the layer count allows; Llama 3 counts 128 layer slots "
                  "(126 layers + the embedding and output stages, its §3.3.2)",
    "bytes": "BF16 (2 bytes) for activations, parameters and gradients on the wire",
    "overlap": "nothing overlaps: every collective, send and the gradient sync is exposed",
    "latency": "5 us per NVLink hop, 10 us per network hop (Llama 3 §3.3.3: 'latency up to tens of "
               "microseconds' across a large cluster)",
    "flops": "Megatron rows: equation (3) of Narayanan et al. (with full recomputation); "
             "Llama rows: the PaLM MFU count 6N + 12 L H Q T per token (no recomputation)",
}


# ---------------------------------------------------------------------------
# Collectives on an alpha-beta model
# ---------------------------------------------------------------------------
def collective(kind: str, size: float, p: int, bw: float, alpha: float) -> dict:
    """Time and bytes each rank sends for one collective among p ranks.
    `size` is the full vector's bytes (for all-gather: the gathered result;
    for all-to-all: what each rank holds before the exchange).

      ring all-reduce   2 (p-1)/p * size / bw + 2 (p-1) alpha   (reduce-scatter, then all-gather)
      reduce-scatter    (p-1)/p * size / bw + (p-1) alpha
      all-gather        (p-1)/p * size / bw + (p-1) alpha
      tree all-reduce   2 ceil(log2 p) (alpha + size / bw)      (reduce up a binary tree, broadcast
                                                                 down; not pipelined)
      all-to-all        (p-1)/p * size / bw + (p-1) alpha
    """
    if p <= 1:
        return {"time": 0.0, "wire": 0.0}
    f = (p - 1) / p
    if kind == "allreduce":
        return {"time": 2 * f * size / bw + 2 * (p - 1) * alpha, "wire": 2 * f * size}
    if kind in ("reducescatter", "allgather", "alltoall"):
        return {"time": f * size / bw + (p - 1) * alpha, "wire": f * size}
    if kind == "tree":
        hops = math.ceil(math.log2(p))
        return {"time": 2 * hops * (alpha + size / bw), "wire": 2 * size}
    raise ValueError(kind)


def ring_states(vectors: list[list[float]]) -> list[dict]:
    """A ring all-reduce of N equal-length vectors (one per rank), run chunk
    by chunk: each vector is cut into N chunks; in reduce-scatter step k
    (k = 0 .. N-2) rank r sends chunk (r - k) mod N to rank r + 1, which adds
    it to its own copy; after N - 1 steps rank r holds the full sum of chunk
    (r + 1) mod N. In all-gather step k rank r sends chunk (r + 1 - k) mod N,
    which the receiver copies. After 2 (N - 1) steps every rank holds the
    sum. Returns the state before the first step and after every step:
    each rank's buffer, the transfers of the step and its phase.
    Additions are in ring order, so the TypeScript port matches exactly."""
    n = len(vectors)
    size = len(vectors[0])
    bounds = [(c * size) // n for c in range(n + 1)]
    buf = [list(v) for v in vectors]
    states = [{"phase": "start", "step": 0, "sends": [], "bufs": [list(b) for b in buf]}]
    for k in range(n - 1):
        sends = []
        incoming = []
        for r in range(n):
            c = (r - k) % n
            lo, hi = bounds[c], bounds[c + 1]
            incoming.append(((r + 1) % n, c, buf[r][lo:hi]))
            sends.append({"from": r, "to": (r + 1) % n, "chunk": c})
        for dst, c, vals in incoming:
            lo = bounds[c]
            for j, v in enumerate(vals):
                buf[dst][lo + j] = buf[dst][lo + j] + v
        states.append({"phase": "reduce-scatter", "step": k + 1, "sends": sends,
                       "bufs": [list(b) for b in buf]})
    for k in range(n - 1):
        sends = []
        incoming = []
        for r in range(n):
            c = (r + 1 - k) % n
            lo, hi = bounds[c], bounds[c + 1]
            incoming.append(((r + 1) % n, c, buf[r][lo:hi]))
            sends.append({"from": r, "to": (r + 1) % n, "chunk": c})
        for dst, c, vals in incoming:
            lo = bounds[c]
            for j, v in enumerate(vals):
                buf[dst][lo + j] = v
        states.append({"phase": "all-gather", "step": n + k, "sends": sends,
                       "bufs": [list(b) for b in buf]})
    return states


# ---------------------------------------------------------------------------
# Models: parameter counts and FLOPs
# ---------------------------------------------------------------------------
def gpt_arch(name, L, h, a, s, V=51200, f=None, kv=None, swiglu=False, tied=True) -> dict:
    """A decoder-only transformer's shape: L layers, width h, a heads,
    sequence s, vocabulary V, feed-forward width f (default 4h), kv heads
    (default a), SwiGLU (three FFN matrices) or GELU (two)."""
    return dict(name=name, L=L, h=h, a=a, s=s, V=V, f=f or 4 * h, kv=kv or a,
                swiglu=swiglu, tied=tied)


def layer_params(arch: dict) -> int:
    """One layer's matrix parameters (norms and biases are ignored: < 0.1%)."""
    h, f = arch["h"], arch["f"]
    hk = h * arch["kv"] // arch["a"]
    attn = 2 * h * h + 2 * h * hk
    ffn = (3 if arch["swiglu"] else 2) * h * f
    return attn + ffn


def n_params(arch: dict) -> int:
    emb = arch["V"] * arch["h"] * (1 if arch["tied"] else 2)
    return arch["L"] * layer_params(arch) + emb


def model_flops_per_token(arch: dict) -> float:
    """PaLM's MFU count (Chowdhery et al. 2022, arXiv:2204.02311, App. B):
    6 N + 12 L H Q T per token, the attention term with H Q = h, T = s."""
    return 6 * n_params(arch) + 12 * arch["L"] * arch["h"] * arch["s"]


def megatron_flops_per_step(arch: dict, B: int) -> float:
    """Narayanan et al. 2021 eq. (3): 96 B s l h^2 (1 + s/(6h) + V/(16 l h)),
    with full recomputation (four passes: forward, recomputed forward, two
    for the backward)."""
    s, L, h, V = arch["s"], arch["L"], arch["h"], arch["V"]
    return 96 * B * s * L * h * h * (1 + s / (6 * h) + V / (16 * L * h))


# ---------------------------------------------------------------------------
# Memory per device
# ---------------------------------------------------------------------------
def zero_memory(params: float, d: int, stage: int, K: int = 12) -> dict:
    """Model-state bytes per device for data parallelism over d ranks
    (ZeRO, Rajbhandari et al. 2020, Figure 1): weights 2, gradients 2 and K
    bytes of optimiser state per parameter, the stage choosing which are
    sharded: 0 none, 1 optimiser states, 2 + gradients, 3 + weights. Also
    the bytes each rank sends per step for the gradient sync (an
    all-reduce's worth; stage 3 adds an all-gather of the weights in the
    backward pass: 1.5x, the paper's §7)."""
    w = 2 * params / (d if stage >= 3 else 1)
    g = 2 * params / (d if stage >= 2 else 1)
    o = K * params / (d if stage >= 1 else 1)
    f = (d - 1) / d if d > 1 else 0.0
    wire = 2 * f * 2 * params * (1.5 if stage >= 3 else 1.0)
    return {"weights": w, "grads": g, "optimizer": o, "total": w + g + o, "wire": wire}


def activations_per_layer(arch: dict, b: int, t: int, sp: bool, recompute: str, cp: int = 1) -> float:
    """Korthikanti et al. Table 2, bytes per layer per micro-batch of b
    sequences (context parallelism splits the sequence over cp ranks)."""
    s, h, a = arch["s"] / cp, arch["h"], arch["a"]
    sbh = s * b * h
    if recompute == "full":
        return 2 * sbh
    if recompute == "selective":
        return sbh * (34 / t if sp else 10 + 24 / t)
    if sp:
        return sbh / t * (34 + 5 * a * s / h)
    return sbh * (10 + 24 / t + 5 * a * s / (h * t))


def device_memory(arch: dict, lay: dict) -> dict:
    """Bytes on the busiest GPU (the first pipeline stage: L/p layers, the
    embedding, and activations for p micro-batches in flight under 1F1B,
    times (1 + (p-1)/(p v)) when interleaved; Korthikanti et al. §4.2.3).
    Model states use 2 + 2 + 12 bytes per parameter with the ZeRO stage
    sharding over the d data-parallel ranks. Experts (MoE: `experts` E,
    sharded over ep ranks) multiply the FFN parameters."""
    t, p, d, ep = lay["t"], lay["p"], lay["d"], lay.get("ep", 1)
    v = lay.get("v", 1)
    per_layer = layer_params(arch)
    E = lay.get("experts", 1)
    if E > 1:
        ffn = (3 if arch["swiglu"] else 2) * arch["h"] * arch["f"]
        per_layer = per_layer - ffn + ffn * E / ep
    params = (arch["L"] / p) * per_layer / t + arch["V"] * arch["h"] / t
    states = zero_memory(params, d, lay.get("zero", 0))
    act_layer = activations_per_layer(arch, lay.get("b", 1), t, lay.get("sp", False),
                                      lay.get("recompute", "none"), lay.get("cp", 1))
    in_flight = min(p, lay.get("m", p)) if p > 1 else 1
    act = act_layer * (arch["L"] / p) * in_flight
    if p > 1 and v > 1:
        act *= 1 + (p - 1) / (p * v)
    total = states["total"] + act
    return {"params": params, "weights": states["weights"], "grads": states["grads"],
            "optimizer": states["optimizer"], "activations": act, "total": total}


# ---------------------------------------------------------------------------
# Pipeline schedules: a discrete-event simulation
# ---------------------------------------------------------------------------
def _order(kind: str, p: int, m: int, v: int, s: int) -> list[tuple]:
    """The order device s runs its work items: ("F" | "B", micro-batch,
    chunk). Chunk c of device s holds model stage c * p + s."""
    if kind == "gpipe":
        return [("F", i, 0) for i in range(m)] + [("B", i, 0) for i in range(m)]
    if kind == "1f1b":
        warm = min(p - s - 1, m)
        out = [("F", i, 0) for i in range(warm)]
        f, b = warm, 0
        while f < m:
            out.append(("F", f, 0))
            f += 1
            out.append(("B", b, 0))
            b += 1
        out += [("B", i, 0) for i in range(b, m)]
        return out
    if kind == "interleaved":
        # Megatron-LM's interleaved 1F1B (Narayanan et al. 2021 §2.2.2):
        # micro-batches go through the chunks in groups of p; warmup runs
        # (p - s - 1) * 2 + (v - 1) * p forwards first.
        if m % p:
            raise ValueError("interleaving needs m divisible by p")
        total = m * v

        def fwd(k):
            return ("F", (k // (p * v)) * p + k % p, (k // p) % v)

        def bwd(k):
            return ("B", (k // (p * v)) * p + k % p, v - 1 - (k // p) % v)

        warm = min((p - s - 1) * 2 + (v - 1) * p, total)
        out = [fwd(k) for k in range(warm)]
        f, b = warm, 0
        while f < total:
            out.append(fwd(f))
            f += 1
            out.append(bwd(b))
            b += 1
        out += [bwd(k) for k in range(b, total)]
        return out
    raise ValueError(kind)


def pipeline(kind: str, p: int, m: int, v: int = 1, tf: float = 1.0, tb: float = 2.0) -> dict:
    """Simulate one training step's pipeline: p devices, m micro-batches, v
    chunks per device (interleaved only); a forward of one chunk takes tf/v,
    a backward tb/v. A work item starts when its device is free and its
    input exists: a forward needs the same micro-batch's forward at the
    previous model stage, a backward the backward at the next stage (and,
    at the last stage, its own forward). Returns every item's start and end,
    the makespan, and the bubble: idle time over busy time per device."""
    if kind != "interleaved":
        v = 1
    S = p * v  # model stages
    order = [_order(kind, p, m, v, s) for s in range(p)]
    done: dict = {}
    items = []
    free = [0.0] * p
    pos = [0] * p
    tf_c, tb_c = tf / v, tb / v
    remaining = sum(len(o) for o in order)
    while remaining:
        progressed = False
        for s in range(p):
            while pos[s] < len(order[s]):
                op, mb, c = order[s][pos[s]]
                stage = c * p + s
                if op == "F":
                    dep = ("F", mb, stage - 1) if stage > 0 else None
                else:
                    dep = ("B", mb, stage + 1) if stage < S - 1 else ("F", mb, stage)
                if dep is not None and dep not in done:
                    break
                start = max(free[s], done[dep] if dep is not None else 0.0)
                end = start + (tf_c if op == "F" else tb_c)
                done[(op, mb, stage)] = end
                free[s] = end
                items.append({"device": s, "op": op, "mb": mb, "chunk": c, "start": start, "end": end})
                pos[s] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise RuntimeError("pipeline schedule deadlocked")
    makespan = max(it["end"] for it in items)
    ideal = m * (tf + tb)
    closed = (p - 1) / (m * v)
    items.sort(key=lambda it: (it["start"], it["device"]))
    return {"kind": kind, "p": p, "m": m, "v": v, "items": items, "makespan": makespan,
            "ideal": ideal, "bubble": (makespan - ideal) / ideal, "closedForm": closed}


# ---------------------------------------------------------------------------
# Step time
# ---------------------------------------------------------------------------
def step_time(arch: dict, lay: dict, hw: dict, eta: float, flops: str = "palm") -> dict:
    """One step's time on n = t p d cp GPUs with B sequences per step.

    compute    hardware FLOPs per GPU / (peak * eta), split over the
               micro-batches; with full recomputation one extra forward
    tensor     per layer per micro-batch: 4 all-reduces of the activations
               (s b h bytes x 2) among t GPUs over NVLink (2 forward, 2
               backward; 6 with full recomputation). Sequence parallelism
               swaps each for a reduce-scatter + all-gather: the same bytes.
    context    per layer per micro-batch: an all-gather of K and V among cp
               GPUs (forward) and a reduce-scatter of their gradients
               (backward), over the network
    pipeline   (m + (p - 1)/v) micro-batch slots of compute + tensor +
               context time (the bubble is (p - 1)/v of them), plus 2 v
               point-to-point sends of the activations per micro-batch
    data       the gradient sync: an all-reduce of the GPU's parameters among
               d GPUs over the network (ZeRO-3/FSDP: reduce-scatter +
               all-gather, the same bytes)
    """
    t, p, d, cp = lay["t"], lay["p"], lay["d"], lay.get("cp", 1)
    b, v = lay.get("b", 1), lay.get("v", 1)
    n = t * p * d * cp
    B = lay["B"]
    m = B // (d * b)
    s, h, L = arch["s"], arch["h"], arch["L"]
    recompute = lay.get("recompute", "none")
    if flops == "megatron":
        hw_flops = megatron_flops_per_step(arch, B)
        model_flops = hw_flops * 3 / 4
    else:
        model_flops = model_flops_per_token(arch) * B * s
        hw_flops = model_flops * (4 / 3 if recompute == "full" else 1)
    compute = hw_flops / n / (hw["peak"] * eta)
    per_mb_compute = compute / m
    # tensor parallel, per micro-batch, all layers of this GPU's stage
    act_bytes = (s / cp) * b * h * 2
    n_ar = 6 if recompute == "full" else 4
    nv = collective("allreduce", act_bytes, t, hw["nvlink"], hw["nvlink_alpha"]) if t > 1 else {"time": 0.0}
    per_mb_tensor = (L / p) * n_ar * nv["time"]
    # context parallel: K and V (kv heads x head size) all-gathered, grads reduce-scattered
    if cp > 1:
        kv_bytes = s * b * 2 * (h * arch["kv"] // arch["a"]) * 2 / t
        ag = collective("allgather", kv_bytes, cp, hw["net"], hw["net_alpha"])
        per_mb_context = (L / p) * 2 * ag["time"]
    else:
        per_mb_context = 0.0
    slot = per_mb_compute + per_mb_tensor + per_mb_context
    bubble_slots = (p - 1) / v if p > 1 else 0.0
    p2p = (2 * v * (act_bytes / t / hw["net"] + hw["net_alpha"]) * m) if p > 1 else 0.0
    pipe = (m + bubble_slots) * slot
    grads_bytes = 2 * n_params(arch) / (t * p)
    dp = collective("allreduce", grads_bytes, d * cp, hw["net"], hw["net_alpha"])["time"] if d * cp > 1 else 0.0
    total = pipe + p2p + dp
    per_gpu = hw_flops / n / total
    return {
        "gpus": n, "microBatches": m, "time": total,
        "compute": m * per_mb_compute, "tensor": m * per_mb_tensor, "context": m * per_mb_context,
        "bubble": bubble_slots * slot, "p2p": p2p, "data": dp,
        "hwFlopsPerGpu": per_gpu, "modelFlopsPerGpu": model_flops / n / total,
        "hfu": per_gpu / hw["peak"], "mfu": model_flops / n / total / hw["peak"],
        "tokensPerSecond": B * s / total,
    }


# ---------------------------------------------------------------------------
# Validation against published runs
# ---------------------------------------------------------------------------
# Narayanan et al. 2021, Table 1 (A100, s = 2048, V = 51,200): params (B),
# heads, hidden, layers, t, p, GPUs, batch, achieved TFLOP/s per GPU, % of peak.
MEGATRON_TABLE1 = [
    (1.7, 24, 2304, 24, 1, 1, 32, 512, 137, 44),
    (3.6, 32, 3072, 30, 2, 1, 64, 512, 138, 44),
    (7.5, 32, 4096, 36, 4, 1, 128, 512, 142, 46),
    (18.4, 48, 6144, 40, 8, 1, 256, 1024, 135, 43),
    (39.1, 64, 8192, 48, 8, 2, 512, 1536, 138, 44),
    (76.1, 80, 10240, 60, 8, 4, 1024, 1792, 140, 45),
    (145.6, 96, 12288, 80, 8, 8, 1536, 2304, 148, 47),
    (310.1, 128, 16384, 96, 8, 16, 1920, 2160, 155, 50),
    (529.6, 128, 20480, 105, 8, 35, 2520, 2520, 163, 52),
    (1008.0, 160, 25600, 128, 8, 64, 3072, 3072, 163, 52),
]

# Grattafiori et al. 2024, Table 3 (405B shape) and Table 4 (GPUs, TP, CP, PP,
# DP, sequence length, batch size per DP group, TFLOP/s per GPU, BF16 MFU).
LLAMA3_405B = gpt_arch("Llama 3 405B", L=126, h=16384, a=128, s=8192, V=128000, f=53248, kv=8,
                       swiglu=True, tied=False)
LLAMA3_TABLE4 = [
    (8192, 8, 1, 16, 64, 8192, 32, 430, 43),
    (16384, 8, 1, 16, 128, 8192, 16, 400, 41),
    (16384, 8, 16, 16, 8, 131072, 16, 380, 38),
]


def megatron_rows() -> list[dict]:
    rows = []
    for (pb, a, h, L, t, p, n, B, tf, pct) in MEGATRON_TABLE1:
        arch = gpt_arch(f"GPT {pb}B", L=L, h=h, a=a, s=2048)
        d = n // (t * p)
        v = L // p if p > 1 else 1
        lay = dict(t=t, p=p, d=d, B=B, b=1, v=v, recompute="full")
        rows.append({"source": "megatron", "label": f"{pb}B", "paramsB": pb, "arch": arch, "lay": lay,
                     "hw": "a100", "published": tf * 1e12, "publishedPct": pct, "flops": "megatron"})
    return rows


def llama_rows() -> list[dict]:
    rows = []
    for (n, t, cp, p, d, s, bdp, tf, mfu) in LLAMA3_TABLE4:
        arch = dict(LLAMA3_405B, s=s)
        lay = dict(t=t, p=p, d=d, cp=cp, B=bdp * d, b=1, v=128 // p, recompute="none", sp=True)
        rows.append({"source": "llama3", "label": f"{n:,} GPUs, seq {s:,}", "arch": arch, "lay": lay,
                     "hw": "h100", "published": tf * 1e12, "publishedPct": mfu, "flops": "palm"})
    return rows


def validation() -> dict:
    """Fit eta on Megatron's 1.7B run (t = p = 1: compute and the gradient
    sync only), then predict every other row of both tables with that eta."""
    rows = megatron_rows() + llama_rows()
    fit = rows[0]
    hw = HARDWARE[fit["hw"]]
    one = step_time(fit["arch"], fit["lay"], hw, 1.0, fit["flops"])
    # TFLOP/s per GPU = F / (n T); T = compute/eta + rest, so solve for eta
    rest = one["time"] - one["compute"]
    target_time = megatron_flops_per_step(fit["arch"], fit["lay"]["B"]) / one["gpus"] / fit["published"]
    eta = one["compute"] / (target_time - rest)
    out = []
    for r in rows:
        st = step_time(r["arch"], r["lay"], HARDWARE[r["hw"]], eta, r["flops"])
        pred = st["hwFlopsPerGpu"] if r["flops"] == "megatron" else st["modelFlopsPerGpu"]
        out.append({
            "source": r["source"], "label": r["label"], "hw": r["hw"], "arch": r["arch"], "flops": r["flops"],
            "params": n_params(r["arch"]), "layout": {k: r["lay"][k] for k in r["lay"]},
            "gpus": st["gpus"], "microBatches": st["microBatches"],
            "published": r["published"], "publishedPct": r["publishedPct"],
            "predicted": pred, "predictedPct": 100 * pred / HARDWARE[r["hw"]]["peak"],
            "error": pred / r["published"] - 1, "fitted": r is fit,
            "breakdown": {k: st[k] for k in ("time", "compute", "tensor", "context", "bubble", "p2p", "data")},
        })
    held = [o for o in out if not o["fitted"]]
    errs = [abs(o["error"]) for o in held]
    by = {src: max(abs(o["error"]) for o in held if o["source"] == src) for src in ("megatron", "llama3")}
    return {"eta": eta, "rows": out, "maxAbsError": max(errs),
            "meanAbsError": sum(errs) / len(errs), "maxAbsErrorBySource": by,
            "assumptions": ASSUMPTIONS}


def overlapped(compute: float, comm: float, backward_share: float = 2 / 3) -> float:
    """A step whose gradient sync runs during the backward pass (about two
    thirds of the compute): only the communication that outlasts the
    backward pass is exposed."""
    return compute + max(0.0, comm - backward_share * compute)


# ---------------------------------------------------------------------------
# Reliability: failures, checkpoints, goodput
# ---------------------------------------------------------------------------
def young_interval(delta: float, mtbf: float) -> float:
    """Young (1974): the checkpoint interval that minimises lost time to
    first order, tau* = sqrt(2 delta M)."""
    return math.sqrt(2 * delta * mtbf)


def expected_goodput(tau: float, delta: float, mtbf: float, restart: float) -> float:
    """First-order fraction of time doing useful work: each interval of
    useful work tau costs a checkpoint delta, and each failure (one per M
    of running time) loses on average half an interval plus the restart:
    goodput ~ 1 / (1 + delta / tau) * (1 - (tau / 2 + restart) / M)."""
    return (1 / (1 + delta / tau)) * max(0.0, 1 - (tau / 2 + restart) / mtbf)


def simulate_goodput(tau: float, delta: float, mtbf: float, restart: float, horizon: float,
                     seed: int) -> dict:
    """Run a job for `horizon` seconds: work in intervals of tau, each
    followed by a checkpoint of delta; failures arrive at exponential times
    (mean mtbf, mulberry32(seed), -ln(1-u) * M); a failure throws away the
    work since the last completed checkpoint and costs `restart`. Returns
    the useful time saved by checkpoints, the failures and the timeline
    (for chapter 15's animation): segments of work, checkpoint, lost work
    and restart."""
    rng = M.mulberry32(seed)

    def next_fail(now):
        return now + -math.log(1 - rng()) * mtbf

    now = 0.0
    fail = next_fail(now)
    useful = 0.0
    segs = []
    failures = 0
    while now < horizon:
        # work for tau then checkpoint for delta, unless a failure comes first
        end_work = now + tau
        end_ckpt = end_work + delta
        if fail >= end_ckpt:
            segs.append(("work", now, end_work))
            segs.append(("checkpoint", end_work, end_ckpt))
            useful += tau
            now = end_ckpt
            continue
        segs.append(("lost", now, fail))
        failures += 1
        r_end = fail + restart
        segs.append(("restart", fail, r_end))
        now = r_end
        fail = next_fail(now)
    # the last segment may run past the horizon: clip the accounting
    useful_in = sum(min(e, horizon) - s for k, s, e in segs if k == "work" and s < horizon)
    return {"goodput": useful_in / horizon, "failures": failures,
            "segments": [{"kind": k, "start": s, "end": e} for k, s, e in segs if s < horizon]}


def goodput_sweep(delta: float, mtbf: float, restart: float, horizon: float, taus: list[float],
                  seeds: list[int]) -> dict:
    rows = []
    for tau in taus:
        sims = [simulate_goodput(tau, delta, mtbf, restart, horizon, s)["goodput"] for s in seeds]
        rows.append({"tau": tau, "simulated": sum(sims) / len(sims),
                     "formula": expected_goodput(tau, delta, mtbf, restart)})
    best = max(rows, key=lambda r: r["simulated"])
    return {"delta": delta, "mtbf": mtbf, "restart": restart, "horizon": horizon, "seeds": seeds,
            "rows": rows, "bestTau": best["tau"], "young": young_interval(delta, mtbf)}


# ---------------------------------------------------------------------------
# Stragglers and queueing (chapter 13)
# ---------------------------------------------------------------------------
def stragglers(workers: int, steps: int, H: int, base: float, jitter: float, seed: int) -> dict:
    """Each worker's step takes base * (1 + jitter * e), e ~ Exp(1) drawn
    from mulberry32(seed) in worker order per step. Syncing every step, a
    step lasts as long as the slowest worker's; syncing every H steps,
    each worker runs H steps alone and the sync waits for the slowest sum.
    Returns the mean time per step both ways and the mean worker's."""
    rng = M.mulberry32(seed)
    times = [[base * (1 + jitter * -math.log(1 - rng())) for _ in range(workers)] for _ in range(steps)]
    sync_every = sum(max(row) for row in times) / steps
    rounds = steps // H
    sync_h = 0.0
    for k in range(rounds):
        sync_h += max(sum(times[k * H + j][w] for j in range(H)) for w in range(workers))
    sync_h /= rounds * H
    mean = sum(sum(row) for row in times) / (steps * workers)
    return {"workers": workers, "steps": steps, "H": H, "base": base, "jitter": jitter,
            "syncEveryStep": sync_every, "syncEveryH": sync_h, "meanWorker": mean}


def md1_wait(load: float, service: float) -> float:
    """Mean wait in an M/D/1 queue (Poisson arrivals, every transfer the
    same length `service`), Pollaczek-Khinchine: W = load * service /
    (2 (1 - load)). A sync message arriving at a link busy with background
    transfers at this load waits W on average before it starts."""
    if load >= 1:
        return math.inf
    return load * service / (2 * (1 - load))


def simulate_queue(load: float, service: float, horizon: float, probes: int, seed: int) -> float:
    """A link serving background transfers (Poisson arrivals at rate
    load / service, each `service` long) first come, first served; a probe
    (our sync, of negligible size) arrives at `probes` evenly spaced times
    and waits for the work ahead of it. Returns its mean wait."""
    rng = M.mulberry32(seed)
    rate = load / service
    t = 0.0
    arrivals = []
    while t < horizon:
        t += -math.log(1 - rng()) / rate
        arrivals.append(t)
    probe_times = [horizon * (k + 0.5) / probes for k in range(probes)]
    waits = []
    free_at = 0.0
    i = 0
    for pt in probe_times:
        while i < len(arrivals) and arrivals[i] <= pt:
            free_at = max(free_at, arrivals[i]) + service
            i += 1
        waits.append(max(0.0, free_at - pt))
    return sum(waits) / len(waits)
