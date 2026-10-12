/*
 * src/lib/training/systems.ts
 *
 * Operation: the training-systems model of chapters 11, 12, 13 and 15: what
 *            a training step costs a cluster in memory, time and bytes on
 *            the wire, and what failures cost a long run.
 * Shapes:    plain numbers and small records (bytes, seconds, FLOP).
 * Intuition: a step is compute (FLOPs / (peak × η)) plus every piece of
 *            communication nothing hides: tensor-parallel all-reduces inside
 *            a server, pipeline sends and the pipeline's bubble, the
 *            data-parallel gradient sync across servers. Collectives cost
 *            latency × hops + bytes / bandwidth (an α–β model); a ring
 *            all-reduce sends 2 (p − 1)/p of the vector from every rank,
 *            whatever the ring's size.
 * Sources:   ZeRO (Rajbhandari et al. 2020, arXiv:1910.02054, Figure 1);
 *            Korthikanti et al. 2022 (arXiv:2205.05198, Table 2, §4.2.3);
 *            Narayanan et al. 2021 (arXiv:2104.04473, §2.2, eq. 3, Table 1);
 *            Grattafiori et al. 2024 (Llama 3, arXiv:2407.21783, Tables 3–4);
 *            Chowdhery et al. 2022 (PaLM, arXiv:2204.02311, App. B: MFU);
 *            Young 1974 (the checkpoint interval).
 *            Port of scripts/training/systems.py, matched to its fixtures
 *            (tests/unit/training/systems.test.ts).
 * MDX:       /training/11-data-parallelism, /training/12-model-parallelism,
 *            /training/13-slow-links, /training/15-running-a-real-job.
 */
import { mulberry32 } from "@/lib/transformer/random";

// ─── hardware ────────────────────────────────────────────────────────────
export type Hardware = {
  name: string;
  /** dense BF16/FP16 matrix FLOP/s */
  peak: number;
  /** NVLink bytes/s per GPU per direction */
  nvlink: number;
  /** scale-out network bytes/s per GPU */
  net: number;
  nvlinkAlpha: number;
  netAlpha: number;
  perNode: number;
  /** HBM bytes */
  hbm: number;
};

export const HARDWARE: Record<"a100" | "h100", Hardware> = {
  a100: {
    name: "A100 80GB (DGX A100 nodes)",
    peak: 312e12,
    nvlink: 300e9,
    net: 25e9,
    nvlinkAlpha: 5e-6,
    netAlpha: 10e-6,
    perNode: 8,
    hbm: 80e9,
  },
  h100: {
    name: "H100 80GB (8-GPU servers)",
    peak: 989e12,
    nvlink: 450e9,
    net: 50e9,
    nvlinkAlpha: 5e-6,
    netAlpha: 10e-6,
    perNode: 8,
    hbm: 80e9,
  },
};

// ─── collectives ─────────────────────────────────────────────────────────
export type CollectiveKind =
  | "allreduce"
  | "reducescatter"
  | "allgather"
  | "alltoall"
  | "tree";

/**
 * Time and bytes each rank sends for one collective among p ranks; `size`
 * is the full vector's bytes.
 *   ring all-reduce  2 (p−1)/p · size/bw + 2 (p−1) α
 *   reduce-scatter, all-gather, all-to-all  (p−1)/p · size/bw + (p−1) α
 *   tree all-reduce  2 ⌈log₂ p⌉ (α + size/bw)  (not pipelined)
 */
export function collective(
  kind: CollectiveKind,
  size: number,
  p: number,
  bw: number,
  alpha: number,
): { time: number; wire: number } {
  if (p <= 1) return { time: 0, wire: 0 };
  const f = (p - 1) / p;
  if (kind === "allreduce")
    return {
      time: (2 * f * size) / bw + 2 * (p - 1) * alpha,
      wire: 2 * f * size,
    };
  if (kind === "tree") {
    const hops = Math.ceil(Math.log2(p));
    return { time: 2 * hops * (alpha + size / bw), wire: 2 * size };
  }
  return { time: (f * size) / bw + (p - 1) * alpha, wire: f * size };
}

export type RingSend = { from: number; to: number; chunk: number };
export type RingState = {
  phase: "start" | "reduce-scatter" | "all-gather";
  step: number;
  sends: RingSend[];
  bufs: number[][];
};

/**
 * A ring all-reduce of N vectors (one per rank), chunk by chunk: in
 * reduce-scatter step k rank r sends chunk (r − k) mod N to rank r + 1,
 * which adds it to its own; in all-gather step k rank r sends chunk
 * (r + 1 − k) mod N, which the receiver copies. After 2 (N − 1) steps every
 * rank holds the sum. Returns the start state and the state after each step.
 */
export function ringStates(vectors: number[][]): RingState[] {
  const n = vectors.length;
  const size = vectors[0]!.length;
  const bounds = Array.from({ length: n + 1 }, (_, c) =>
    Math.floor((c * size) / n),
  );
  const buf = vectors.map((v) => [...v]);
  const states: RingState[] = [
    { phase: "start", step: 0, sends: [], bufs: buf.map((b) => [...b]) },
  ];
  const run = (phase: "reduce-scatter" | "all-gather", k: number) => {
    const sends: RingSend[] = [];
    const incoming: { dst: number; c: number; vals: number[] }[] = [];
    for (let r = 0; r < n; r++) {
      const c =
        phase === "reduce-scatter"
          ? (((r - k) % n) + n) % n
          : (((r + 1 - k) % n) + n) % n;
      incoming.push({
        dst: (r + 1) % n,
        c,
        vals: buf[r]!.slice(bounds[c]!, bounds[c + 1]!),
      });
      sends.push({ from: r, to: (r + 1) % n, chunk: c });
    }
    for (const { dst, c, vals } of incoming) {
      const lo = bounds[c]!;
      vals.forEach((v, j) => {
        buf[dst]![lo + j] =
          phase === "reduce-scatter" ? buf[dst]![lo + j]! + v : v;
      });
    }
    return sends;
  };
  for (let k = 0; k < n - 1; k++) {
    const sends = run("reduce-scatter", k);
    states.push({
      phase: "reduce-scatter",
      step: k + 1,
      sends,
      bufs: buf.map((b) => [...b]),
    });
  }
  for (let k = 0; k < n - 1; k++) {
    const sends = run("all-gather", k);
    states.push({
      phase: "all-gather",
      step: n + k,
      sends,
      bufs: buf.map((b) => [...b]),
    });
  }
  return states;
}

// ─── models ──────────────────────────────────────────────────────────────
export type SysArch = {
  name: string;
  L: number;
  h: number;
  a: number;
  s: number;
  V: number;
  f: number;
  kv: number;
  swiglu: boolean;
  tied: boolean;
};

export function gptArch(
  name: string,
  L: number,
  h: number,
  a: number,
  s: number,
  opts: Partial<Pick<SysArch, "V" | "f" | "kv" | "swiglu" | "tied">> = {},
): SysArch {
  return {
    name,
    L,
    h,
    a,
    s,
    V: opts.V ?? 51200,
    f: opts.f ?? 4 * h,
    kv: opts.kv ?? a,
    swiglu: opts.swiglu ?? false,
    tied: opts.tied ?? true,
  };
}

/** One layer's matrix parameters (norms and biases ignored: < 0.1%). */
export function layerParams(arch: SysArch): number {
  const { h, f } = arch;
  const hk = Math.floor((h * arch.kv) / arch.a);
  return 2 * h * h + 2 * h * hk + (arch.swiglu ? 3 : 2) * h * f;
}

export function nParams(arch: SysArch): number {
  return arch.L * layerParams(arch) + arch.V * arch.h * (arch.tied ? 1 : 2);
}

/** PaLM's MFU count per token: 6N + 12 L H Q T (H Q = h, T = s). */
export function modelFlopsPerToken(arch: SysArch): number {
  return 6 * nParams(arch) + 12 * arch.L * arch.h * arch.s;
}

/** Narayanan et al. eq. 3, with full recomputation. */
export function megatronFlopsPerStep(arch: SysArch, B: number): number {
  const { s, L, h, V } = arch;
  return 96 * B * s * L * h * h * (1 + s / (6 * h) + V / (16 * L * h));
}

// ─── memory ──────────────────────────────────────────────────────────────
export type ZeroMem = {
  weights: number;
  grads: number;
  optimizer: number;
  total: number;
  wire: number;
};

/**
 * ZeRO's model states per device, data parallel over d ranks: 2 + 2 + K
 * bytes per parameter; stage 1 shards the optimiser states, 2 also the
 * gradients, 3 also the weights. `wire`: bytes a rank sends per step for
 * the gradient sync (×1.5 at stage 3, which all-gathers the weights again).
 */
export function zeroMemory(
  params: number,
  d: number,
  stage: number,
  K = 12,
): ZeroMem {
  const w = (2 * params) / (stage >= 3 ? d : 1);
  const g = (2 * params) / (stage >= 2 ? d : 1);
  const o = (K * params) / (stage >= 1 ? d : 1);
  const f = d > 1 ? (d - 1) / d : 0;
  const wire = 2 * f * 2 * params * (stage >= 3 ? 1.5 : 1);
  return { weights: w, grads: g, optimizer: o, total: w + g + o, wire };
}

export type Recompute = "none" | "selective" | "full";

/** Korthikanti et al. Table 2: activation bytes per layer per micro-batch. */
export function activationsPerLayer(
  arch: SysArch,
  b: number,
  t: number,
  sp: boolean,
  recompute: Recompute,
  cp = 1,
): number {
  const s = arch.s / cp;
  const { h, a } = arch;
  const sbh = s * b * h;
  if (recompute === "full") return 2 * sbh;
  if (recompute === "selective") return sbh * (sp ? 34 / t : 10 + 24 / t);
  if (sp) return (sbh / t) * (34 + (5 * a * s) / h);
  return sbh * (10 + 24 / t + (5 * a * s) / (h * t));
}

export type Layout = {
  t: number;
  p: number;
  d: number;
  cp?: number;
  ep?: number;
  experts?: number;
  zero?: number;
  sp?: boolean;
  recompute?: Recompute;
  /** sequences per micro-batch */
  b?: number;
  /** model chunks per pipeline rank (interleaving) */
  v?: number;
  /** micro-batches in flight (default p) */
  m?: number;
  /** global batch, sequences */
  B?: number;
};

export type DeviceMemory = {
  params: number;
  weights: number;
  grads: number;
  optimizer: number;
  activations: number;
  total: number;
};

/** Bytes on the first pipeline stage's GPU (the busiest). */
export function deviceMemory(arch: SysArch, lay: Layout): DeviceMemory {
  const { t, p, d } = lay;
  const ep = lay.ep ?? 1;
  const v = lay.v ?? 1;
  let perLayer = layerParams(arch);
  const E = lay.experts ?? 1;
  if (E > 1) {
    const ffn = (arch.swiglu ? 3 : 2) * arch.h * arch.f;
    perLayer = perLayer - ffn + (ffn * E) / ep;
  }
  const params = ((arch.L / p) * perLayer) / t + (arch.V * arch.h) / t;
  const states = zeroMemory(params, d, lay.zero ?? 0);
  const actLayer = activationsPerLayer(
    arch,
    lay.b ?? 1,
    t,
    lay.sp ?? false,
    lay.recompute ?? "none",
    lay.cp ?? 1,
  );
  const inFlight = p > 1 ? Math.min(p, lay.m ?? p) : 1;
  let act = actLayer * (arch.L / p) * inFlight;
  if (p > 1 && v > 1) act *= 1 + (p - 1) / (p * v);
  return {
    params,
    weights: states.weights,
    grads: states.grads,
    optimizer: states.optimizer,
    activations: act,
    total: states.total + act,
  };
}

// ─── pipeline schedules (discrete-event simulation) ──────────────────────
export type PipeKind = "gpipe" | "1f1b" | "interleaved";
type Work = ["F" | "B", number, number];

function order(kind: PipeKind, p: number, m: number, v: number, s: number) {
  const out: Work[] = [];
  if (kind === "gpipe") {
    for (let i = 0; i < m; i++) out.push(["F", i, 0]);
    for (let i = 0; i < m; i++) out.push(["B", i, 0]);
    return out;
  }
  if (kind === "1f1b") {
    const warm = Math.min(p - s - 1, m);
    for (let i = 0; i < warm; i++) out.push(["F", i, 0]);
    let f = warm;
    let b = 0;
    while (f < m) {
      out.push(["F", f++, 0]);
      out.push(["B", b++, 0]);
    }
    for (let i = b; i < m; i++) out.push(["B", i, 0]);
    return out;
  }
  if (m % p) throw new Error("interleaving needs m divisible by p");
  const total = m * v;
  const fwd = (k: number): Work => [
    "F",
    Math.floor(k / (p * v)) * p + (k % p),
    Math.floor(k / p) % v,
  ];
  const bwd = (k: number): Work => [
    "B",
    Math.floor(k / (p * v)) * p + (k % p),
    v - 1 - (Math.floor(k / p) % v),
  ];
  const warm = Math.min((p - s - 1) * 2 + (v - 1) * p, total);
  for (let k = 0; k < warm; k++) out.push(fwd(k));
  let f = warm;
  let b = 0;
  while (f < total) {
    out.push(fwd(f++));
    out.push(bwd(b++));
  }
  for (let k = b; k < total; k++) out.push(bwd(k));
  return out;
}

export type PipeItem = {
  device: number;
  op: "F" | "B";
  mb: number;
  chunk: number;
  start: number;
  end: number;
};

export type PipeResult = {
  kind: PipeKind;
  p: number;
  m: number;
  v: number;
  items: PipeItem[];
  makespan: number;
  ideal: number;
  bubble: number;
  closedForm: number;
};

/**
 * Simulate one step's pipeline: p devices, m micro-batches, v chunks per
 * device (interleaved only). A forward of one chunk takes tf/v, a backward
 * tb/v; an item starts when its device is free and its input exists.
 * Bubble = (makespan − ideal)/ideal, ideal = m (tf + tb).
 */
export function pipeline(
  kind: PipeKind,
  p: number,
  m: number,
  v = 1,
  tf = 1,
  tb = 2,
): PipeResult {
  const V = kind === "interleaved" ? v : 1;
  const S = p * V;
  const orders = Array.from({ length: p }, (_, s) => order(kind, p, m, V, s));
  const done = new Map<string, number>();
  const items: PipeItem[] = [];
  const free = new Array<number>(p).fill(0);
  const pos = new Array<number>(p).fill(0);
  const tfc = tf / V;
  const tbc = tb / V;
  let remaining = orders.reduce((a, o) => a + o.length, 0);
  while (remaining) {
    let progressed = false;
    for (let s = 0; s < p; s++) {
      while (pos[s]! < orders[s]!.length) {
        const [op, mb, c] = orders[s]![pos[s]!]!;
        const stage = c * p + s;
        let dep: string | null;
        if (op === "F") dep = stage > 0 ? `F,${mb},${stage - 1}` : null;
        else dep = stage < S - 1 ? `B,${mb},${stage + 1}` : `F,${mb},${stage}`;
        if (dep !== null && !done.has(dep)) break;
        const start = Math.max(free[s]!, dep !== null ? done.get(dep)! : 0);
        const end = start + (op === "F" ? tfc : tbc);
        done.set(`${op},${mb},${stage}`, end);
        free[s] = end;
        items.push({ device: s, op, mb, chunk: c, start, end });
        pos[s]!++;
        remaining--;
        progressed = true;
      }
    }
    if (!progressed) throw new Error("pipeline schedule deadlocked");
  }
  const makespan = Math.max(...items.map((it) => it.end));
  const ideal = m * (tf + tb);
  items.sort((a, b) => a.start - b.start || a.device - b.device);
  return {
    kind,
    p,
    m,
    v: V,
    items,
    makespan,
    ideal,
    bubble: (makespan - ideal) / ideal,
    closedForm: (p - 1) / (m * V),
  };
}

// ─── step time ───────────────────────────────────────────────────────────
export type StepTime = {
  gpus: number;
  microBatches: number;
  time: number;
  compute: number;
  tensor: number;
  context: number;
  bubble: number;
  p2p: number;
  data: number;
  hwFlopsPerGpu: number;
  modelFlopsPerGpu: number;
  hfu: number;
  mfu: number;
  tokensPerSecond: number;
};

/** One step's time on t·p·d·cp GPUs (see systems.py:step_time). */
export function stepTime(
  arch: SysArch,
  lay: Layout & { B: number },
  hw: Hardware,
  eta: number,
  flops: "palm" | "megatron" = "palm",
): StepTime {
  const { t, p, d } = lay;
  const cp = lay.cp ?? 1;
  const b = lay.b ?? 1;
  const v = lay.v ?? 1;
  const n = t * p * d * cp;
  const B = lay.B;
  const m = Math.floor(B / (d * b));
  const { s, h, L } = arch;
  const recompute = lay.recompute ?? "none";
  let hwFlops: number;
  let modelFlops: number;
  if (flops === "megatron") {
    hwFlops = megatronFlopsPerStep(arch, B);
    modelFlops = (hwFlops * 3) / 4;
  } else {
    modelFlops = modelFlopsPerToken(arch) * B * s;
    hwFlops = modelFlops * (recompute === "full" ? 4 / 3 : 1);
  }
  const compute = hwFlops / n / (hw.peak * eta);
  const perMbCompute = compute / m;
  const actBytes = (s / cp) * b * h * 2;
  const nAr = recompute === "full" ? 6 : 4;
  const nv =
    t > 1
      ? collective("allreduce", actBytes, t, hw.nvlink, hw.nvlinkAlpha).time
      : 0;
  const perMbTensor = (L / p) * nAr * nv;
  let perMbContext = 0;
  if (cp > 1) {
    const kvBytes = (s * b * 2 * Math.floor((h * arch.kv) / arch.a) * 2) / t;
    const ag = collective("allgather", kvBytes, cp, hw.net, hw.netAlpha);
    perMbContext = (L / p) * 2 * ag.time;
  }
  const slot = perMbCompute + perMbTensor + perMbContext;
  const bubbleSlots = p > 1 ? (p - 1) / v : 0;
  const p2p = p > 1 ? 2 * v * (actBytes / t / hw.net + hw.netAlpha) * m : 0;
  const pipe = (m + bubbleSlots) * slot;
  const gradsBytes = (2 * nParams(arch)) / (t * p);
  const dp =
    d * cp > 1
      ? collective("allreduce", gradsBytes, d * cp, hw.net, hw.netAlpha).time
      : 0;
  const total = pipe + p2p + dp;
  const perGpu = hwFlops / n / total;
  return {
    gpus: n,
    microBatches: m,
    time: total,
    compute: m * perMbCompute,
    tensor: m * perMbTensor,
    context: m * perMbContext,
    bubble: bubbleSlots * slot,
    p2p,
    data: dp,
    hwFlopsPerGpu: perGpu,
    modelFlopsPerGpu: modelFlops / n / total,
    hfu: perGpu / hw.peak,
    mfu: modelFlops / n / total / hw.peak,
    tokensPerSecond: (B * s) / total,
  };
}

// ─── reliability ─────────────────────────────────────────────────────────
/** Young (1974): τ* = √(2 δ M). */
export function youngInterval(delta: number, mtbf: number): number {
  return Math.sqrt(2 * delta * mtbf);
}

/** First-order goodput: 1/(1 + δ/τ) · (1 − (τ/2 + R)/M). */
export function expectedGoodput(
  tau: number,
  delta: number,
  mtbf: number,
  restart: number,
): number {
  return (1 / (1 + delta / tau)) * Math.max(0, 1 - (tau / 2 + restart) / mtbf);
}

export type Segment = {
  kind: "work" | "checkpoint" | "lost" | "restart";
  start: number;
  end: number;
};

/**
 * Work in intervals of τ, each followed by a checkpoint of δ; failures at
 * exponential times (mean M) from mulberry32(seed); a failure loses the
 * work since the last checkpoint and costs R to restart.
 */
export function simulateGoodput(
  tau: number,
  delta: number,
  mtbf: number,
  restart: number,
  horizon: number,
  seed: number,
): { goodput: number; failures: number; segments: Segment[] } {
  const rng = mulberry32(seed);
  const nextFail = (now: number) => now + -Math.log(1 - rng()) * mtbf;
  let now = 0;
  let fail = nextFail(now);
  const segs: Segment[] = [];
  let failures = 0;
  while (now < horizon) {
    const endWork = now + tau;
    const endCkpt = endWork + delta;
    if (fail >= endCkpt) {
      segs.push({ kind: "work", start: now, end: endWork });
      segs.push({ kind: "checkpoint", start: endWork, end: endCkpt });
      now = endCkpt;
      continue;
    }
    segs.push({ kind: "lost", start: now, end: fail });
    failures++;
    const rEnd = fail + restart;
    segs.push({ kind: "restart", start: fail, end: rEnd });
    now = rEnd;
    fail = nextFail(now);
  }
  let useful = 0;
  for (const g of segs)
    if (g.kind === "work" && g.start < horizon)
      useful += Math.min(g.end, horizon) - g.start;
  return {
    goodput: useful / horizon,
    failures,
    segments: segs.filter((g) => g.start < horizon),
  };
}

/** M/D/1 mean wait (Pollaczek–Khinchine): load · service / (2 (1 − load)). */
export function md1Wait(load: number, service: number): number {
  if (load >= 1) return Infinity;
  return (load * service) / (2 * (1 - load));
}
