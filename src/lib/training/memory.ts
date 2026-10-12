/*
 * src/lib/training/memory.ts
 *
 * Operation: the bytes one device holds while training a decoder: weights,
 *            gradients, optimiser states (and an FP32 master copy), the
 *            activations saved for the backward pass, the logits; and the
 *            memory in use after each event of one training step.
 * Shapes:    breakdown(arch, recipe, recompute) → bytes by kind;
 *            timeline(arch, recipe, recompute) → MemEvent[].
 * Intuition: the weights are the small part. Adam keeps two numbers per
 *            weight, mixed precision keeps an FP32 master copy as well, and
 *            the forward pass leaves every layer's activations behind for
 *            the backward pass: their size grows with the batch and the
 *            sequence (and with the sequence squared in attention).
 *            Activation checkpointing keeps only each layer's input and
 *            recomputes the rest during the backward pass.
 * Sources:   ZeRO (Rajbhandari et al. 2020, arXiv:1910.02054, §3.1) for the
 *            16 bytes per parameter of mixed-precision Adam; Korthikanti et
 *            al. 2022 (arXiv:2205.05198, §4.1 eq. 1, §5) for activations.
 *            Port of scripts/training/memory.py (tests/unit/training/memory.test.ts).
 * MDX:       /training/10-memory.
 */

export type Arch = {
  name: string;
  /** layers */
  L: number;
  /** width (d_model) */
  h: number;
  /** attention heads */
  a: number;
  /** FFN width */
  f: number;
  /** vocabulary */
  V: number;
  /** sequence length */
  s: number;
  /** micro-batch (sequences) */
  b: number;
  ffn: "gelu" | "swiglu";
  norm: "layer" | "rms";
  tied: boolean;
  attnBias: boolean;
};

export type Recipe = {
  /** bytes per weight (4 = FP32, 2 = BF16/FP16) */
  weightBytes: number;
  gradBytes: number;
  optimizer: "adam" | "momentum" | "sgd";
  /** keep an FP32 master copy when the weights are 16-bit */
  master: boolean;
};

export type Recompute = "none" | "selective" | "full";

export const OPT_STATE_BYTES = { adam: 8, momentum: 4, sgd: 0 } as const;

/** LLaMA 7B's shape (Touvron et al. 2023, arXiv:2302.13971, Table 2: 6.7B). */
export const LLAMA_7B: Arch = {
  name: "LLaMA 7B",
  L: 32,
  h: 4096,
  a: 32,
  f: 11008,
  V: 32000,
  s: 4096,
  b: 1,
  ffn: "swiglu",
  norm: "rms",
  tied: false,
  attnBias: false,
};

/** Part 1's model, at the default run's batch of 16 windows. */
export const PART1_MEM: Arch = {
  name: "Part 1's model",
  L: 2,
  h: 16,
  a: 2,
  f: 32,
  V: 64,
  s: 8,
  b: 16,
  ffn: "gelu",
  norm: "layer",
  tied: true,
  attnBias: false,
};

/** Mixed-precision Adam: BF16 weights and gradients, FP32 master, m and v. */
export const MIXED_ADAM: Recipe = {
  weightBytes: 2,
  gradBytes: 2,
  optimizer: "adam",
  master: true,
};

/** Everything in FP32 with Adam. */
export const FP32_ADAM: Recipe = {
  weightBytes: 4,
  gradBytes: 4,
  optimizer: "adam",
  master: false,
};

/** The exact parameter count of the architecture. */
export function paramCount(c: Arch): number {
  const { h, f, L, V } = c;
  const norm = c.norm === "rms" ? h : 2 * h;
  const attn = 4 * h * h + (c.attnBias ? 4 * h : 0);
  const ffn = c.ffn === "swiglu" ? 3 * h * f : 2 * h * f + f + h;
  const perLayer = attn + ffn + 2 * norm;
  const emb = V * h * (c.tied ? 1 : 2);
  return L * perLayer + emb + norm;
}

export type BytesPerParam = {
  weights: number;
  grads: number;
  optimizer: number;
  master: number;
};

/** Bytes per parameter of each kind. */
export function bytesPerParam(r: Recipe): BytesPerParam {
  return {
    weights: r.weightBytes,
    grads: r.gradBytes,
    optimizer: OPT_STATE_BYTES[r.optimizer],
    master: r.master && r.weightBytes < 4 ? 4 : 0,
  };
}

/**
 * Saved activations per layer, in bytes (16-bit activations):
 * s b h (34 + 5 a s / h) without recomputation (Korthikanti et al. 2022,
 * eq. 1), 34 s b h with selective recomputation, 2 s b h with full.
 */
export function activationsPerLayer(c: Arch, recompute: Recompute): number {
  const { s, b, h, a } = c;
  if (recompute === "none") return s * b * h * (34 + (5 * a * s) / h);
  if (recompute === "selective") return 34 * s * b * h;
  return 2 * s * b * h;
}

export type Breakdown = {
  params: number;
  weights: number;
  grads: number;
  optimizer: number;
  master: number;
  activations: number;
  logits: number;
  /** one layer's activations re-created during the backward pass */
  recompute: number;
  static: number;
  peak: number;
  bytesPerParam: number;
};

/** Bytes of each kind (gradients, activations and logits at their largest), and the peak over one step. */
export function breakdown(
  c: Arch,
  r: Recipe,
  recompute: Recompute = "none",
): Breakdown {
  const n = paramCount(c);
  const bpp = bytesPerParam(r);
  const act = c.L * activationsPerLayer(c, recompute);
  const logits = c.s * c.b * c.V * 6;
  const transient = recompute !== "none" ? activationsPerLayer(c, "none") : 0;
  const weights = n * bpp.weights;
  const grads = n * bpp.grads;
  const optimizer = n * bpp.optimizer;
  const master = n * bpp.master;
  const stat = weights + grads + optimizer + master;
  // the largest total during the step (timeline): gradients are written
  // during the backward pass, as activations are freed
  let peak = 0;
  for (const e of timeline(c, r, recompute)) peak = Math.max(peak, e.total);
  return {
    params: n,
    weights,
    grads,
    optimizer,
    master,
    activations: act,
    logits,
    recompute: transient,
    static: stat,
    peak,
    bytesPerParam: bpp.weights + bpp.grads + bpp.optimizer + bpp.master,
  };
}

export type MemEvent = {
  event:
    | "start"
    | "forward"
    | "loss"
    | "head"
    | "recompute"
    | "backward"
    | "step";
  layer: number;
  weights: number;
  optimizer: number;
  master: number;
  grads: number;
  activations: number;
  perLayer: number[];
  transient: number;
  logits: number;
  total: number;
};

/**
 * Memory in use after each event of one training step: the forward pass
 * saves each layer's activations; the loss adds the logits; the backward
 * pass allocates each layer's gradients and frees its activations (with
 * recomputation, first re-creating that layer's full activations).
 */
export function timeline(
  c: Arch,
  r: Recipe,
  recompute: Recompute = "none",
): MemEvent[] {
  const n = paramCount(c);
  const bpp = bytesPerParam(r);
  const L = c.L;
  const full = activationsPerLayer(c, "none");
  const kept = activationsPerLayer(c, recompute);
  const logits = c.s * c.b * c.V * 6;
  const head = c.V * c.h * bpp.grads;
  const layerGrad = (n * bpp.grads - head) / L;
  const base = {
    weights: n * bpp.weights,
    optimizer: n * bpp.optimizer,
    master: n * bpp.master,
  };
  const acts = new Array<number>(L).fill(0);
  let grads = 0;
  const out: MemEvent[] = [];
  const snap = (
    event: MemEvent["event"],
    layer: number,
    extra = 0,
    lg = 0,
  ): void => {
    let sumActs = 0;
    for (const v of acts) sumActs += v;
    const total =
      base.weights +
      base.optimizer +
      base.master +
      grads +
      sumActs +
      extra +
      lg;
    out.push({
      event,
      layer,
      ...base,
      grads,
      activations: sumActs,
      perLayer: acts.slice(),
      transient: extra,
      logits: lg,
      total,
    });
  };
  snap("start", -1);
  for (let i = 0; i < L; i++) {
    acts[i] = kept;
    snap("forward", i);
  }
  snap("loss", L, 0, logits);
  grads += head;
  snap("head", L, 0, logits);
  for (let i = L - 1; i >= 0; i--) {
    if (recompute !== "none") snap("recompute", i, full);
    grads += layerGrad;
    acts[i] = 0;
    snap("backward", i);
  }
  snap("step", -1);
  return out;
}

/** Bytes as "1.2 GB" / "340 MB" / "12 kB" (decimal units). */
export function fmtBytes(x: number): string {
  const units = ["B", "kB", "MB", "GB", "TB"];
  let v = x;
  let u = 0;
  while (v >= 1000 && u < units.length - 1) {
    v /= 1000;
    u++;
  }
  return `${v >= 100 || u === 0 ? v.toFixed(0) : v.toFixed(1)} ${units[u]}`;
}
