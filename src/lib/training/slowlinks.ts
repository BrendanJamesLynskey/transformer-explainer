/*
 * src/lib/training/slowlinks.ts
 *
 * Operation: Part 1's model trained by W simulated data-parallel workers,
 *            synchronised four ways: dense (AdamW on the all-reduced
 *            gradient), top-k with error feedback (Deep Gradient
 *            Compression's sparse momentum SGD), DeMo (DCT of each worker's
 *            momentum in chunks, top-k coefficients sent, the rest kept in
 *            the momentum, sign update) and DiLoCo (each worker alone for H
 *            AdamW steps, then an outer Nesterov step on the averaged change).
 * Shapes:    ParamMap in, ParamMap out; each tensor is flattened row-major
 *            and cut into chunks of up to 64 values for the DCT.
 * Intuition: a dense all-reduce sends every gradient value every step. The
 *            compressors send 1 value in 16 (and where it sits), keep the
 *            rest locally and add it back next time (error feedback), so
 *            nothing is lost, only late. DiLoCo sends everything, but only
 *            every H steps.
 * Sources:   Lin et al. 2018 (DGC, arXiv:1712.01887); Stich et al. 2018
 *            (arXiv:1809.07599); Karimireddy et al. 2019 (arXiv:1901.09847);
 *            Peng et al. 2024 (DeMo, arXiv:2411.19870, Algorithm 1);
 *            Douillard et al. 2023 (DiLoCo, arXiv:2311.08105, Algorithm 1).
 *            Port of scripts/training/slowlinks.py, matched to its fixtures
 *            (tests/unit/training/slowlinks.test.ts).
 * MDX:       /training/13-slow-links.
 */
import type { ModelConfig } from "@/lib/transformer/model";
import { initModelWeights } from "@/lib/transformer/init";
import { mulberry32 } from "@/lib/transformer/random";

import { sampleBatch, splitIds, valBatch } from "./data";
import { lossAndGrads, lossOnly } from "./model-grad";
import {
  clipByGlobalNorm,
  createOptimizer,
  optimizerStep,
  type OptimizerState,
} from "./optimizers";
import {
  cloneParams,
  fromWeights,
  isMatrix,
  type ParamMap,
  type Tensor,
} from "./params";
import { lrAt } from "./schedules";

export const KEEP = 16;
export const CHUNK = 64;
export const BETA_DEMO = 0.999;
export const MU_TOPK = 0.9;
export const OUTER_LR = 0.7;
export const OUTER_MU = 0.9;
export const WD = 0.01;
export const VALUE_BYTES = 2;

export type Method = "dense" | "topk" | "demo" | "diloco";

// ─── flat views of tensors ───────────────────────────────────────────────
/** A tensor's entries, row-major. */
export function flat(t: Tensor): number[] {
  return isMatrix(t) ? t.flat() : (t as number[]).slice();
}

/** Write row-major values back into a tensor of the same shape (in place). */
export function setFlat(t: Tensor, values: number[]): void {
  if (isMatrix(t)) {
    const cols = t[0]!.length;
    t.forEach((row, i) => {
      for (let j = 0; j < cols; j++) row[j] = values[i * cols + j]!;
    });
  } else {
    const v = t as number[];
    for (let j = 0; j < v.length; j++) v[j] = values[j]!;
  }
}

// ─── DCT and chunks ──────────────────────────────────────────────────────
/** Orthonormal DCT-II: D[k][i] = √(2/n) cos(π (2i + 1) k / 2n), row 0 / √2. */
export function dctMatrix(n: number): number[][] {
  const d: number[][] = [];
  for (let k = 0; k < n; k++) {
    const row: number[] = [];
    for (let i = 0; i < n; i++) {
      let v =
        Math.cos((Math.PI * (2 * i + 1) * k) / (2 * n)) * Math.sqrt(2 / n);
      if (k === 0) v = v / Math.sqrt(2);
      row.push(v);
    }
    d.push(row);
  }
  return d;
}

const DCT_CACHE = new Map<number, number[][]>();
function dct(n: number): number[][] {
  let d = DCT_CACHE.get(n);
  if (!d) {
    d = dctMatrix(n);
    DCT_CACHE.set(n, d);
  }
  return d;
}

/** The largest divisor of `size` not above `target`. */
export function chunkLen(size: number, target = CHUNK): number {
  let best = 1;
  for (let d = 1; d <= Math.min(size, target); d++)
    if (size % d === 0) best = d;
  return best;
}

/** Python's round() (half to even) of m / KEEP, at least 1. */
export function keepCount(m: number): number {
  const x = m / KEEP;
  const f = Math.floor(x);
  const r = x - f === 0.5 ? (f % 2 === 0 ? f : f + 1) : Math.round(x);
  return Math.max(1, r);
}

/** coef[k] = Σ_i x[i] D[k][i] */
export function dctForward(x: number[], D: number[][]): number[] {
  return D.map((row) => {
    let s = 0;
    for (let i = 0; i < x.length; i++) s += x[i]! * row[i]!;
    return s;
  });
}

/** x[i] = Σ_k coef[k] D[k][i] */
export function dctInverse(c: number[], D: number[][]): number[] {
  const n = c.length;
  const out = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let k = 0; k < n; k++) s += c[k]! * D[k]![i]!;
    out[i] = s;
  }
  return out;
}

/** Indices of the k largest |a|, ties to the lower index, ascending. */
export function topkIdx(a: number[], k: number): number[] {
  const order = a.map((_, i) => i);
  order.sort((i, j) => Math.abs(a[j]!) - Math.abs(a[i]!) || i - j);
  return order.slice(0, k).sort((i, j) => i - j);
}

// ─── bytes ───────────────────────────────────────────────────────────────
export type Bytes = {
  params: number;
  payload: number;
  wire: number;
  denseWire: number;
};

/** What one worker originates (payload) and sends in the ring (wire) per step. */
export function bytesFor(
  method: Method,
  sizes: number[],
  W: number,
  H = 1,
): Bytes {
  const P = sizes.reduce((a, b) => a + b, 0);
  const denseWire = ((2 * (W - 1)) / W) * P * VALUE_BYTES;
  if (method === "dense") {
    return { params: P, payload: P * VALUE_BYTES, wire: denseWire, denseWire };
  }
  if (method === "diloco") {
    return {
      params: P,
      payload: (P * VALUE_BYTES) / H,
      wire: denseWire / H,
      denseWire,
    };
  }
  let k = 0;
  let idxBytes = 2;
  if (method === "topk") {
    for (const n of sizes) k += keepCount(n);
  } else {
    idxBytes = 1;
    for (const n of sizes) {
      const m = chunkLen(n);
      k += (n / m) * keepCount(m);
    }
  }
  const payload = k * (VALUE_BYTES + idxBytes);
  return { params: P, payload, wire: (W - 1) * payload, denseWire };
}

// ─── DeMo ────────────────────────────────────────────────────────────────
export type DemoWorkerRecord = {
  grad: number[];
  momentum: number[];
  coef: number[];
  idx: number[];
  sent: number[];
  residual: number[];
};
export type DemoRecord = {
  name: string;
  chunk: number;
  k: number;
  workers: DemoWorkerRecord[];
  meanCoef: number[];
  aggregate: number[];
};

/** Per-worker momenta (flat, by tensor). */
export type DemoState = { beta: number; M: Record<string, number[]>[] };

export function createDemo(params: ParamMap, W: number): DemoState {
  const M: Record<string, number[]>[] = [];
  for (let w = 0; w < W; w++) {
    const m: Record<string, number[]> = {};
    for (const k of Object.keys(params))
      m[k] = new Array<number>(flat(params[k]!).length).fill(0);
    M.push(m);
  }
  return { beta: BETA_DEMO, M };
}

/**
 * One DeMo sync: every worker folds its gradient into its momentum, DCTs
 * each chunk, sends the top-k coefficients and keeps the rest; the sent
 * coefficients are averaged per position over the workers that sent them
 * and inverse-transformed. Returns M* per tensor (flat), and the record of
 * the first chunk of tensor `record`.
 */
export function demoSync(
  st: DemoState,
  grads: ParamMap[],
  record?: string,
): { agg: Record<string, number[]>; rec: DemoRecord | null } {
  const agg: Record<string, number[]> = {};
  let rec: DemoRecord | null = null;
  const W = grads.length;
  for (const name of Object.keys(grads[0]!)) {
    const gFlat = grads.map((g) => flat(g[name]!));
    const n = gFlat[0]!.length;
    const m = chunkLen(n);
    const k = keepCount(m);
    const D = dct(m);
    const chunks = n / m;
    const total = Array.from({ length: chunks }, () =>
      new Array<number>(m).fill(0),
    );
    const count = Array.from({ length: chunks }, () =>
      new Array<number>(m).fill(0),
    );
    if (record === name)
      rec = { name, chunk: m, k, workers: [], meanCoef: [], aggregate: [] };
    for (let w = 0; w < W; w++) {
      const Mw = st.M[w]![name]!;
      const mom = Mw.map((x, i) => st.beta * x + gFlat[w]![i]!);
      const resid = new Array<number>(n);
      for (let c = 0; c < chunks; c++) {
        const ch = mom.slice(c * m, (c + 1) * m);
        const coef = dctForward(ch, D);
        const idx = topkIdx(coef, k);
        const q = new Array<number>(m).fill(0);
        const sent = idx.map((i) => coef[i]!);
        idx.forEach((i, j) => {
          q[i] = sent[j]!;
          total[c]![i] = total[c]![i]! + sent[j]!;
          count[c]![i] = count[c]![i]! + 1;
        });
        const back = dctInverse(q, D);
        for (let i = 0; i < m; i++) resid[c * m + i] = ch[i]! - back[i]!;
        if (rec && rec.name === name && c === 0)
          rec.workers.push({
            grad: gFlat[w]!.slice(0, m),
            momentum: ch,
            coef,
            idx,
            sent,
            residual: resid.slice(0, m),
          });
      }
      st.M[w]![name] = resid;
    }
    const out = new Array<number>(n);
    for (let c = 0; c < chunks; c++) {
      const mean = total[c]!.map((s, i) =>
        count[c]![i]! > 0 ? s / Math.max(count[c]![i]!, 1) : 0,
      );
      const back = dctInverse(mean, D);
      for (let i = 0; i < m; i++) out[c * m + i] = back[i]!;
      if (rec && c === 0 && rec.name === name) {
        rec.meanCoef = mean;
        rec.aggregate = back.slice();
      }
    }
    agg[name] = out;
  }
  return { agg, rec };
}

// ─── top-k with error feedback (DGC) ─────────────────────────────────────
export type TopKState = {
  mu: number;
  u: Record<string, number[]>[];
  v: Record<string, number[]>[];
};

export function createTopK(params: ParamMap, W: number): TopKState {
  const mk = () => {
    const o: Record<string, number[]> = {};
    for (const k of Object.keys(params))
      o[k] = new Array<number>(flat(params[k]!).length).fill(0);
    return o;
  };
  return {
    mu: MU_TOPK,
    u: Array.from({ length: W }, mk),
    v: Array.from({ length: W }, mk),
  };
}

export function topkSync(
  st: TopKState,
  grads: ParamMap[],
): Record<string, number[]> {
  const agg: Record<string, number[]> = {};
  const W = grads.length;
  for (const name of Object.keys(grads[0]!)) {
    const n = st.u[0]![name]!.length;
    const k = keepCount(n);
    const total = new Array<number>(n).fill(0);
    for (let w = 0; w < W; w++) {
      const g = flat(grads[w]![name]!);
      const u = st.u[w]![name]!.map((x, i) => st.mu * x + g[i]!);
      const v = st.v[w]![name]!.map((x, i) => x + u[i]!);
      const idx = topkIdx(v, k);
      const sentMask = new Array<boolean>(n).fill(false);
      for (const i of idx) {
        total[i] = total[i]! + v[i]!;
        sentMask[i] = true;
      }
      st.v[w]![name] = v.map((x, i) => (sentMask[i] ? 0 : x));
      st.u[w]![name] = u.map((x, i) => (sentMask[i] ? 0 : x));
    }
    agg[name] = total.map((x) => x / W);
  }
  return agg;
}

// ─── the simulated workers ───────────────────────────────────────────────
export type WorkerConfig = {
  method: Method;
  workers: number;
  windows: number;
  steps: number;
  lr: number;
  warmup: number;
  minRatio: number;
  clip: number;
  H: number;
  dataSeed: number;
  evalEvery: number;
};

export const DEFAULT_WORKERS: WorkerConfig = {
  method: "dense",
  workers: 4,
  windows: 4,
  steps: 2000,
  lr: 0.01,
  warmup: 50,
  minRatio: 0.1,
  clip: 1.0,
  H: 10,
  dataSeed: 1,
  evalEvery: 50,
};

/** Mean of the workers' tensors, added in worker order. */
function meanParams(ds: ParamMap[]): ParamMap {
  const out = cloneParams(ds[0]!);
  for (const k of Object.keys(out)) {
    const acc = flat(ds[0]![k]!);
    for (const d of ds.slice(1)) {
      const f = flat(d[k]!);
      for (let i = 0; i < acc.length; i++) acc[i] = acc[i]! + f[i]!;
    }
    setFlat(
      out[k]!,
      acc.map((x) => x / ds.length),
    );
  }
  return out;
}

export type WorkerRun = {
  cfg: WorkerConfig;
  params: ParamMap;
  t: number;
  evals: { step: number; val: number }[];
  losses: number[];
  step: () => number;
  evaluate: () => number;
};

/**
 * A run of `cfg.method` that steps on demand (the chapter's live run uses
 * it a few steps per frame); `step()` returns the mean training loss.
 */
export function createWorkerRun(
  cfg: WorkerConfig,
  model: ModelConfig,
  corpusIds: number[],
): WorkerRun {
  const S = model.seq_len;
  const W = cfg.workers;
  const per = cfg.windows;
  const params = fromWeights(initModelWeights(model));
  const rng = mulberry32(cfg.dataSeed);
  const { train, val } = splitIds(corpusIds);
  const vs = valBatch(val, S);
  const sched = {
    kind: "cosine" as const,
    lr: cfg.lr,
    warmup: cfg.warmup,
    minRatio: cfg.minRatio,
    total: cfg.steps,
  };
  const adam = (p: ParamMap): OptimizerState =>
    createOptimizer(p, {
      kind: "adamw",
      beta1: 0.9,
      beta2: 0.99,
      eps: 1e-8,
      weightDecay: WD,
    });
  const opt = cfg.method === "dense" ? adam(params) : null;
  let local: ParamMap[] = [];
  let opts: OptimizerState[] = [];
  let outerBuf: ParamMap | null = null;
  if (cfg.method === "diloco") {
    local = Array.from({ length: W }, () => cloneParams(params));
    opts = local.map((p) => adam(p));
  }
  const demo = cfg.method === "demo" ? createDemo(params, W) : null;
  const topk = cfg.method === "topk" ? createTopK(params, W) : null;
  const run: WorkerRun = {
    cfg,
    params,
    t: 0,
    evals: [],
    losses: [],
    evaluate: () => lossOnly(params, model, vs.inputs, vs.targets),
    step: () => {
      const t = run.t;
      const b = sampleBatch(train, W * per, S, rng);
      const lr = lrAt(t, sched);
      const slice = (w: number) => ({
        x: b.inputs.slice(w * per, (w + 1) * per),
        y: b.targets.slice(w * per, (w + 1) * per),
      });
      let stepLoss = 0;
      if (cfg.method === "diloco") {
        for (let w = 0; w < W; w++) {
          const { x, y } = slice(w);
          const r = lossAndGrads(local[w]!, model, x, y);
          clipByGlobalNorm(r.grads, cfg.clip);
          optimizerStep(opts[w]!, local[w]!, r.grads, lr);
          stepLoss += r.loss / W;
        }
        if ((t + 1) % cfg.H === 0) {
          const deltas = local.map((lw) => {
            const d = cloneParams(params);
            for (const k of Object.keys(d)) {
              const a = flat(params[k]!);
              const c = flat(lw[k]!);
              setFlat(
                d[k]!,
                a.map((x, i) => x - c[i]!),
              );
            }
            return d;
          });
          const delta = meanParams(deltas);
          if (outerBuf === null) outerBuf = cloneParams(delta);
          else {
            for (const k of Object.keys(delta)) {
              const ob = flat(outerBuf[k]!);
              const dk = flat(delta[k]!);
              setFlat(
                outerBuf[k]!,
                ob.map((x, i) => OUTER_MU * x + dk[i]!),
              );
            }
          }
          for (const k of Object.keys(params)) {
            const p = flat(params[k]!);
            const dk = flat(delta[k]!);
            const ob = flat(outerBuf[k]!);
            setFlat(
              params[k]!,
              p.map((x, i) => x - OUTER_LR * (dk[i]! + OUTER_MU * ob[i]!)),
            );
          }
          // every worker restarts from the shared weights; the inner
          // optimisers keep their moments (they are keyed by name)
          local = Array.from({ length: W }, () => cloneParams(params));
        }
      } else {
        const grads: ParamMap[] = [];
        for (let w = 0; w < W; w++) {
          const { x, y } = slice(w);
          const r = lossAndGrads(params, model, x, y);
          stepLoss += r.loss;
          grads.push(r.grads);
        }
        stepLoss = stepLoss / W;
        if (cfg.method === "dense") {
          const g = meanParams(grads);
          clipByGlobalNorm(g, cfg.clip);
          optimizerStep(opt!, params, g, lr);
        } else {
          for (const g of grads) clipByGlobalNorm(g, cfg.clip);
          const agg =
            cfg.method === "demo"
              ? demoSync(demo!, grads).agg
              : topkSync(topk!, grads);
          for (const k of Object.keys(params)) {
            const p = params[k]!;
            const decay = isMatrix(p) ? WD : 0;
            const pf = flat(p);
            const a = agg[k]!;
            setFlat(
              p,
              cfg.method === "demo"
                ? pf.map((x, i) => x - lr * (Math.sign(a[i]!) + decay * x))
                : pf.map((x, i) => x * (1 - lr * decay) - lr * a[i]!),
            );
          }
        }
      }
      run.t += 1;
      run.losses.push(stepLoss);
      return stepLoss;
    },
  };
  return run;
}

/** Run to the end, evaluating every `evalEvery` steps (as slowlinks.py). */
export function trainWorkers(
  cfg: WorkerConfig,
  model: ModelConfig,
  corpusIds: number[],
): WorkerRun {
  const run = createWorkerRun(cfg, model, corpusIds);
  for (let t = 0; t < cfg.steps; t++) {
    if (t % cfg.evalEvery === 0)
      run.evals.push({ step: t, val: run.evaluate() });
    run.step();
  }
  run.evals.push({ step: cfg.steps, val: run.evaluate() });
  return run;
}
