/*
 * src/lib/training/optimizers.ts
 *
 * Operation: the optimisers (SGD, momentum, Nesterov momentum, Adam, AdamW)
 *            and gradient clipping by global norm.
 * Shapes:    every function walks a ParamMap; state tensors have the
 *            parameters' shapes.
 * Intuition: an optimiser turns a gradient into a step. Plain SGD steps
 *            straight downhill; momentum keeps a running sum of past
 *            gradients so it rolls through narrow valleys; Adam divides each
 *            coordinate by a running estimate of its gradient's size, so
 *            every parameter moves at roughly the learning rate; AdamW
 *            shrinks the weights directly instead of adding the decay to the
 *            gradient (which Adam would then rescale).
 *              SGD       p ← p − η g
 *              momentum  b ← μ b + g (b = g at the first step); p ← p − η b
 *              Nesterov  b as momentum; p ← p − η (g + μ b): PyTorch's
 *                        "lookahead-free" form (SGD(nesterov=True)), whose
 *                        stored p is the point momentum is about to carry
 *                        the weights to, so g is the gradient there
 *              Adam      m ← β₁ m + (1 − β₁) g;  v ← β₂ v + (1 − β₂) g²
 *                        p ← p − η m̂ / (√v̂ + ε),  m̂ = m/(1 − β₁ᵗ), v̂ = v/(1 − β₂ᵗ)
 *                        (L2: g ← g + λ p first, "coupled" decay)
 *              AdamW     p ← p (1 − η λ), then the Adam step
 *            Kingma & Ba, arXiv:1412.6980 (Adam); Loshchilov & Hutter,
 *            arXiv:1711.05101 (AdamW). Each step is checked against
 *            torch.optim in scripts/training/checks.py.
 *            Weight decay (λ) applies to matrices only, not to LayerNorm's
 *            γ, β or the biases.
 * MDX:       /training/05-optimisers.
 */
import type { Matrix, Vector } from "@/lib/transformer/types";

import { isMatrix, sumSquares, zerosLike, type ParamMap } from "./params";

export type OptimizerKind = "sgd" | "momentum" | "nesterov" | "adam" | "adamw";

export type OptimizerOptions = {
  kind: OptimizerKind;
  momentum?: number; // μ, default 0.9
  beta1?: number; // default 0.9
  beta2?: number; // default 0.999
  eps?: number; // default 1e-8
  weightDecay?: number; // λ, default 0 (Adam: L2; AdamW: decoupled)
};

export type OptimizerState = {
  opts: Required<OptimizerOptions>;
  t: number;
  /** momentum buffer (momentum, Nesterov) */
  buf: ParamMap | null;
  /** first moment m (Adam/AdamW) */
  m: ParamMap;
  /** second moment v (Adam/AdamW) */
  v: ParamMap;
};

/** A fresh optimiser state for these parameters. */
export function createOptimizer(
  params: ParamMap,
  options: OptimizerOptions,
): OptimizerState {
  const opts: Required<OptimizerOptions> = {
    momentum: 0.9,
    beta1: 0.9,
    beta2: 0.999,
    eps: 1e-8,
    weightDecay: 0,
    ...options,
  };
  return { opts, t: 0, buf: null, m: zerosLike(params), v: zerosLike(params) };
}

/** Apply `f(p, g, …state)` to every entry: the per-element update rule. */
function eachEntry(
  p: Matrix | Vector,
  fn: (row: Vector, i: number) => void,
): void {
  if (isMatrix(p)) p.forEach((row, i) => fn(row, i));
  else fn(p as Vector, -1);
}
const rowOf = (t: Matrix | Vector, i: number): Vector =>
  i < 0 ? (t as Vector) : (t as Matrix)[i]!;

/**
 * One optimiser step, updating `params` in place.
 *
 * @param lr  The learning rate for this step (from a schedule).
 */
export function optimizerStep(
  state: OptimizerState,
  params: ParamMap,
  grads: ParamMap,
  lr: number,
): void {
  const o = state.opts;
  state.t += 1;
  const t = state.t;
  const heavy = o.kind === "momentum" || o.kind === "nesterov";
  if (heavy && state.buf === null) state.buf = zerosLike(params);
  const bc1 = 1 - Math.pow(o.beta1, t);
  const bc2 = 1 - Math.pow(o.beta2, t);
  for (const k of Object.keys(params)) {
    const p = params[k]!;
    const wd = isMatrix(p) ? o.weightDecay : 0;
    eachEntry(p, (row, i) => {
      const g = rowOf(grads[k]!, i);
      if (o.kind === "sgd") {
        for (let j = 0; j < row.length; j++) row[j] = row[j]! - lr * g[j]!;
      } else if (heavy) {
        const b = rowOf(state.buf![k]!, i);
        for (let j = 0; j < row.length; j++) {
          b[j] = t === 1 ? g[j]! : o.momentum * b[j]! + g[j]!;
          // Nesterov steps along g + μ b: the gradient's own push plus
          // momentum's, as if the gradient were taken one step ahead
          row[j] =
            o.kind === "nesterov"
              ? row[j]! - lr * (g[j]! + o.momentum * b[j]!)
              : row[j]! - lr * b[j]!;
        }
      } else {
        const m = rowOf(state.m[k]!, i);
        const v = rowOf(state.v[k]!, i);
        for (let j = 0; j < row.length; j++) {
          let pj = row[j]!;
          let gj = g[j]!;
          if (o.kind === "adamw") pj = pj * (1 - lr * wd);
          else if (wd) gj = gj + wd * pj;
          m[j] = o.beta1 * m[j]! + (1 - o.beta1) * gj;
          v[j] = o.beta2 * v[j]! + (1 - o.beta2) * gj * gj;
          const mhat = m[j]! / bc1;
          const vhat = v[j]! / bc2;
          row[j] = pj - (lr * mhat) / (Math.sqrt(vhat) + o.eps);
        }
      }
    });
  }
}

/** ‖g‖ over every parameter: √(Σ g²), summed in parameter order. */
export function globalNorm(grads: ParamMap): number {
  let total = 0;
  for (const k of Object.keys(grads)) total += sumSquares(grads[k]!);
  return Math.sqrt(total);
}

/**
 * Clip by global norm, as torch.nn.utils.clip_grad_norm_: if ‖g‖ is above
 * `maxNorm`, scale every gradient by maxNorm / (‖g‖ + 1e-6). In place.
 *
 * @returns  the norm before clipping and the scale applied (1 = unclipped).
 */
export function clipByGlobalNorm(
  grads: ParamMap,
  maxNorm: number,
): { norm: number; scale: number } {
  const norm = globalNorm(grads);
  const scale = Math.min(1, maxNorm / (norm + 1e-6));
  if (scale < 1) {
    for (const k of Object.keys(grads)) {
      eachEntry(grads[k]!, (row) => {
        for (let j = 0; j < row.length; j++) row[j] = row[j]! * scale;
      });
    }
  }
  return { norm, scale };
}
