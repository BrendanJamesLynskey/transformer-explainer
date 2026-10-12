/**
 * Chapter 7's hero: an 8-block version of Part 1's model at
 * initialisation, pre-LN beside post-LN. Forward, block by block: the RMS
 * of the residual stream entering each block. Backward, from the top: the
 * norm of each block's weight-matrix gradients. Everything is computed by
 * the trainer's own forward and backward passes (model-grad.ts, which also
 * does post-LN) on the window "alice was"; the matrices start at the site's
 * std 0.02 or at 0.25 (= 1/√16, the scale of Xavier initialisation).
 * scripts/training/anim_b.py:depth_states builds the same list.
 */
import { lossAndGrads, forwardCached } from "@/lib/training/model-grad";
import { fromWeights, isMatrix, type ParamMap } from "@/lib/training/params";
import { initModelWeights } from "@/lib/transformer/init";
import type { ModelConfig } from "@/lib/transformer/model";
import { encodeChars } from "@/lib/training/data";

export const DEPTH = 8;
export const DEPTH_TEXT = "alice was";
export const DEPTH_STDS = [0.02, 0.25] as const;

export type DepthScale = { residualRms: number[]; gradNorm: number[] };

export type DepthState = {
  phase: "forward" | "backward";
  layer: number;
  /** [pre-LN, post-LN] */
  rms: [number[], number[]];
  grad: [number[], number[]];
};

const CONFIG: ModelConfig = {
  seq_len: 8,
  d_model: 16,
  n_heads: 2,
  d_ff: 32,
  n_blocks: DEPTH,
  vocab_size: 64,
  seed: 42,
};

function rms(m: number[][]): number {
  let s = 0;
  let n = 0;
  for (const r of m)
    for (const v of r) {
      s += v * v;
      n++;
    }
  return Math.sqrt(s / n);
}

/** Initial weights with every matrix rescaled to std `std` (the site's is 0.02). */
export function scaledInit(std: number): ParamMap {
  const p = fromWeights(initModelWeights(CONFIG));
  const out: ParamMap = {};
  for (const [k, t] of Object.entries(p))
    out[k] = isMatrix(t) ? t.map((r) => r.map((x) => x * (std / 0.02))) : t;
  return out;
}

const HIDDEN = /\.(attn\.W_|ffn\.W)/;

/** Residual-stream RMS into each block (and out of the last), and each block's gradient norm. */
export function depthScale(std: number, norm: "pre" | "post"): DepthScale {
  const p = scaledInit(std);
  const ids = encodeChars(DEPTH_TEXT);
  const arch = { ...CONFIG, norm };
  const c = forwardCached(p, arch, ids.slice(0, -1));
  const residualRms = c.blocks.map((b) => rms(b.x));
  residualRms.push(rms(c.xLast));
  const { grads } = lossAndGrads(p, arch, [ids.slice(0, -1)], [ids.slice(1)]);
  const gradNorm: number[] = [];
  for (let i = 0; i < DEPTH; i++) {
    let s = 0;
    for (const [k, g] of Object.entries(grads)) {
      if (!k.startsWith(`block_${i}.`) || !HIDDEN.test(k)) continue;
      for (const r of g as number[][]) for (const v of r) s += v * v;
    }
    gradNorm.push(Math.sqrt(s));
  }
  return { residualRms, gradNorm };
}

export function depthStates(std: number): DepthState[] {
  const pre = depthScale(std, "pre");
  const post = depthScale(std, "post");
  const out: DepthState[] = [];
  for (let k = 0; k <= DEPTH; k++)
    out.push({
      phase: "forward",
      layer: k,
      rms: [pre.residualRms.slice(0, k + 1), post.residualRms.slice(0, k + 1)],
      grad: [[], []],
    });
  for (let k = DEPTH - 1; k >= 0; k--)
    out.push({
      phase: "backward",
      layer: k,
      rms: [pre.residualRms, post.residualRms],
      grad: [pre.gradNorm.slice(k), post.gradNorm.slice(k)],
    });
  return out;
}

export function depthCaption(s: DepthState): string {
  if (s.phase === "forward") {
    const k = s.layer;
    const where = k === DEPTH ? "leaving block 8" : `entering block ${k + 1}`;
    return `Forward: the residual stream ${where} has RMS ${s.rms[0][k]!.toFixed(2)} (pre-LN) and ${s.rms[1][k]!.toFixed(2)} (post-LN).`;
  }
  const k = s.layer;
  return `Backward: block ${k + 1}'s weight gradients have norm ${s.grad[0][0]!.toPrecision(3)} (pre-LN) and ${s.grad[1][0]!.toPrecision(3)} (post-LN).`;
}
