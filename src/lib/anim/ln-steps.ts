/**
 * The LayerNorm-and-residuals animation (chapter 05) as a list of model
 * states, for one position p of block 0. The block has two sub-layers, and
 * each one goes through the same eight steps:
 *
 *   input     the residual stream x arrives (x₀ for attention, h for the FFN)
 *   mean      μ = (1/d) Σ xᵢ
 *   var       σ² = (1/d) Σ (xᵢ − μ)²
 *   centre    x − μ                     (the distribution shifts to 0)
 *   scale     x̂ = (x − μ) / √(σ² + ε)   (and is rescaled to unit variance)
 *   affine    LN(x) = γ ⊙ x̂ + β         (the learned gain and bias)
 *   sublayer  Δ = Attn(LN₁(x)) or FFN(LN₂(h)) at position p
 *   residual  the stream carries on with x + Δ
 *
 * The numbers are a real run of block 0 (`block` from the model) with the
 * reader's γ and β in both LayerNorms; the model's own values are γ = 1,
 * β = 0. The mean, variance and normalised vector are recomputed here in
 * the same order as `layernorm`, so the `affine` step equals the block's
 * LN output exactly (`tests/unit/anim-steps-b.test.ts`).
 */
import { configFor, type ComputeParams } from "@/lib/compute/traces";
import { multiHeadAttention } from "@/lib/transformer/attention";
import { block } from "@/lib/transformer/block";
import {
  positionalEncoding,
  tokenEmbedding,
} from "@/lib/transformer/embeddings";
import { ffn } from "@/lib/transformer/ffn";
import { initModelWeights } from "@/lib/transformer/init";
import { LAYERNORM_EPS } from "@/lib/transformer/layernorm";
import { addMat } from "@/lib/transformer/tensor";
import { ALPHABET, encode } from "@/lib/transformer/tokenizer";
import { emptyBlockTrace } from "@/lib/transformer/trace";

import { norm, showChar, sig } from "./format";

/** Block 0 with every intermediate the animation shows, per position. */
export type ResidualTrace = {
  tokens: string[];
  /** The stream into the block, x₀ = E + P. */
  x: number[][];
  ln1: number[][];
  /** The attention sub-layer's output (before the residual add). */
  attn: number[][];
  /** h = x + Attn(LN₁(x)). */
  h: number[][];
  ln2: number[][];
  /** The FFN sub-layer's output (before the residual add). */
  ffn: number[][];
  /** y = h + FFN(LN₂(h)): the block's output. */
  y: number[][];
  gamma: number;
  beta: number;
};

/**
 * Run block 0 on `p.text` with γ and β set to `gamma` and `beta` in both
 * of its LayerNorms (the model initialises them to 1 and 0).
 */
export function residualTrace(
  p: ComputeParams,
  gamma = 1,
  beta = 0,
): ResidualTrace {
  const w = initModelWeights(configFor(p, { n_blocks: 1 }));
  const D = p.dModel;
  const ln = () => ({
    gamma: new Array<number>(D).fill(gamma),
    beta: new Array<number>(D).fill(beta),
  });
  const b0 = { ...w.blocks[0]!, ln1: ln(), ln2: ln() };
  const ids = encode(p.text, p.seqLen);
  const x = addMat(
    tokenEmbedding(ids, w.tok_emb),
    positionalEncoding(p.seqLen, D),
  );
  const tr = emptyBlockTrace();
  const y = block(x, b0, p.nHeads, tr);
  return {
    tokens: ids.map((id) => ALPHABET[id] ?? "?"),
    x,
    ln1: tr.ln1,
    // the sub-layers again, on the traced LN outputs: the same deltas the
    // block added (it keeps only the sums)
    attn: multiHeadAttention(tr.ln1, b0.attn, p.nHeads),
    h: tr.attnOut,
    ln2: tr.ln2,
    ffn: ffn(tr.ln2, b0.ffn),
    y,
    gamma,
    beta,
  };
}

export type LnPhase =
  | "input"
  | "mean"
  | "var"
  | "centre"
  | "scale"
  | "affine"
  | "sublayer"
  | "residual";

export const LN_PHASES: readonly LnPhase[] = [
  "input",
  "mean",
  "var",
  "centre",
  "scale",
  "affine",
  "sublayer",
  "residual",
];

export type LnState = {
  /** 1: LN₁ and attention; 2: LN₂ and the FFN. */
  sub: 1 | 2;
  phase: LnPhase;
  pos: number;
  /** The stream into this sub-layer at `pos`. */
  x: number[];
  mean: number | null;
  variance: number | null;
  /** The vector the distribution shows now: x, x − μ, x̂ or LN(x). */
  values: number[];
  /** LN(x) = γ ⊙ x̂ + β (from affine on). */
  ln: number[] | null;
  /** The sub-layer's output Δ (from sublayer on). */
  delta: number[] | null;
  /** x + Δ (residual). */
  out: number[] | null;
  gamma: number;
  beta: number;
};

/** All states for position `pos`: LN₁ + attention, then LN₂ + FFN. */
export function lnStates(tr: ResidualTrace, pos: number): LnState[] {
  const S = tr.x.length;
  const p = Math.max(0, Math.min(S - 1, pos));
  const out: LnState[] = [];
  const subs = [
    { sub: 1 as const, x: tr.x[p]!, delta: tr.attn[p]!, ln: tr.ln1[p]! },
    { sub: 2 as const, x: tr.h[p]!, delta: tr.ffn[p]!, ln: tr.ln2[p]! },
  ];
  for (const { sub, x, delta } of subs) {
    const n = x.length;
    // the same operations, in the same order, as layernorm.ts
    let mean = 0;
    for (let i = 0; i < n; i++) mean += x[i]!;
    mean /= n;
    let varSum = 0;
    for (let i = 0; i < n; i++) {
      const d = x[i]! - mean;
      varSum += d * d;
    }
    const variance = varSum / n;
    const invStd = 1 / Math.sqrt(variance + LAYERNORM_EPS);
    const centred = x.map((v) => v - mean);
    const scaled = x.map((v) => (v - mean) * invStd);
    const ln = x.map((v) => (v - mean) * invStd * tr.gamma + tr.beta);
    const sum = x.map((v, k) => v + delta[k]!);
    const base = {
      sub,
      pos: p,
      x,
      gamma: tr.gamma,
      beta: tr.beta,
      ln: null,
      delta: null,
      out: null,
    };
    const stats = { mean, variance };
    out.push({
      ...base,
      phase: "input",
      mean: null,
      variance: null,
      values: x,
    });
    out.push({ ...base, phase: "mean", mean, variance: null, values: x });
    out.push({ ...base, ...stats, phase: "var", values: x });
    out.push({ ...base, ...stats, phase: "centre", values: centred });
    out.push({ ...base, ...stats, phase: "scale", values: scaled });
    out.push({ ...base, ...stats, phase: "affine", values: ln, ln });
    out.push({ ...base, ...stats, phase: "sublayer", values: ln, ln, delta });
    out.push({
      ...base,
      ...stats,
      phase: "residual",
      values: ln,
      ln,
      delta,
      out: sum,
    });
  }
  return out;
}

/** The KaTeX term each phase highlights. */
export const LN_HL: Record<LnPhase, string> = {
  input: "x",
  mean: "mu",
  var: "sig",
  centre: "hat",
  scale: "hat",
  affine: "gb",
  sublayer: "sub",
  residual: "res",
};

/**
 * The KaTeX term to highlight for a state: the residual line of the
 * equation has one term per sub-layer, so sub-layer 2 uses the `…2` keys.
 */
export function lnHl(s: LnState): string {
  const k = LN_HL[s.phase];
  return s.sub === 2 && (k === "x" || k === "sub" || k === "res") ? `${k}2` : k;
}

/** The population mean and standard deviation of `v` (for the captions). */
export function meanStd(v: readonly number[]): {
  mean: number;
  std: number;
} {
  let m = 0;
  for (const x of v) m += x;
  m /= v.length;
  let s = 0;
  for (const x of v) s += (x - m) * (x - m);
  return { mean: m, std: Math.sqrt(s / v.length) };
}

/** The caption for one state (`tokens`: the input's characters). */
export function lnCaption(s: LnState, tokens: readonly string[]): string {
  const ln = s.sub === 1 ? "LN₁" : "LN₂";
  const sublayer = s.sub === 1 ? "attention" : "the FFN";
  const xName = s.sub === 1 ? "x" : "h";
  const at = `Sub-layer ${s.sub}, position ${s.pos} '${showChar(tokens[s.pos] ?? "?")}'`;
  switch (s.phase) {
    case "input":
      return `${at}: the residual stream ${xName} arrives, ${s.x.length} numbers; ${ln} normalises a copy before ${sublayer} reads it.`;
    case "mean":
      return `${ln}: the mean of the ${s.x.length} numbers is μ = ${sig(s.mean!)}.`;
    case "var":
      return `${ln}: the variance is σ² = ${sig(s.variance!)}, so the spread is σ = ${sig(Math.sqrt(s.variance!))}.`;
    case "centre":
      return `${ln}: subtract μ; the numbers now average 0 (the distribution shifts by ${sig(-s.mean!)}).`;
    case "scale": {
      const k = 1 / Math.sqrt(s.variance! + LAYERNORM_EPS);
      return `${ln}: divide by √(σ² + ε) = ${sig(1 / k)}; the spread is now 1 (every number scaled by ${sig(k)}).`;
    }
    case "affine": {
      const ident = s.gamma === 1 && s.beta === 0;
      const ms = meanStd(s.ln!);
      // the mean of a centred vector is 0 up to rounding (≈ 1e−17): say 0
      if (Math.abs(ms.mean) < 1e-9) ms.mean = 0;
      return ident
        ? `${ln}: γ ⊙ x̂ + β with the model's γ = 1, β = 0 changes nothing at initialisation; training learns them. Mean ${sig(ms.mean)}, spread ${sig(ms.std)}.`
        : `${ln}: γ ⊙ x̂ + β with γ = ${sig(s.gamma)}, β = ${sig(s.beta)}: the spread becomes ${sig(ms.std)} and the mean ${sig(ms.mean)}.`;
    }
    case "sublayer":
      return `${s.sub === 1 ? "Attention" : "The FFN"} reads ${ln}(${xName}) and returns its update Δ, ‖Δ‖ = ${sig(norm(s.delta!))}, against ‖${xName}‖ = ${sig(norm(s.x))}.`;
    case "residual":
      return `Residual add: the stream carries on as ${xName} + Δ (‖${s.sub === 1 ? "h" : "y"}‖ = ${sig(norm(s.out!))}); the normalised copy is not kept.`;
  }
}
