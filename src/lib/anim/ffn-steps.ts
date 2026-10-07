/**
 * The feed-forward animation (chapter 04) as a list of model states, for
 * one position p of block 0:
 *
 *   input x = LN₂(h)_p            (d_model numbers)
 *   expand  u = x W₁ + b₁         (d_ff numbers: the bars grow)
 *   GELU    g_n = GELU(u_n)       (one neuron per step, marked on the curve)
 *   contract y = g W₂ + b₂        (back to d_model)
 *   fired   the neurons with u_n > 0, which GELU passes nearly unchanged
 *
 * The numbers are the `/api/compute/ffn` trace (`computeFfn`, run in the
 * browser).
 */
import { gelu } from "@/lib/transformer/gelu";

import { sig } from "./format";

/** What the FFN trace provides (a subset of `computeFfn`'s result). */
export type FfnTrace = {
  tokens: string[];
  input: number[][];
  pre: number[][];
  act: number[][];
  output: number[][];
};

export type FfnPhase = "input" | "expand" | "gelu" | "contract" | "fired";

export type FfnState = {
  phase: FfnPhase;
  pos: number;
  /** The neuron GELU was just applied to (gelu), else −1. */
  neuron: number;
  x: number[];
  /** u = x W₁ + b₁ (from expand on). */
  pre: number[] | null;
  /** GELU(u): neurons up to `neuron` (gelu), all after; null = not yet. */
  act: (number | null)[] | null;
  /** y = g W₂ + b₂ (from contract on). */
  out: number[] | null;
  /** Neurons with u_n > 0 (fired). */
  fired: number[] | null;
};

/** All states for position `pos`. */
export function ffnStates(tr: FfnTrace, pos: number): FfnState[] {
  const S = tr.input.length;
  const p = Math.max(0, Math.min(S - 1, pos));
  const x = tr.input[p]!;
  const pre = tr.pre[p]!;
  const act = tr.act[p]!;
  const y = tr.output[p]!;
  const F = pre.length;
  const base = { pos: p, neuron: -1, x, out: null, fired: null };
  const out: FfnState[] = [
    { ...base, phase: "input", pre: null, act: null },
    { ...base, phase: "expand", pre, act: null },
  ];
  for (let n = 0; n < F; n++) {
    out.push({
      ...base,
      phase: "gelu",
      neuron: n,
      pre,
      act: act.map((v, k) => (k <= n ? v : null)),
    });
  }
  out.push({ ...base, phase: "contract", pre, act: act.slice(), out: y });
  const fired = pre.flatMap((u, n) => (u > 0 ? [n] : []));
  out.push({ ...base, phase: "fired", pre, act: act.slice(), out: y, fired });
  return out;
}

/** The KaTeX term each phase highlights. */
export const FFN_HL: Record<FfnPhase, string> = {
  input: "x",
  expand: "up",
  gelu: "gelu",
  contract: "down",
  fired: "gelu",
};

/** The caption for one state. */
export function ffnCaption(s: FfnState): string {
  const d = s.x.length;
  switch (s.phase) {
    case "input":
      return `Position ${s.pos}: the FFN reads x = LN₂(h), ${d} numbers.`;
    case "expand":
      return `Expand: u = x W₁ + b₁ has ${s.pre!.length} numbers (d_ff = ${s.pre!.length / d}·d). u₀ = ${sig(s.pre![0]!)}.`;
    case "gelu": {
      const n = s.neuron;
      const u = s.pre![n]!;
      return `Neuron ${n}: GELU(${sig(u)}) = ${sig(s.act![n]!)}${u > 0 ? ", passed (nearly) unchanged" : ", squashed towards 0"}.`;
    }
    case "contract":
      return `Contract: y = GELU(u) W₂ + b₂ is back to ${d} numbers; y₀ = ${sig(s.out![0]!)}. It is added to the residual stream.`;
    case "fired":
      return `${s.fired!.length} of ${s.pre!.length} neurons fired (u > 0) for this position; the rest were squashed by GELU.`;
  }
}

/** Points on the GELU curve for x in [lo, hi] (for the chart). */
export function geluCurve(lo: number, hi: number, n = 81): [number, number][] {
  const pts: [number, number][] = [];
  for (let k = 0; k < n; k++) {
    const x = lo + ((hi - lo) * k) / (n - 1);
    pts.push([x, gelu(x)]);
  }
  return pts;
}
