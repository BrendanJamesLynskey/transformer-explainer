/**
 * Chapter 4: the gradient on its way back through Part 1's model, for the
 * window "alice was". Each state is one stop: which gradient it is, its
 * value (a matrix: one row per position), each row's size, and the weights
 * whose gradients are complete there. The values are recorded by the
 * trainer's own backward pass (model-grad.ts `record`); scripts/training's
 * backward records the same names and values.
 */
import { lossAndGrads, type GradRecord } from "@/lib/training/model-grad";
import { sumSquares, type ParamMap } from "@/lib/training/params";
import { PART1_MODEL } from "@/lib/training/trainer";

import { norm, sig } from "../format";

import { windowIds } from "./objective-steps";

/** What each stop is, and which weights' gradients are finished there. */
export const STOPS: Record<
  string,
  { title: string; op: string; params: (b: string) => string[]; saved: string; hl: string }
> = {
  dlogits: {
    title: "∂L/∂logits = (p − y)/n",
    op: "cross-entropy",
    params: () => [],
    saved: "the probabilities p",
    hl: "loss",
  },
  dxf: {
    title: "through the tied head: ∂L/∂x_f = ∂L/∂logits · E",
    op: "logits = x_f Eᵀ",
    params: () => [],
    saved: "x_f (and E)",
    hl: "head",
  },
  dx_top: {
    title: "through the final LayerNorm",
    op: "LN_f",
    params: () => ["ln_final.gamma", "ln_final.beta"],
    saved: "the LayerNorm's input (to recompute μ, σ, x̂)",
    hl: "ln",
  },
  dact: {
    title: "into the FFN: ∂L/∂GELU(u) = ∂L/∂y · W₂ᵀ",
    op: "y = GELU(u) W₂ + b₂",
    params: (b) => [`${b}.ffn.W2`, `${b}.ffn.b2`],
    saved: "GELU(u)",
    hl: "ffn",
  },
  dpre: {
    title: "through GELU: × GELU′(u)",
    op: "GELU",
    params: () => [],
    saved: "u (the pre-activation)",
    hl: "ffn",
  },
  dln2: {
    title: "through W₁: ∂L/∂LN₂(h) = ∂L/∂u · W₁ᵀ",
    op: "u = LN₂(h) W₁ + b₁",
    params: (b) => [`${b}.ffn.W1`, `${b}.ffn.b1`],
    saved: "LN₂(h)",
    hl: "ffn",
  },
  dh: {
    title: "the residual: ∂L/∂h = ∂L/∂y + the FFN branch's gradient",
    op: "y = h + FFN(LN₂(h))",
    params: (b) => [`${b}.ln2.gamma`, `${b}.ln2.beta`],
    saved: "h (LayerNorm 2's input)",
    hl: "res",
  },
  dscores_h0: {
    title: "inside attention, head 0: softmax backward, dS = P ⊙ (dP − Σ dP ⊙ P)",
    op: "P = softmax(QKᵀ/√d_k + mask)",
    params: (b) => [`${b}.attn.W_o`],
    saved: "P, Q, K, V",
    hl: "attn",
  },
  dln1: {
    title: "out of attention: ∂L/∂LN₁(x) = dQ W_qᵀ + dK W_kᵀ + dV W_vᵀ",
    op: "Q, K, V = LN₁(x) W_q, W_k, W_v",
    params: (b) => [`${b}.attn.W_q`, `${b}.attn.W_k`, `${b}.attn.W_v`],
    saved: "LN₁(x)",
    hl: "attn",
  },
  dx: {
    title: "the residual again: ∂L/∂x = ∂L/∂h + the attention branch's gradient",
    op: "h = x + Attn(LN₁(x))",
    params: (b) => [`${b}.ln1.gamma`, `${b}.ln1.beta`],
    saved: "x (LayerNorm 1's input)",
    hl: "res",
  },
};

export type BackpropState = {
  /** the record's name, e.g. "block_1.dact"; "embedding" for the last stop */
  name: string;
  /** the stop's key in STOPS (or "embedding") */
  kind: string;
  /** "block 1" for block stops, "" otherwise */
  block: string;
  value: number[][];
  rowNorms: number[];
  /** weights whose gradients are complete at this stop, with ‖g‖ */
  params: { name: string; norm: number }[];
  loss: number;
};

/** All stops, from the loss back to the embedding table. */
export function backpropStates(params: ParamMap): BackpropState[] {
  const { inputs, targets } = windowIds();
  const rec: GradRecord[] = [];
  const { loss, grads } = lossAndGrads(params, PART1_MODEL, [inputs], [targets], rec);
  const gnorm = (n: string) => Math.sqrt(sumSquares(grads[n]!));
  const out: BackpropState[] = rec.map((r) => {
    const m = r.name.match(/^(block_(\d+))\.(.+)$/);
    const kind = m ? m[3]! : r.name;
    const bname = m ? m[1]! : "";
    return {
      name: r.name,
      kind,
      block: m ? `block ${m[2]}` : "",
      value: r.value,
      rowNorms: r.value.map((row) => norm(row)),
      params: STOPS[kind]!.params(bname).map((n) => ({ name: n, norm: gnorm(n) })),
      loss,
    };
  });
  const last = rec[rec.length - 1]!.value;
  out.push({
    name: "embedding",
    kind: "embedding",
    block: "",
    value: last,
    rowNorms: last.map((row) => norm(row)),
    params: [{ name: "tok_emb", norm: gnorm("tok_emb") }],
    loss,
  });
  return out;
}

export function backpropCaption(s: BackpropState, i: number, n: number): string {
  const total = s.rowNorms.reduce((a, b) => a + b * b, 0);
  const size = `‖∂L‖ = ${sig(Math.sqrt(total))}`;
  if (s.kind === "embedding")
    return `Stop ${i + 1}/${n}: the embedding. Each position's gradient is added to its character's row of E (the head's gradient was added at stop 2). ‖∂L/∂E‖ = ${sig(s.params[0]!.norm)}.`;
  const st = STOPS[s.kind]!;
  const where = s.block ? `${s.block}: ` : "";
  const ps = s.params.length
    ? ` Gradients done: ${s.params.map((p) => `${p.name.replace(/^block_\d+\./, "")} ${sig(p.norm)}`).join(", ")}.`
    : "";
  return `Stop ${i + 1}/${n}: ${where}${st.title}. ${size}.${ps}`;
}
