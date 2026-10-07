/**
 * The stacking animation (chapter 06) as a list of model states: the
 * residual stream at one position p climbing through N blocks.
 *
 *   embed   layer 0: x₀ = E[t] + P[p]
 *   attn    block b adds its attention update
 *   ffn     block b adds its FFN update: the stream is now x_{b+1}
 *   final   the final LayerNorm and the tied output head give the logits
 *
 * After the embedding and after every block the animation also shows a
 * "logit lens": the final LayerNorm and the output head applied to the
 * stream early, LN_f(x_l) Eᵀ, as if the model stopped there. This model
 * ties its output head to the embedding table and ends with one LayerNorm,
 * so the lens is exactly the model's own read-out; at layer N it is the
 * model's logits (`tests/unit/anim-steps-b.test.ts` checks this exactly).
 *
 * Every number comes from a traced forward pass (`tracedForward`, the
 * `/api/compute/forward` trace) on the reader's input.
 */
import { tracedForward } from "@/lib/compute/traces";
import { layernorm } from "@/lib/transformer/layernorm";
import { matmul, transpose } from "@/lib/transformer/matmul";
import type { ModelConfig, ModelWeights } from "@/lib/transformer/model";
import { softmax } from "@/lib/transformer/softmax";
import { ALPHABET, encode } from "@/lib/transformer/tokenizer";

import { norm, pct, showChar, sig } from "./format";
import { topTokens, type TopToken } from "./overview-steps";

export type StackKind = "embed" | "attn" | "ffn" | "final";

/** One finished layer of the tower: the stream and its logit lens. */
export type LayerRow = {
  /** 0 = after the embedding, l = after block l. */
  layer: number;
  vec: number[];
  /** ‖update‖ of the block's attention and FFN (null for layer 0). */
  attnNorm: number | null;
  ffnNorm: number | null;
  /** The five most likely next tokens under the logit lens. */
  lens: TopToken[];
};

export type StackState = {
  kind: StackKind;
  pos: number;
  /** The block (attn, ffn), else −1. */
  block: number;
  nBlocks: number;
  /** The residual stream at `pos` after this step. */
  vec: number[];
  /** What this step added (attn, ffn), else null. */
  delta: number[] | null;
  /** The layers finished so far, bottom first. */
  rows: LayerRow[];
  /** The model's top five next tokens (final). */
  top: TopToken[] | null;
};

/**
 * The logit lens of a residual-stream vector: LN_f(x) Eᵀ, the same
 * operations, in the same order, as the end of `forwardTyped`.
 */
export function logitLens(x: number[], w: ModelWeights): number[] {
  const xf = layernorm(x, w.ln_final.gamma, w.ln_final.beta);
  return matmul([xf], transpose(w.tok_emb))[0]!;
}

function lensTop(x: number[], w: ModelWeights): TopToken[] {
  const logits = logitLens(x, w);
  return topTokens(logits, softmax(logits));
}

/** All states for position `pos` of `text` through `config.n_blocks` blocks. */
export function stackStates(
  text: string,
  config: ModelConfig,
  w: ModelWeights,
  pos: number,
): StackState[] {
  const S = config.seq_len;
  const N = config.n_blocks;
  const p = Math.max(0, Math.min(S - 1, pos));
  const tr = tracedForward(encode(text, S), config, w);
  let x = tr.tokEmb[p]!.map((v, k) => v + tr.posEmb[p]![k]!);
  const rows: LayerRow[] = [
    { layer: 0, vec: x, attnNorm: null, ffnNorm: null, lens: lensTop(x, w) },
  ];
  const base = { pos: p, nBlocks: N, top: null };
  const out: StackState[] = [
    {
      ...base,
      kind: "embed",
      block: -1,
      vec: x,
      delta: null,
      rows: rows.slice(),
    },
  ];
  for (let b = 0; b < N; b++) {
    const bt = tr.blocks[b]!;
    const h = bt.attnOut[p]!;
    const dA = h.map((v, k) => v - x[k]!);
    out.push({
      ...base,
      kind: "attn",
      block: b,
      vec: h,
      delta: dA,
      rows: rows.slice(),
    });
    const y = bt.ffnOut[p]!;
    const dF = y.map((v, k) => v - h[k]!);
    rows.push({
      layer: b + 1,
      vec: y,
      attnNorm: norm(dA),
      ffnNorm: norm(dF),
      lens: lensTop(y, w),
    });
    out.push({
      ...base,
      kind: "ffn",
      block: b,
      vec: y,
      delta: dF,
      rows: rows.slice(),
    });
    x = y;
  }
  const logits = tr.logits[p]!;
  out.push({
    ...base,
    kind: "final",
    block: -1,
    vec: tr.xFinal[p]!,
    delta: null,
    rows: rows.slice(),
    top: topTokens(logits, softmax(logits)),
  });
  return out;
}

/** The KaTeX term each step highlights. */
export const STACK_HL: Record<StackKind, string> = {
  embed: "x0",
  attn: "blk",
  ffn: "blk",
  final: "lens",
};

const ch = (id: number) => `'${showChar(ALPHABET[id] ?? "?")}'`;

/** The caption for one state (`tokens`: the input's characters). */
export function stackCaption(s: StackState, tokens: readonly string[]): string {
  const last = s.rows[s.rows.length - 1]!;
  const guess = (r: LayerRow) =>
    `${ch(r.lens[0]!.id)} at ${pct(r.lens[0]!.prob)}`;
  switch (s.kind) {
    case "embed":
      return `Position ${s.pos} '${showChar(tokens[s.pos] ?? "?")}', layer 0: the stream starts as E[t] + P[p] (‖x‖ = ${sig(norm(s.vec))}). Read out now, the logit lens's top guess is ${guess(last)}.`;
    case "attn":
      return `Block ${s.block + 1} of ${s.nBlocks}, attention: an update of size ${sig(norm(s.delta!))} is added (‖x‖ = ${sig(norm(s.vec))}).`;
    case "ffn":
      return `Block ${s.block + 1} of ${s.nBlocks}, FFN: an update of size ${sig(norm(s.delta!))} is added. The logit lens after block ${s.block + 1}: top guess ${guess(last)}.`;
    case "final": {
      const t = s.top![0]!;
      return `After block ${s.nBlocks}: the final LayerNorm and the output head give the model's logits (the lens at layer ${s.nBlocks}, exactly). Top next token: ${ch(t.id)} at ${pct(t.prob)}.`;
    }
  }
}
