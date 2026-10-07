/**
 * The chapter 01 hero: one token's journey through the toy decoder, as a
 * list of model states.
 *
 * Each round follows the newest token at position p: its id becomes a
 * vector (the embedding row E[t]), the positional vector P[p] is added,
 * then every block adds an attention update and an FFN update to that
 * residual stream; the final LayerNorm and the tied output head give the
 * logits, a seeded sample picks the next token, and the token is appended,
 * so the next round starts one position further on. Every value is read
 * from a real forward pass (`tracedForward`) on the reader's input.
 *
 * Sampling uses the site's sampler (`sample`, temperature 1) with a
 * mulberry32 generator seeded by the model seed, so a given input always
 * produces the same continuation.
 */
import { tracedForward } from "@/lib/compute/traces";
import type { ModelConfig, ModelWeights } from "@/lib/transformer/model";
import { mulberry32 } from "@/lib/transformer/random";
import { sample } from "@/lib/transformer/sampling";
import { softmax } from "@/lib/transformer/softmax";
import { ALPHABET, encode } from "@/lib/transformer/tokenizer";

import { norm, pct, showChar, sig } from "./format";

export type OverviewKind =
  | "token"
  | "embed"
  | "position"
  | "attn"
  | "ffn"
  | "norm"
  | "logits"
  | "sample"
  | "append";

export type TopToken = { id: number; logit: number; prob: number };

export type OverviewState = {
  kind: OverviewKind;
  round: number;
  /** The position being followed (the newest token's). */
  pos: number;
  /** The block (attn, ffn), else −1. */
  block: number;
  /** Token ids in context; after `append` it includes the new token. */
  context: number[];
  /** The id at `pos`. */
  id: number;
  /** The residual stream at `pos` after this step (null before embed). */
  vec: number[] | null;
  /** What this step added: P[pos], or a block's attention / FFN update. */
  delta: number[] | null;
  /** The five most likely next tokens (logits, sample, append). */
  top: TopToken[] | null;
  /** The sampled id and its probability (sample, append). */
  sampled: number | null;
  prob: number | null;
};

export type OverviewOptions = {
  /** Most rounds to play (each appends one token). */
  rounds: number;
};

/** The ids of the prompt: at most S − 1 characters, so one can be added. */
export function promptIds(text: string, seqLen: number): number[] {
  const t = text.length === 0 ? " " : text.slice(0, seqLen - 1);
  return encode(t, t.length);
}

/** The top `k` tokens by probability (ties: lower id first). */
export function topTokens(
  logits: number[],
  probs: number[],
  k = 5,
): TopToken[] {
  return logits
    .map((logit, id) => ({ id, logit, prob: probs[id]! }))
    .sort((a, b) => b.prob - a.prob || a.id - b.id)
    .slice(0, k);
}

/** All states of the hero for `text` under `config` and `weights`. */
export function overviewStates(
  text: string,
  config: ModelConfig,
  weights: ModelWeights,
  opts: OverviewOptions,
): OverviewState[] {
  const S = config.seq_len;
  const rng = mulberry32(config.seed);
  let context = promptIds(text, S);
  const out: OverviewState[] = [];
  for (let round = 0; round < opts.rounds && context.length < S; round++) {
    const p = context.length - 1;
    const ids = context.concat(new Array<number>(S - context.length).fill(0));
    const tr = tracedForward(ids, config, weights);
    const id = context[p]!;
    const base = {
      round,
      pos: p,
      block: -1,
      context: context.slice(),
      id,
      top: null,
      sampled: null,
      prob: null,
    };
    out.push({ ...base, kind: "token", vec: null, delta: null });
    out.push({ ...base, kind: "embed", vec: tr.tokEmb[p]!, delta: null });
    let x = tr.tokEmb[p]!.map((v, k) => v + tr.posEmb[p]![k]!);
    out.push({ ...base, kind: "position", vec: x, delta: tr.posEmb[p]! });
    tr.blocks.forEach((bt, b) => {
      const h = bt.attnOut[p]!;
      out.push({
        ...base,
        kind: "attn",
        block: b,
        vec: h,
        delta: h.map((v, k) => v - x[k]!),
      });
      const y = bt.ffnOut[p]!;
      out.push({
        ...base,
        kind: "ffn",
        block: b,
        vec: y,
        delta: y.map((v, k) => v - h[k]!),
      });
      x = y;
    });
    const xf = tr.xFinal[p]!;
    out.push({ ...base, kind: "norm", vec: xf, delta: null });
    const logits = tr.logits[p]!;
    // temperature 1: the sampler's distribution is softmax(logits / 1)
    const probs = softmax(logits.map((v) => v / 1));
    const top = topTokens(logits, probs);
    out.push({ ...base, kind: "logits", vec: xf, delta: null, top });
    const next = sample(logits, { kind: "temperature", temperature: 1 }, rng);
    const picked = { top, sampled: next, prob: probs[next]! };
    out.push({ ...base, kind: "sample", vec: xf, delta: null, ...picked });
    context = context.concat([next]);
    out.push({
      ...base,
      ...picked,
      kind: "append",
      context: context.slice(),
      vec: xf,
      delta: null,
    });
  }
  return out;
}

/** The KaTeX term each step highlights. */
export const OVERVIEW_HL: Record<OverviewKind, string> = {
  token: "tok",
  embed: "emb",
  position: "pos",
  attn: "attn",
  ffn: "ffn",
  norm: "ln",
  logits: "logits",
  sample: "sample",
  append: "sample",
};

const ch = (id: number) => `'${showChar(ALPHABET[id] ?? "?")}'`;

/** The caption for one state. */
export function overviewCaption(s: OverviewState): string {
  const at = `Position ${s.pos}`;
  switch (s.kind) {
    case "token":
      return s.round === 0
        ? `${at}: the newest character ${ch(s.id)} is token id ${s.id}.`
        : `Round ${s.round + 1}: the token just added, ${ch(s.id)} (id ${s.id}), is now at position ${s.pos}.`;
    case "embed":
      return `${at}: id ${s.id} selects row ${s.id} of the embedding table, a vector of ${s.vec!.length} numbers (‖x‖ = ${sig(norm(s.vec!))}).`;
    case "position":
      return `${at}: add the positional vector P[${s.pos}] (‖P‖ = ${sig(norm(s.delta!))}); the residual stream starts with ‖x‖ = ${sig(norm(s.vec!))}.`;
    case "attn":
      return `Block ${s.block + 1}, attention: position ${s.pos} reads positions 0–${s.pos} and adds an update of size ${sig(norm(s.delta!))} (‖x‖ = ${sig(norm(s.vec!))}).`;
    case "ffn":
      return `Block ${s.block + 1}, feed-forward: an update of size ${sig(norm(s.delta!))} is added (‖x‖ = ${sig(norm(s.vec!))}).`;
    case "norm":
      return `Final LayerNorm: the stream is rescaled to mean 0 and variance 1 (‖x‖ = ${sig(norm(s.vec!))}).`;
    case "logits": {
      const t = s.top![0]!;
      return `Logits: one score per token (x · Eᵀ); the most likely next token is ${ch(t.id)} at ${pct(t.prob)}.`;
    }
    case "sample":
      return `Sample (temperature 1, seeded): ${ch(s.sampled!)} with probability ${pct(s.prob!)}.`;
    case "append":
      return `Append ${ch(s.sampled!)} at position ${s.pos + 1}; the next round runs the model again on ${s.context.length} tokens.`;
  }
}
