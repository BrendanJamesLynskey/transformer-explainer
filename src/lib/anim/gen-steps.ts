/**
 * The sampling animation and generation loop (chapter 07 and the
 * playground) as a list of model states. Each round:
 *
 *   forward      the model runs; the newest token's key and value (in every
 *                block) join the KV cache (the prompt's, all at once, in
 *                round 1: the "prefill")
 *   logits       ℓ at the last position: one score per character
 *   softmax      p = softmax(ℓ)
 *   temperature  p_τ = softmax(ℓ / τ)
 *   truncate     top-k or top-p keeps some tokens and renormalises
 *                (skipped in plain temperature mode)
 *   draw         a seeded u ~ U[0, 1) walks the cumulative probabilities
 *   append       the sampled token joins the context
 *
 * then the next round starts, until the context is full. The probability
 * vectors are computed with the sampler's own functions (`applyTemperature`,
 * `topKMask`, `topPMask`, `softmax`, `sampleFromProbs`), and every sampled
 * token equals `sample(ℓ, mode, rng)` with the same seeded generator
 * (`tests/unit/anim-steps-b.test.ts`).
 *
 * The site's model recomputes the whole context each round; a server keeps
 * the earlier keys and values instead. The cache drawn here holds exactly
 * the K and V rows of the traced forward pass, and the tests check that a
 * cached column never changes from one round to the next, which is why
 * caching them is safe.
 */
import { tracedForward } from "@/lib/compute/traces";
import type { ModelConfig, ModelWeights } from "@/lib/transformer/model";
import { mulberry32 } from "@/lib/transformer/random";
import {
  applyTemperature,
  sampleFromProbs,
  topKMask,
  topPMask,
  type SampleMode,
} from "@/lib/transformer/sampling";
import { softmax } from "@/lib/transformer/softmax";
import { ALPHABET } from "@/lib/transformer/tokenizer";

import { pct, showChar, sig } from "./format";
import { promptIds } from "./overview-steps";

export type GenMode = "temperature" | "top-k" | "top-p";

export type GenOptions = {
  mode: GenMode;
  temperature: number;
  k: number;
  p: number;
  /** Most rounds to play (each appends one token). */
  rounds: number;
};

export type GenPhase =
  | "forward"
  | "logits"
  | "softmax"
  | "temperature"
  | "truncate"
  | "draw"
  | "append";

/** One block's KV cache: a K row and a V row per cached position. */
export type KvBlock = { K: number[][]; V: number[][] };

export type GenState = {
  phase: GenPhase;
  round: number;
  /** The last position of the context (whose logits are sampled). */
  pos: number;
  /** The model's context length S (generation stops when it is full). */
  seqLen: number;
  /** Token ids in context; after `append` it includes the new token. */
  context: number[];
  /** The KV cache, per block, for positions 0 … pos. */
  cache: KvBlock[];
  /** The positions whose K and V this round's forward step wrote. */
  fresh: number[];
  mode: GenMode;
  tau: number;
  k: number;
  p: number;
  logits: number[] | null;
  /** softmax(ℓ), τ = 1. */
  probs1: number[] | null;
  /** softmax(ℓ / τ). */
  probsT: number[] | null;
  /** The ids that survive top-k / top-p (null: no truncation yet or ever). */
  kept: number[] | null;
  /** The distribution the token is drawn from. */
  final: number[] | null;
  u: number | null;
  sampled: number | null;
};

/** The sampler's mode object for these options. */
export function sampleMode(o: GenOptions): SampleMode {
  if (o.mode === "top-k")
    return { kind: "top-k", k: o.k, temperature: o.temperature };
  if (o.mode === "top-p")
    return { kind: "top-p", p: o.p, temperature: o.temperature };
  return { kind: "temperature", temperature: o.temperature };
}

/** All states of the loop for `text` under `config` and `weights`. */
export function genStates(
  text: string,
  config: ModelConfig,
  weights: ModelWeights,
  o: GenOptions,
): GenState[] {
  const S = config.seq_len;
  const rng = mulberry32(config.seed);
  let context = promptIds(text, S);
  const out: GenState[] = [];
  for (let round = 0; round < o.rounds && context.length < S; round++) {
    const L = context.length;
    const p = L - 1;
    const ids = context.concat(new Array<number>(S - L).fill(0));
    const tr = tracedForward(ids, config, weights);
    const cache = tr.blocks.map((bt) => ({
      K: bt.attn.K.slice(0, L),
      V: bt.attn.V.slice(0, L),
    }));
    const fresh =
      round === 0 ? Array.from({ length: L }, (_, i) => i) : [L - 1];
    const base: GenState = {
      phase: "forward",
      round,
      pos: p,
      seqLen: S,
      context: context.slice(),
      cache,
      fresh,
      mode: o.mode,
      tau: o.temperature,
      k: o.k,
      p: o.p,
      logits: null,
      probs1: null,
      probsT: null,
      kept: null,
      final: null,
      u: null,
      sampled: null,
    };
    out.push(base);
    const logits = tr.logits[p]!;
    const s1 = { ...base, logits };
    out.push({ ...s1, phase: "logits" });
    const probs1 = softmax(logits);
    out.push({ ...s1, phase: "softmax", probs1 });
    // the sampler's steps: ℓ / τ, then (optionally) a mask, then softmax
    const tempered = applyTemperature(logits, o.temperature);
    const probsT = softmax(tempered);
    const s2 = { ...s1, probs1, probsT };
    let final = probsT;
    let kept: number[] | null = null;
    out.push({
      ...s2,
      phase: "temperature",
      final: o.mode === "temperature" ? final : null,
    });
    if (o.mode !== "temperature") {
      const masked =
        o.mode === "top-k" ? topKMask(tempered, o.k) : topPMask(tempered, o.p);
      final = softmax(masked);
      kept = masked.flatMap((v, i) => (Number.isFinite(v) ? [i] : []));
      out.push({ ...s2, phase: "truncate", kept, final });
    }
    const u = rng();
    const sampled = sampleFromProbs(final, () => u);
    const s3 = { ...s2, kept, final, u, sampled };
    out.push({ ...s3, phase: "draw" });
    context = context.concat([sampled]);
    out.push({ ...s3, phase: "append", context: context.slice() });
  }
  return out;
}

/**
 * The distribution on screen during the temperature step, a fraction `t`
 * of the way from τ = 1 to the chosen τ (the frame is a pure function of
 * the state and t): softmax(ℓ / τ(t)), τ(t) = 1 + (τ − 1)·t. At t = 1 it is
 * the state's p_τ exactly.
 */
export function temperatureFrame(
  s: GenState,
  t: number,
): { tau: number; probs: number[] } {
  if (t >= 1 || !s.logits) return { tau: s.tau, probs: s.probsT ?? [] };
  const tau = 1 + (s.tau - 1) * Math.max(0, t);
  return { tau, probs: softmax(applyTemperature(s.logits, tau)) };
}

/** Token ids in display order: most likely first at τ = 1 (ties by id). */
export function displayOrder(logits: readonly number[]): number[] {
  return logits
    .map((v, i) => ({ v, i }))
    .sort((a, b) => b.v - a.v || a.i - b.i)
    .map((x) => x.i);
}

/** The KaTeX term each phase highlights. */
export const GEN_HL: Record<GenPhase, string> = {
  forward: "kv",
  logits: "logits",
  softmax: "tau",
  temperature: "tau",
  truncate: "trunc",
  draw: "draw",
  append: "sample",
};

const ch = (id: number) => `'${showChar(ALPHABET[id] ?? "?")}'`;

function maxP(v: readonly number[]): number {
  let m = 0;
  for (const x of v) if (x > m) m = x;
  return m;
}

/** The caption for one state. */
export function genCaption(s: GenState): string {
  const r = `Round ${s.round + 1}`;
  const L = s.context.length;
  switch (s.phase) {
    case "forward":
      return s.round === 0
        ? `${r} (prefill): the model reads the ${L} prompt tokens and writes their keys and values into the KV cache, ${L} columns per block.`
        : `${r}: the new token ${ch(s.context[s.pos]!)} at position ${s.pos} adds one key and one value column per block (${L} now); the ${L - 1} earlier columns are unchanged, so a server reuses them.`;
    case "logits": {
      const lo = Math.min(...s.logits!);
      const hi = Math.max(...s.logits!);
      return `${r}: the logits at position ${s.pos}, one score per character (${s.logits!.length}), from ${sig(lo)} to ${sig(hi)}.`;
    }
    case "softmax":
      return `${r}: softmax turns them into probabilities; the most likely character has ${pct(maxP(s.probs1!))}.`;
    case "temperature":
      return s.tau === 1
        ? `${r}: temperature τ = 1 leaves the distribution as it is.`
        : `${r}: temperature τ = ${sig(s.tau)} divides the logits by τ before softmax; the most likely character goes from ${pct(maxP(s.probs1!))} to ${pct(maxP(s.probsT!))}.`;
    case "truncate": {
      const n = s.kept!.length;
      const rule =
        s.mode === "top-k"
          ? `top-k (k = ${s.k}) keeps the ${n} most likely`
          : `top-p (p = ${sig(s.p)}) keeps the ${n} most likely, the fewest holding at least ${pct(s.p)}`;
      return `${r}: ${rule}, drops the other ${s.logits!.length - n} and renormalises; the top one now has ${pct(maxP(s.final!))}.`;
    }
    case "draw":
      return `${r}: the seeded generator draws u = ${sig(s.u!)}; walking the cumulative probabilities in id order, u lands on ${ch(s.sampled!)} (${pct(s.final![s.sampled!]!)}).`;
    case "append":
      return L >= s.seqLen
        ? `${r}: append ${ch(s.sampled!)} at position ${s.pos + 1}; the context is full (${L} tokens), so generation stops.`
        : `${r}: append ${ch(s.sampled!)} at position ${s.pos + 1}; the context is now ${L} tokens, and the loop runs again.`;
  }
}
