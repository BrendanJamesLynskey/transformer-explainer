/**
 * Chapter 2: BPE learning its first merges on the corpus. State 0 is the
 * excerpt as characters; each later state is one merge: the five most
 * frequent pairs (the winner first), the new token, the corpus's token count
 * before and after, and the excerpt re-tokenised with the merges so far.
 * scripts/training/anim.py:bpe_states builds the same list.
 */
import {
  bpeEncode,
  bpeTrainSteps,
  type BpeModel,
  type PairCount,
} from "@/lib/training/bpe";

export const BPE_ANIM_MERGES = 24;
export const BPE_EXCERPT = 96;

export type BpeState = {
  step: number; // −1: before any merge
  pair: [number, number] | null;
  count: number | null;
  token: string | null;
  top: PairCount[] | null;
  tokensBefore: number;
  tokensAfter: number;
  excerpt: string[];
};

/** All states for `text` (the corpus). */
export function bpeStates(text: string, merges = BPE_ANIM_MERGES): BpeState[] {
  const it = bpeStatesSteps(text, merges);
  let r = it.next();
  while (!r.done) r = it.next();
  return r.value;
}

/** `bpeStates` in slices (a generator over `bpeTrainSteps`), for the page. */
export function* bpeStatesSteps(
  text: string,
  merges = BPE_ANIM_MERGES,
): Generator<void, BpeState[]> {
  const model: BpeModel = yield* bpeTrainSteps(text, merges);
  const excerpt = text.slice(0, BPE_EXCERPT);
  const tokens = (k: number) =>
    bpeEncode(excerpt, model.merges.slice(0, k)).map((i) => model.vocab[i]!);
  const total = model.history[0]?.tokensBefore ?? text.length;
  const out: BpeState[] = [
    {
      step: -1,
      pair: null,
      count: null,
      token: null,
      top: null,
      tokensBefore: total,
      tokensAfter: total,
      excerpt: tokens(0),
    },
  ];
  for (const h of model.history) {
    out.push({
      step: h.step,
      pair: h.pair,
      count: h.count,
      token: h.token,
      top: h.top,
      tokensBefore: h.tokensBefore,
      tokensAfter: h.tokensAfter,
      excerpt: tokens(h.step + 1),
    });
  }
  return out;
}

/** A token as shown in a label: spaces and newlines made visible. */
export function showToken(t: string): string {
  return t.replace(/ /g, "␣").replace(/\n/g, "↵");
}

export function bpeCaption(s: BpeState, vocab: (id: number) => string): string {
  if (s.step < 0)
    return `Start: one token per character. The excerpt is ${s.excerpt.length} tokens; the corpus ${s.tokensBefore.toLocaleString("en-GB")}.`;
  const [a, b] = s.pair!;
  return `Merge ${s.step + 1}: '${showToken(vocab(a))}' + '${showToken(vocab(b))}' occurs ${s.count!.toLocaleString("en-GB")} times, so it becomes the token '${showToken(s.token!)}'. The corpus shrinks from ${s.tokensBefore.toLocaleString("en-GB")} to ${s.tokensAfter.toLocaleString("en-GB")} tokens.`;
}
