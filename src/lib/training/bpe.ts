/*
 * src/lib/training/bpe.ts
 *
 * Operation: byte-pair encoding (BPE) over the 64-character alphabet:
 *            learn merges from a corpus, then encode text with them.
 * Shapes:    bpeTrain(text, nMerges) → { vocab: string[], merges: [a, b][],
 *              history: one record per merge }
 *            bpeEncode(text, merges) → number[]
 * Intuition: start with one token per character. Count every adjacent pair
 *            of tokens inside words; the most frequent pair becomes a new
 *            token (" t" + "h" → " th"); replace it everywhere and count
 *            again. Frequent fragments become single tokens, so the same
 *            text takes fewer tokens: the model sees more text per step of
 *            its context. (Sennrich, Haddow & Birch, arXiv:1508.07909;
 *            GPT-2 does the same over bytes.)
 *            Words: an optional leading space and a run of non-space
 *            characters, or a lone space or newline; merges never cross
 *            them. Ties go to the smaller pair of ids, so Python
 *            (scripts/training/data.py) and this file learn the same merges.
 * MDX:       /training/02-data-and-tokens.
 */
import { ALPHABET } from "@/lib/transformer/tokenizer";

import { encodeChars } from "./data";

const WORD_RE = / ?[^ \n]+|\n| /g;

/** Split text into BPE words (each keeps its leading space). */
export function wordsOf(text: string): string[] {
  return text.match(WORD_RE) ?? [];
}

type Word = { ids: number[]; count: number };

function key(ids: number[]): string {
  return ids.join(",");
}

function wordCounts(text: string): Word[] {
  const map = new Map<string, Word>();
  for (const w of wordsOf(text)) {
    const ids = encodeChars(w);
    const k = key(ids);
    const e = map.get(k);
    if (e) e.count += 1;
    else map.set(k, { ids, count: 1 });
  }
  return [...map.values()];
}

/** Replace every non-overlapping (a, b), left to right, by `id`. */
export function mergeWord(
  word: number[],
  a: number,
  b: number,
  id: number,
): number[] {
  const out: number[] = [];
  let i = 0;
  while (i < word.length) {
    if (i + 1 < word.length && word[i] === a && word[i + 1] === b) {
      out.push(id);
      i += 2;
    } else {
      out.push(word[i]!);
      i += 1;
    }
  }
  return out;
}

export type PairCount = { pair: [number, number]; count: number };

/** Every adjacent pair's count, weighted by word frequency. */
function pairCounts(words: Word[]): Map<string, PairCount> {
  const pairs = new Map<string, PairCount>();
  for (const w of words) {
    for (let i = 0; i + 1 < w.ids.length; i++) {
      const a = w.ids[i]!;
      const b = w.ids[i + 1]!;
      const k = `${a},${b}`;
      const e = pairs.get(k);
      if (e) e.count += w.count;
      else pairs.set(k, { pair: [a, b], count: w.count });
    }
  }
  return pairs;
}

/** Most frequent first; ties broken by the smaller (a, b). */
function byRank(x: PairCount, y: PairCount): number {
  if (x.count !== y.count) return y.count - x.count;
  if (x.pair[0] !== y.pair[0]) return x.pair[0] - y.pair[0];
  return x.pair[1] - y.pair[1];
}

export type MergeRecord = {
  step: number;
  pair: [number, number];
  count: number;
  new: number;
  token: string;
  top: PairCount[];
  tokensBefore: number;
  tokensAfter: number;
};

export type BpeModel = {
  vocab: string[];
  merges: [number, number][];
  history: MergeRecord[];
};

/** Learn `nMerges` merges from `text` (records the top `top` pairs each time). */
export function bpeTrain(text: string, nMerges: number, top = 5): BpeModel {
  let words = wordCounts(text);
  const vocab = ALPHABET.split("");
  const merges: [number, number][] = [];
  const history: MergeRecord[] = [];
  const tokens = (ws: Word[]) =>
    ws.reduce((s, w) => s + w.ids.length * w.count, 0);
  let total = tokens(words);
  for (let step = 0; step < nMerges; step++) {
    const ranked = [...pairCounts(words).values()].sort(byRank);
    if (ranked.length === 0) break;
    const best = ranked[0]!;
    const [a, b] = best.pair;
    const id = vocab.length;
    vocab.push(vocab[a]! + vocab[b]!);
    merges.push([a, b]);
    const merged = new Map<string, Word>();
    for (const w of words) {
      const ids = mergeWord(w.ids, a, b, id);
      const k = key(ids);
      const e = merged.get(k);
      if (e) e.count += w.count;
      else merged.set(k, { ids, count: w.count });
    }
    words = [...merged.values()];
    const before = total;
    total = tokens(words);
    history.push({
      step,
      pair: [a, b],
      count: best.count,
      new: id,
      token: vocab[id]!,
      top: ranked
        .slice(0, top)
        .map((p) => ({ pair: [p.pair[0], p.pair[1]], count: p.count })),
      tokensBefore: before,
      tokensAfter: total,
    });
  }
  return { vocab, merges, history };
}

/** Encode text with the merges, applied in the order they were learnt. */
export function bpeEncode(text: string, merges: [number, number][]): number[] {
  const out: number[] = [];
  const cache = new Map<string, number[]>();
  for (const w of wordsOf(text)) {
    let t = cache.get(w);
    if (!t) {
      t = encodeChars(w);
      merges.forEach(([a, b], k) => {
        t = mergeWord(t!, a, b, ALPHABET.length + k);
      });
      cache.set(w, t);
    }
    out.push(...t);
  }
  return out;
}
