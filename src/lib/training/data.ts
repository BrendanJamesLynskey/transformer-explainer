/*
 * src/lib/training/data.ts
 *
 * Operation: the corpus as token ids, the train/validation split, and the
 *            batches a training step reads.
 * Shapes:    sampleBatch(train, B, S, rng) → { inputs: [B][S], targets: [B][S] }
 * Intuition: a language model learns from windows of text. Each window of
 *            S + 1 tokens gives S training examples at once: the first S
 *            tokens are the input, and the target at every position is the
 *            token after it (teacher forcing: the model always sees the
 *            true text so far, never its own guesses). The last 10% of the
 *            corpus is held out as validation text the model never trains
 *            on; its loss shows whether the model learnt the language or
 *            memorised the training text.
 *            Matches scripts/training/data.py draw for draw.
 * MDX:       /training/02-data-and-tokens.
 */
import { ALPHABET } from "@/lib/transformer/tokenizer";

/** The fraction of the corpus held out for validation. */
export const VAL_FRACTION = 0.1;
/** The number of fixed validation windows. */
export const VAL_WINDOWS = 64;

const CHAR_TO_ID = new Map<string, number>();
for (let i = 0; i < ALPHABET.length; i++) CHAR_TO_ID.set(ALPHABET[i]!, i);

/** Characters to ids. The corpus only holds alphabet characters. */
export function encodeChars(text: string): number[] {
  const out = new Array<number>(text.length);
  for (let i = 0; i < text.length; i++) {
    const id = CHAR_TO_ID.get(text[i]!);
    if (id === undefined)
      throw new Error(`encodeChars: '${text[i]}' is not in the alphabet`);
    out[i] = id;
  }
  return out;
}

/** Train on the first 90%, validate on the last 10%. */
export function splitIds(ids: number[]): { train: number[]; val: number[] } {
  const cut = ids.length - Math.floor(ids.length * VAL_FRACTION);
  return { train: ids.slice(0, cut), val: ids.slice(cut) };
}

export type Batch = {
  inputs: number[][];
  targets: number[][];
  starts: number[];
};

/** B random windows of S + 1 tokens, starts ⌊u (N − S − 1)⌋ with u from `rng`. */
export function sampleBatch(
  train: number[],
  B: number,
  S: number,
  rng: () => number,
): Batch {
  const inputs: number[][] = [];
  const targets: number[][] = [];
  const starts: number[] = [];
  for (let b = 0; b < B; b++) {
    const start = Math.floor(rng() * (train.length - S - 1));
    const w = train.slice(start, start + S + 1);
    starts.push(start);
    inputs.push(w.slice(0, S));
    targets.push(w.slice(1));
  }
  return { inputs, targets, starts };
}

/** n evenly spaced windows of the validation split (the same every step). */
export function valBatch(val: number[], S: number, n = VAL_WINDOWS): Batch {
  const span = val.length - S - 1;
  const inputs: number[][] = [];
  const targets: number[][] = [];
  const starts: number[] = [];
  for (let k = 0; k < n; k++) {
    const start = Math.floor((k * span) / n);
    const w = val.slice(start, start + S + 1);
    starts.push(start);
    inputs.push(w.slice(0, S));
    targets.push(w.slice(1));
  }
  return { inputs, targets, starts };
}
