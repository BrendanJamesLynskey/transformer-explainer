/**
 * What each attention head does on real text: the numbers the landing
 * page and chapter 3 quote about the trained model's heads.
 *
 * Over 8-character windows of the corpus (every `stride`-th start) and
 * every query row i ≥ 1, each head's mean weight on the current character
 * (`self`), on the one before it (`prev`), and its mean normalised entropy
 * (`entropy`: 1 = spread evenly over the i + 1 visible keys, 0 = all on
 * one). For query rows that are a space (i ≥ 2), `spaceStart` is the mean
 * weight on the window's first two characters. `scripts/reference.py`
 * (`head_stats`) computes the same in PyTorch and writes
 * `src/data/decoder/trained-heads.json`; `tests/unit/attention-stats.test.ts`
 * requires the two to agree.
 */
import { configFor, DEFAULT_PARAMS, tracedForward } from "@/lib/compute/traces";
import type { ModelWeights } from "@/lib/transformer/model";
import { ALPHABET, encode } from "@/lib/transformer/tokenizer";

export type HeadStats = {
  block: number;
  head: number;
  windows: number;
  rows: number;
  self: number;
  prev: number;
  entropy: number;
  spaceRows: number;
  spaceStart: number;
};

/** Per-head statistics of `w` on `text` (block-major, as the JSON). */
export function headStats(
  w: ModelWeights,
  text: string,
  stride: number,
): HeadStats[] {
  const config = configFor(DEFAULT_PARAMS);
  const S = config.seq_len;
  const H = config.n_heads;
  const N = config.n_blocks;
  const ids = encode(text, text.length);
  const space = ALPHABET.indexOf(" ");
  const acc = Array.from({ length: N * H }, () => ({
    self: 0,
    prev: 0,
    ent: 0,
    n: 0,
    sp: 0,
    nsp: 0,
  }));
  let windows = 0;
  for (let s0 = 0; s0 + S <= ids.length; s0 += stride) {
    const win = ids.slice(s0, s0 + S);
    const tr = tracedForward(win, config, w);
    windows++;
    tr.blocks.forEach((bt, b) => {
      for (let h = 0; h < H; h++) {
        const a = acc[b * H + h]!;
        const wt = bt.attn.weights[h]!;
        for (let i = 1; i < S; i++) {
          const row = wt[i]!.slice(0, i + 1);
          a.self += row[i]!;
          a.prev += row[i - 1]!;
          let e = 0;
          for (const v of row) if (v > 0) e += v * Math.log(v);
          a.ent += -e / Math.log(i + 1);
          a.n++;
          if (i >= 2 && win[i] === space) {
            a.sp += row[0]! + row[1]!;
            a.nsp++;
          }
        }
      }
    });
  }
  return acc.map((a, k) => ({
    block: Math.floor(k / H),
    head: k % H,
    windows,
    rows: a.n,
    self: a.self / a.n,
    prev: a.prev / a.n,
    entropy: a.ent / a.n,
    spaceRows: a.nsp,
    spaceStart: a.nsp ? a.sp / a.nsp : 0,
  }));
}
