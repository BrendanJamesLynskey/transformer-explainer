/**
 * The attention animation (chapter 03) as a list of model states.
 *
 * For a chosen head and query row i the animation builds row i of the
 * attention pattern the way the code computes it:
 *
 *   1. the dot products q_i · k_j fill the row one key at a time;
 *   2. the row is scaled by 1/√d_k;
 *   3. the causal mask blanks the future (j > i → −∞);
 *   4. softmax: the exponentials e_j = exp(s_j − max), then their sum,
 *      then the weights a_j = e_j / Σ e;
 *   5. the output o_i = Σ_j a_j v_j accumulates one value row at a time;
 *
 * then moves on to the next row. "All heads" repeats this for every head.
 * The last two steps concatenate the heads and project through W_o.
 *
 * The numbers come from the `/api/compute/attention` trace (block 0, after
 * LN₁), computed in the browser by `computeAttention`. The dot products,
 * exponentials and partial sums are recomputed here in the same order as
 * `matmul` and `softmax`, so the final values equal the trace's exactly
 * (`tests/unit/anim-attention.test.ts`).
 */
import { sig, showChar } from "./format";

/** What the attention trace provides (a subset of `computeAttention`). */
export type AttentionTrace = {
  tokens: string[];
  Q: number[][];
  K: number[][];
  V: number[][];
  scores: number[][][];
  weights: number[][][];
  output: number[][];
  nHeads: number;
};

export type AttnPhase =
  | "dot"
  | "scale"
  | "mask"
  | "exp"
  | "sum"
  | "softmax"
  | "wsum"
  | "concat"
  | "project";

export type AttnState = {
  phase: AttnPhase;
  /** −1 for the concat and project steps (they involve every head). */
  head: number;
  row: number;
  /** The head dimension d_k (= d_model / n_heads). */
  dk: number;
  /** The key (dot) or value row (wsum) just used, else −1. */
  j: number;
  /** q_i · k_j for keys computed so far (null = not yet). */
  dots: (number | null)[];
  /** dots / √d_k; from the mask step on, null where masked. */
  scores: (number | null)[] | null;
  /** exp(s_j − max_j s_j) for visible keys (null = masked). */
  exps: (number | null)[] | null;
  sum: number | null;
  /** The softmax weights a_j (null = masked). */
  weights: (number | null)[] | null;
  /** Σ_{j' ≤ j} a_j' v_j' so far (d_head numbers). */
  out: number[] | null;
  /** Rows of this head's pattern finished before this row. */
  rowsDone: number[];
  /** Concatenated head outputs [S, d_model] (concat, project). */
  concat: number[][] | null;
  /** The attention output after W_o [S, d_model] (project). */
  projected: number[][] | null;
};

export type AttnOptions = {
  /** A head index, or "all" to play every head in turn. */
  head: number | "all";
  /** The first query row; the animation then continues to the last row. */
  row: number;
};

/** Columns `[start, start + d)` of a matrix: one head's slice. */
function slice(m: number[][], h: number, d: number): number[][] {
  return m.map((r) => r.slice(h * d, (h + 1) * d));
}

/** One head's output row i: o_i = Σ_j a_ij v_j, as `matmul` sums it. */
function outPartial(w: number[], V: number[][], upTo: number): number[] {
  const d = V[0]!.length;
  const o = new Array<number>(d).fill(0);
  for (let j = 0; j <= upTo; j++) {
    const a = w[j]!;
    if (a === 0) continue; // matmul skips zero weights (masked keys)
    for (let c = 0; c < d; c++) o[c] = o[c]! + a * V[j]![c]!;
  }
  return o;
}

/** The states for one head from query row `r0` to the last row. */
function headStates(tr: AttentionTrace, h: number, r0: number): AttnState[] {
  const S = tr.Q.length;
  const dHead = tr.Q[0]!.length / tr.nHeads;
  const Q = slice(tr.Q, h, dHead);
  const K = slice(tr.K, h, dHead);
  const V = slice(tr.V, h, dHead);
  const scale = 1 / Math.sqrt(Math.max(1, dHead));
  const out: AttnState[] = [];
  const rowsDone: number[] = [];
  for (let i = r0; i < S; i++) {
    const base = {
      head: h,
      row: i,
      dk: dHead,
      rowsDone: rowsDone.slice(),
      concat: null,
      projected: null,
    };
    // 1. q_i · k_j, summed over the head dimension in index order (matmul)
    const raw: number[] = [];
    for (let j = 0; j < S; j++) {
      let s = 0;
      for (let c = 0; c < dHead; c++) {
        const a = Q[i]![c]!;
        if (a === 0) continue;
        s = s + a * K[j]![c]!;
      }
      raw.push(s);
    }
    const dots = (n: number) =>
      raw.map((v, j) => (j <= n ? v : null)) as (number | null)[];
    for (let j = 0; j < S; j++) {
      out.push({
        ...base,
        phase: "dot",
        j,
        dots: dots(j),
        scores: null,
        exps: null,
        sum: null,
        weights: null,
        out: null,
      });
    }
    // 2. scale by 1/√d_k (these are the trace's scores)
    const scaled = raw.map((v) => v * scale);
    out.push({
      ...base,
      phase: "scale",
      j: -1,
      dots: raw.slice(),
      scores: scaled.slice(),
      exps: null,
      sum: null,
      weights: null,
      out: null,
    });
    // 3. causal mask: j > i → −∞ (null here)
    const masked = scaled.map((v, j) => (j <= i ? v : null));
    out.push({
      ...base,
      phase: "mask",
      j: -1,
      dots: raw.slice(),
      scores: masked,
      exps: null,
      sum: null,
      weights: null,
      out: null,
    });
    // 4. softmax: y_j = exp(s_j − max) / Σ_j' exp(s_j' − max)
    let max = -Infinity;
    for (let j = 0; j <= i; j++) if (scaled[j]! > max) max = scaled[j]!;
    const exps = masked.map((v) => (v === null ? null : Math.exp(v - max)));
    let sum = 0;
    for (const e of exps) if (e !== null) sum += e;
    const w = tr.weights[h]![i]!;
    const weights = w.map((v, j) => (j <= i ? v : null));
    const soft = { dots: raw.slice(), scores: masked, j: -1, out: null };
    out.push({
      ...base,
      ...soft,
      phase: "exp",
      exps,
      sum: null,
      weights: null,
    });
    out.push({ ...base, ...soft, phase: "sum", exps, sum, weights: null });
    out.push({ ...base, ...soft, phase: "softmax", exps, sum, weights });
    // 5. o_i = Σ_j a_ij v_j, one value row at a time
    for (let j = 0; j <= i; j++) {
      out.push({
        ...base,
        phase: "wsum",
        j,
        dots: raw.slice(),
        scores: masked,
        exps,
        sum,
        weights,
        out: outPartial(w, V, j),
      });
    }
    rowsDone.push(i);
  }
  return out;
}

/** Every head's output, concatenated: [S, d_model]. */
export function concatHeads(tr: AttentionTrace): number[][] {
  const S = tr.Q.length;
  const dHead = tr.Q[0]!.length / tr.nHeads;
  const rows: number[][] = [];
  for (let i = 0; i < S; i++) {
    const r: number[] = [];
    for (let h = 0; h < tr.nHeads; h++)
      r.push(...outPartial(tr.weights[h]![i]!, slice(tr.V, h, dHead), S - 1));
    rows.push(r);
  }
  return rows;
}

/** All states for the chosen head (or all heads) from row `opts.row`. */
export function attentionStates(
  tr: AttentionTrace,
  opts: AttnOptions,
): AttnState[] {
  const S = tr.Q.length;
  const r0 = Math.max(0, Math.min(S - 1, opts.row));
  const heads =
    opts.head === "all"
      ? Array.from({ length: tr.nHeads }, (_, h) => h)
      : [Math.max(0, Math.min(tr.nHeads - 1, opts.head))];
  const out: AttnState[] = [];
  for (const h of heads) out.push(...headStates(tr, h, r0));
  const concat = concatHeads(tr);
  const tail = {
    head: -1,
    row: S - 1,
    dk: tr.Q[0]!.length / tr.nHeads,
    j: -1,
    dots: [],
    scores: null,
    exps: null,
    sum: null,
    weights: null,
    out: null,
    rowsDone: [],
    concat,
  };
  out.push({ ...tail, phase: "concat", projected: null });
  out.push({ ...tail, phase: "project", projected: tr.output });
  return out;
}

/** The KaTeX term each phase highlights (globals.css `.hl-*`). */
export const ATTN_HL: Record<AttnPhase, string> = {
  dot: "dot",
  scale: "scale",
  mask: "mask",
  exp: "exp",
  sum: "sum",
  softmax: "w",
  wsum: "v",
  concat: "cat",
  project: "wo",
};

/** The caption for one state. `tokens` are the input characters. */
export function attentionCaption(s: AttnState, tokens: string[]): string {
  const tok = (j: number) => `'${showChar(tokens[j] ?? "?")}'`;
  const i = s.row;
  const who = `Head ${s.head}, query ${i} ${tok(i)}`;
  switch (s.phase) {
    case "dot":
      return `${who}: q${i} · k${s.j} ${tok(s.j)} = ${sig(s.dots[s.j]!)}${s.j > i ? " (a future key: computed, then masked)" : ""}.`;
    case "scale":
      return `${who}: every score is divided by √d_k = √${s.dk}, so s${i},0 = ${sig(s.scores![0]!)}.`;
    case "mask": {
      const hidden = s.scores!.filter((v) => v === null).length;
      return hidden === 0
        ? `${who}: the last row sees every key, so the causal mask hides nothing.`
        : `${who}: the causal mask sets the ${hidden} future score${hidden === 1 ? "" : "s"} to −∞.`;
    }
    case "exp": {
      const m = maxIndex(s.scores!);
      return `${who}: exponentials e^(s − max); the largest score, key ${m} ${tok(m)}, gives e⁰ = 1.`;
    }
    case "sum":
      return `${who}: the ${i + 1} exponentials sum to ${sig(s.sum!)}.`;
    case "softmax": {
      const m = maxIndex(s.weights!);
      return `${who}: dividing by the sum gives weights that add up to 1; the largest, ${sig(s.weights![m]!)}, is on key ${m} ${tok(m)}.`;
    }
    case "wsum":
      return `${who}: add ${sig(s.weights![s.j]!)} × v${s.j} ${tok(s.j)}; o${i},0 is now ${sig(s.out![0]!)}.`;
    case "concat":
      return `Concatenate the heads: each row of [head 0 ‖ head 1 ‖ …] has ${s.concat![0]!.length} numbers.`;
    case "project":
      return `Project through W_o: the attention output row ${i} starts ${sig(s.projected![i]![0]!)}, ${sig(s.projected![i]![1]!)}, … and is added to the residual stream.`;
  }
}

function maxIndex(v: (number | null)[]): number {
  let m = 0;
  let best = -Infinity;
  v.forEach((x, j) => {
    if (x !== null && x > best) {
      best = x;
      m = j;
    }
  });
  return m;
}
