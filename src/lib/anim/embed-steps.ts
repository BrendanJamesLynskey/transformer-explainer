/**
 * The embeddings animation (chapter 02) as a list of model states.
 *
 * For one chosen position p the animation shows the lookup (token id t
 * selects row t of the embedding table E), then the positional vector
 * P[p], then the sum x_p = E[t] + P[p] built element by element; after
 * that the other positions' rows of X fill in one per step.
 *
 * Every number comes from the trace of `/api/compute/embed`
 * (`computeEmbed`, run in the browser); `scripts/reference.py` builds the
 * same states from its own forward pass and the frame tests compare them.
 */
import { showChar, sig } from "./format";

/** What the embed trace provides (a subset of `computeEmbed`'s result). */
export type EmbedTrace = {
  tokenIds: number[];
  tokens: string[];
  tokEmb: number[][];
  posEmb: number[][];
  xAfterEmb: number[][];
};

export type EmbedKind = "ids" | "lookup" | "position" | "add" | "row";

export type EmbedState = {
  kind: EmbedKind;
  /** The position being walked through. */
  pos: number;
  id: number;
  char: string;
  /** Last element of the sum computed (`add`), else −1. */
  elem: number;
  /** E[id]: the row the lookup selects. */
  tok: number[];
  /** P[pos]. */
  pe: number[];
  /** x_pos so far: elements up to `elem` filled (null = not yet added). */
  sum: (number | null)[];
  /** Rows of X = E + P complete, in the order they were completed. */
  done: number[];
  /** Sequence length S (rows of X). */
  total: number;
  /** The row filled by a `row` step and its character (else −1, ""). */
  row: number;
  rowChar: string;
};

/** All states for walking through position `pos`. */
export function embedStates(tr: EmbedTrace, pos: number): EmbedState[] {
  const S = tr.tokenIds.length;
  const p = Math.max(0, Math.min(S - 1, pos));
  const tok = tr.tokEmb[p]!;
  const pe = tr.posEmb[p]!;
  const x = tr.xAfterEmb[p]!;
  const D = tok.length;
  const base = {
    pos: p,
    id: tr.tokenIds[p]!,
    char: tr.tokens[p]!,
    elem: -1,
    tok,
    pe,
    total: S,
    row: -1,
    rowChar: "",
  };
  const empty = (): (number | null)[] => new Array<number | null>(D).fill(null);
  const out: EmbedState[] = [
    { ...base, kind: "ids", sum: empty(), done: [] },
    { ...base, kind: "lookup", sum: empty(), done: [] },
    { ...base, kind: "position", sum: empty(), done: [] },
  ];
  for (let e = 0; e < D; e++) {
    const sum = empty();
    // x_p,k = E[t]_k + P[p]_k, the value the model computed (addMat)
    for (let k = 0; k <= e; k++) sum[k] = x[k]!;
    out.push({
      ...base,
      kind: "add",
      elem: e,
      sum,
      done: e === D - 1 ? [p] : [],
    });
  }
  const done = [p];
  for (let q = 0; q < S; q++) {
    if (q === p) continue;
    done.push(q);
    out.push({
      ...base,
      kind: "row",
      sum: x.slice(),
      done: done.slice(),
      row: q,
      rowChar: tr.tokens[q]!,
    });
  }
  return out;
}

/** The caption for one state (also the aria-live text). */
export function embedCaption(s: EmbedState): string {
  const c = `'${showChar(s.char)}'`;
  switch (s.kind) {
    case "ids":
      return `Position ${s.pos} holds ${c}, token id ${s.id}. The id is just a row number.`;
    case "lookup":
      return `Lookup: id ${s.id} selects row ${s.id} of the embedding table E, a vector of ${s.tok.length} numbers (E[${s.id}]₀ = ${sig(s.tok[0]!)}).`;
    case "position":
      return `The positional vector P[${s.pos}] depends only on the position: P[${s.pos}]₀ = sin(${s.pos}) = ${sig(s.pe[0]!)}, P[${s.pos}]₁ = cos(${s.pos}) = ${sig(s.pe[1]!)}.`;
    case "add": {
      const k = s.elem;
      return `Element ${k}: ${sig(s.tok[k]!)} + ${sig(s.pe[k]!)} = ${sig(s.sum[k]!)}.`;
    }
    case "row":
      return `Row ${s.row} ('${showChar(s.rowChar)}') of X = E + P, the same lookup and add: ${s.done.length} of ${s.total} rows done.`;
  }
}
