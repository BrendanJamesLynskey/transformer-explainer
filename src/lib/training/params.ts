/*
 * src/lib/training/params.ts
 *
 * Operation: the model's weights as a flat map of named tensors, the form
 *            the optimisers, the gradients and the Python fixtures share.
 * Shapes:    ParamMap = { "tok_emb": [V, D], "block_0.attn.W_q": [D, D], … }
 *            (the names scripts/reference.py and scripts/training use).
 * Intuition: training needs to walk every parameter in a fixed order
 *            (update it, add up its squared gradient). A flat map in
 *            `paramNames` order makes that one loop; `toWeights` turns it
 *            back into the nested `ModelWeights` Part 1's forward pass takes.
 * MDX:       /training/05-optimisers.
 */
import type { BlockWeights } from "@/lib/transformer/block";
import type { ModelConfig, ModelWeights } from "@/lib/transformer/model";
import type { Matrix, Vector } from "@/lib/transformer/types";

/** A parameter tensor: a matrix (2-D) or a vector (1-D). */
export type Tensor = Matrix | Vector;
export type ParamMap = Record<string, Tensor>;

/** Is this tensor a matrix? (Weight decay applies to matrices only.) */
export function isMatrix(t: Tensor): t is Matrix {
  return Array.isArray(t[0]);
}

/** Parameter names in the fixed order every loop uses (as `model.py`). */
export function paramNames(config: Pick<ModelConfig, "n_blocks">): string[] {
  const names = ["tok_emb"];
  for (let i = 0; i < config.n_blocks; i++) {
    const p = `block_${i}`;
    names.push(`${p}.ln1.gamma`, `${p}.ln1.beta`);
    for (const m of ["W_q", "W_k", "W_v", "W_o"]) names.push(`${p}.attn.${m}`);
    names.push(`${p}.ln2.gamma`, `${p}.ln2.beta`);
    names.push(`${p}.ffn.W1`, `${p}.ffn.b1`, `${p}.ffn.W2`, `${p}.ffn.b2`);
  }
  names.push("ln_final.gamma", "ln_final.beta");
  return names;
}

/** Flatten nested weights into a ParamMap (shares the arrays). */
export function fromWeights(w: ModelWeights): ParamMap {
  const out: ParamMap = { tok_emb: w.tok_emb };
  w.blocks.forEach((b, i) => {
    const p = `block_${i}`;
    out[`${p}.ln1.gamma`] = b.ln1.gamma;
    out[`${p}.ln1.beta`] = b.ln1.beta;
    out[`${p}.attn.W_q`] = b.attn.W_q;
    out[`${p}.attn.W_k`] = b.attn.W_k;
    out[`${p}.attn.W_v`] = b.attn.W_v;
    out[`${p}.attn.W_o`] = b.attn.W_o;
    out[`${p}.ln2.gamma`] = b.ln2.gamma;
    out[`${p}.ln2.beta`] = b.ln2.beta;
    out[`${p}.ffn.W1`] = b.ffn.W1;
    out[`${p}.ffn.b1`] = b.ffn.b1;
    out[`${p}.ffn.W2`] = b.ffn.W2;
    out[`${p}.ffn.b2`] = b.ffn.b2;
  });
  out["ln_final.gamma"] = w.ln_final.gamma;
  out["ln_final.beta"] = w.ln_final.beta;
  return out;
}

/** Rebuild nested weights from a ParamMap (shares the arrays). */
export function toWeights(p: ParamMap, nBlocks: number): ModelWeights {
  const m = (k: string) => p[k] as Matrix;
  const v = (k: string) => p[k] as Vector;
  const blocks: BlockWeights[] = [];
  for (let i = 0; i < nBlocks; i++) {
    const q = `block_${i}`;
    blocks.push({
      ln1: { gamma: v(`${q}.ln1.gamma`), beta: v(`${q}.ln1.beta`) },
      attn: {
        W_q: m(`${q}.attn.W_q`),
        W_k: m(`${q}.attn.W_k`),
        W_v: m(`${q}.attn.W_v`),
        W_o: m(`${q}.attn.W_o`),
      },
      ln2: { gamma: v(`${q}.ln2.gamma`), beta: v(`${q}.ln2.beta`) },
      ffn: {
        W1: m(`${q}.ffn.W1`),
        b1: v(`${q}.ffn.b1`),
        W2: m(`${q}.ffn.W2`),
        b2: v(`${q}.ffn.b2`),
      },
    });
  }
  return {
    tok_emb: m("tok_emb"),
    blocks,
    ln_final: { gamma: v("ln_final.gamma"), beta: v("ln_final.beta") },
  };
}

/** A deep copy of one tensor. */
export function cloneTensor(t: Tensor): Tensor {
  return isMatrix(t) ? t.map((r) => r.slice()) : (t as Vector).slice();
}

/** A deep copy of a ParamMap (same key order). */
export function cloneParams(p: ParamMap): ParamMap {
  const out: ParamMap = {};
  for (const k of Object.keys(p)) out[k] = cloneTensor(p[k]!);
  return out;
}

/** Zeros with the same shapes as `p`. */
export function zerosLike(p: ParamMap): ParamMap {
  const out: ParamMap = {};
  for (const k of Object.keys(p)) {
    const t = p[k]!;
    out[k] = isMatrix(t)
      ? t.map((r) => new Array<number>(r.length).fill(0))
      : new Array<number>(t.length).fill(0);
  }
  return out;
}

/** Sum of squares of one tensor's entries. */
export function sumSquares(t: Tensor): number {
  let s = 0;
  if (isMatrix(t)) {
    for (const r of t) for (const x of r) s += x * x;
  } else {
    for (const x of t as Vector) s += x * x;
  }
  return s;
}

/** The number of entries in a ParamMap. */
export function countParams(p: ParamMap): number {
  let n = 0;
  for (const t of Object.values(p)) {
    n += isMatrix(t) ? t.length * (t[0]?.length ?? 0) : t.length;
  }
  return n;
}
