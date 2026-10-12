/*
 * src/lib/training/model-grad.ts
 *
 * Operation: the loss of Part 1's model on a batch, and the gradient of that
 *            loss with respect to every parameter (backpropagation).
 * Shapes:    lossAndGrads(params, config, inputs: [B][S], targets: [B][S])
 *              → { loss, grads: ParamMap (same shapes as params),
 *                  perPosition: [B][S] }
 * Intuition: the forward pass is Part 1's, op for op (the same functions
 *            from src/lib/transformer, so the logits are identical to
 *            `forwardTyped`); it keeps what the backward pass will need
 *            (the activations). The backward pass then walks the same ops
 *            in reverse, turning d loss / d output into d loss / d input and
 *            d loss / d weights at each one (backward-ops.ts). The residual
 *            additions just copy the gradient into both branches.
 *            Checked against scripts/training/model.py, which is checked
 *            against PyTorch autograd and finite differences.
 * MDX:       /training/04-backprop.
 */
import { multiHeadAttention } from "@/lib/transformer/attention";
import {
  positionalEncoding,
  tokenEmbedding,
} from "@/lib/transformer/embeddings";
import { ffn, type FFNTrace } from "@/lib/transformer/ffn";
import { layernormRows } from "@/lib/transformer/layernorm";
import { matmul, transpose } from "@/lib/transformer/matmul";
import type { ModelConfig } from "@/lib/transformer/model";
import { addMat } from "@/lib/transformer/tensor";
import {
  emptyAttentionTrace,
  type AttentionTrace,
} from "@/lib/transformer/trace";
import type { Matrix, Vector } from "@/lib/transformer/types";

import {
  addInto,
  colSums,
  geluBackward,
  layernormBackward,
  linearBackward,
  mm,
  mmTA,
  mmTB,
  softmaxBackward,
} from "./backward-ops";
import { crossEntropy, crossEntropyGrad } from "./loss";
import { zerosLike, type ParamMap } from "./params";

type Arch = Pick<ModelConfig, "n_heads" | "n_blocks" | "d_model">;

/** What one block's forward pass keeps for the backward pass. */
export type BlockCache = {
  x: Matrix; // block input
  ln1: Matrix;
  attn: AttentionTrace; // Q, K, V, per-head weights
  h: Matrix; // x + Attn(LN1(x))
  ln2: Matrix;
  ffn: FFNTrace; // pre = LN2(h) W1 + b1, act = GELU(pre)
};

/** The forward pass of one sequence, with its saved activations. */
export type SequenceCache = {
  ids: number[];
  blocks: BlockCache[];
  xLast: Matrix; // input to the final LayerNorm
  xf: Matrix; // LN_f(x)
  logits: Matrix; // [S, V]
};

const m = (p: ParamMap, k: string) => p[k] as Matrix;
const v = (p: ParamMap, k: string) => p[k] as Vector;

/** Forward pass of one sequence (Part 1's ops, in `forwardTyped`'s order). */
export function forwardCached(
  params: ParamMap,
  config: Arch,
  ids: number[],
): SequenceCache {
  const E = m(params, "tok_emb");
  let x = addMat(
    tokenEmbedding(ids, E),
    positionalEncoding(ids.length, config.d_model),
  );
  const blocks: BlockCache[] = [];
  for (let i = 0; i < config.n_blocks; i++) {
    const p = `block_${i}`;
    const ln1 = layernormRows(
      x,
      v(params, `${p}.ln1.gamma`),
      v(params, `${p}.ln1.beta`),
    );
    const attn = emptyAttentionTrace();
    const a = multiHeadAttention(
      ln1,
      {
        W_q: m(params, `${p}.attn.W_q`),
        W_k: m(params, `${p}.attn.W_k`),
        W_v: m(params, `${p}.attn.W_v`),
        W_o: m(params, `${p}.attn.W_o`),
      },
      config.n_heads,
      attn,
    );
    const h = addMat(x, a);
    const ln2 = layernormRows(
      h,
      v(params, `${p}.ln2.gamma`),
      v(params, `${p}.ln2.beta`),
    );
    const f: FFNTrace = { pre: [], act: [], out: [] };
    const out = ffn(
      ln2,
      {
        W1: m(params, `${p}.ffn.W1`),
        b1: v(params, `${p}.ffn.b1`),
        W2: m(params, `${p}.ffn.W2`),
        b2: v(params, `${p}.ffn.b2`),
      },
      f,
    );
    blocks.push({ x, ln1, attn, h, ln2, ffn: f });
    x = addMat(h, out);
  }
  const xf = layernormRows(
    x,
    v(params, "ln_final.gamma"),
    v(params, "ln_final.beta"),
  );
  const logits = matmul(xf, transpose(E));
  return { ids, blocks, xLast: x, xf, logits };
}

/** One named gradient on the way back (for the chapter 4 animation). */
export type GradRecord = { name: string; value: Matrix };

function addTo(acc: Matrix, g: Matrix): void {
  for (let i = 0; i < g.length; i++) {
    const a = acc[i]!;
    const r = g[i]!;
    for (let j = 0; j < r.length; j++) a[j]! += r[j]!;
  }
}
function addVecTo(acc: Vector, g: Vector): void {
  for (let j = 0; j < g.length; j++) acc[j]! += g[j]!;
}

/** Slice columns [c0, c0 + n) of a matrix. */
function cols(a: Matrix, c0: number, n: number): Matrix {
  return a.map((r) => r.slice(c0, c0 + n));
}

/**
 * Backward pass of one sequence: adds its parameter gradients into `grads`.
 *
 * @param dlogits  d loss / d logits for this sequence `[S, V]`.
 * @param record   If given, receives the gradient at each stop on the way
 *                 back (names as scripts/training/model.py records them).
 */
export function backwardSequence(
  params: ParamMap,
  config: Arch,
  c: SequenceCache,
  dlogits: Matrix,
  grads: ParamMap,
  record?: GradRecord[],
): void {
  const E = m(params, "tok_emb");
  const S = c.ids.length;
  const D = config.d_model;
  const H = config.n_heads;
  const dk = D / H;
  const scale = 1 / Math.sqrt(Math.max(1, dk));

  // logits = xf Eᵀ (the tied head): dE += dlogitsᵀ xf, dxf = dlogits E
  record?.push({ name: "dlogits", value: dlogits });
  addTo(m(grads, "tok_emb"), mmTA(dlogits, c.xf));
  const dxf = mm(dlogits, E);
  record?.push({ name: "dxf", value: dxf });
  const lnf = layernormBackward(c.xLast, v(params, "ln_final.gamma"), dxf);
  addVecTo(v(grads, "ln_final.gamma"), lnf.dGamma);
  addVecTo(v(grads, "ln_final.beta"), lnf.dBeta);
  let dx = lnf.dX;
  record?.push({ name: "dx_top", value: dx });

  for (let i = config.n_blocks - 1; i >= 0; i--) {
    const p = `block_${i}`;
    const b = c.blocks[i]!;

    // y = h + FFN(LN2(h)), FFN(z) = GELU(z W1 + b1) W2 + b2
    const l2 = linearBackward(b.ffn.act, m(params, `${p}.ffn.W2`), dx);
    addTo(m(grads, `${p}.ffn.W2`), l2.dW);
    addVecTo(v(grads, `${p}.ffn.b2`), colSums(dx));
    record?.push({ name: `${p}.dact`, value: l2.dX });
    const dpre = geluBackward(b.ffn.pre, l2.dX);
    record?.push({ name: `${p}.dpre`, value: dpre });
    const l1 = linearBackward(b.ln2, m(params, `${p}.ffn.W1`), dpre);
    addTo(m(grads, `${p}.ffn.W1`), l1.dW);
    addVecTo(v(grads, `${p}.ffn.b1`), colSums(dpre));
    record?.push({ name: `${p}.dln2`, value: l1.dX });
    const n2 = layernormBackward(b.h, v(params, `${p}.ln2.gamma`), l1.dX);
    addVecTo(v(grads, `${p}.ln2.gamma`), n2.dGamma);
    addVecTo(v(grads, `${p}.ln2.beta`), n2.dBeta);
    // the residual: the gradient reaches h directly and through the FFN
    const dh = addInto(dx, n2.dX);
    record?.push({ name: `${p}.dh`, value: dh });

    // h = x + C W_o, C = [P_0 V_0 | … | P_{H-1} V_{H-1}], P = softmax(Q Kᵀ / √d_k)
    const heads: Matrix[] = [];
    for (let hd = 0; hd < H; hd++) {
      heads.push(mm(b.attn.weights[hd]!, cols(b.attn.V, hd * dk, dk)));
    }
    const C = heads[0]!.map((_, r) => heads.flatMap((o) => o[r]!));
    const lo = linearBackward(C, m(params, `${p}.attn.W_o`), dh);
    addTo(m(grads, `${p}.attn.W_o`), lo.dW);
    const dQ: Matrix = Array.from({ length: S }, () =>
      new Array<number>(D).fill(0),
    );
    const dK: Matrix = Array.from({ length: S }, () =>
      new Array<number>(D).fill(0),
    );
    const dV: Matrix = Array.from({ length: S }, () =>
      new Array<number>(D).fill(0),
    );
    for (let hd = 0; hd < H; hd++) {
      const P = b.attn.weights[hd]!;
      const dOh = cols(lo.dX, hd * dk, dk);
      const Qh = cols(b.attn.Q, hd * dk, dk);
      const Kh = cols(b.attn.K, hd * dk, dk);
      const Vh = cols(b.attn.V, hd * dk, dk);
      const dP = mmTB(dOh, Vh);
      const dVh = mmTA(P, dOh);
      const dS = softmaxBackward(P, dP);
      if (hd === 0) record?.push({ name: `${p}.dscores_h0`, value: dS });
      const dQh = mm(dS, Kh);
      const dKh = mmTA(dS, Qh);
      for (let r = 0; r < S; r++) {
        for (let k = 0; k < dk; k++) {
          dQ[r]![hd * dk + k] = dQh[r]![k]! * scale;
          dK[r]![hd * dk + k] = dKh[r]![k]! * scale;
          dV[r]![hd * dk + k] = dVh[r]![k]!;
        }
      }
    }
    const lq = linearBackward(b.ln1, m(params, `${p}.attn.W_q`), dQ);
    const lk = linearBackward(b.ln1, m(params, `${p}.attn.W_k`), dK);
    const lv = linearBackward(b.ln1, m(params, `${p}.attn.W_v`), dV);
    addTo(m(grads, `${p}.attn.W_q`), lq.dW);
    addTo(m(grads, `${p}.attn.W_k`), lk.dW);
    addTo(m(grads, `${p}.attn.W_v`), lv.dW);
    const dln1 = addInto(addInto(lq.dX, lk.dX), lv.dX);
    record?.push({ name: `${p}.dln1`, value: dln1 });
    const n1 = layernormBackward(b.x, v(params, `${p}.ln1.gamma`), dln1);
    addVecTo(v(grads, `${p}.ln1.gamma`), n1.dGamma);
    addVecTo(v(grads, `${p}.ln1.beta`), n1.dBeta);
    dx = addInto(dh, n1.dX);
    record?.push({ name: `${p}.dx`, value: dx });
  }

  // x0 = E[ids] + P: each position's gradient lands on its token's row
  const dE = m(grads, "tok_emb");
  for (let r = 0; r < S; r++) {
    const row = dE[c.ids[r]!]!;
    const g = dx[r]!;
    for (let k = 0; k < D; k++) row[k]! += g[k]!;
  }
}

/** Loss, per-position losses and the gradients for a batch. */
export function lossAndGrads(
  params: ParamMap,
  config: Arch,
  inputs: number[][],
  targets: number[][],
  record?: GradRecord[],
): { loss: number; grads: ParamMap; perPosition: number[][] } {
  const grads = zerosLike(params);
  const n = inputs.length * (inputs[0]?.length ?? 0);
  let total = 0;
  const perPosition: number[][] = [];
  inputs.forEach((ids, bi) => {
    const c = forwardCached(params, config, ids);
    const ce = crossEntropy(c.logits, targets[bi]!);
    perPosition.push(ce.perPosition);
    for (const l of ce.perPosition) total += l;
    const dl = crossEntropyGrad(ce.probs, targets[bi]!, n);
    backwardSequence(
      params,
      config,
      c,
      dl,
      grads,
      bi === 0 ? record : undefined,
    );
  });
  return { loss: total / n, grads, perPosition };
}

/** The loss alone (no gradients), e.g. for validation. */
export function lossOnly(
  params: ParamMap,
  config: Arch,
  inputs: number[][],
  targets: number[][],
): number {
  const n = inputs.length * (inputs[0]?.length ?? 0);
  let total = 0;
  inputs.forEach((ids, bi) => {
    const c = forwardCached(params, config, ids);
    for (const l of crossEntropy(c.logits, targets[bi]!).perPosition)
      total += l;
  });
  return total / n;
}
