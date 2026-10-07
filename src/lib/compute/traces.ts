/*
 * src/lib/compute/traces.ts
 *
 * The traces behind the `/api/compute/*` routes, as plain functions.
 *
 * The routes validate their input and call these; the animations call the
 * same functions in the browser, so stepping through an animation needs no
 * network round trip and still shows exactly what the API would have
 * returned. `tests/unit/compute-traces.test.ts` checks that a route's
 * response equals the client's trace for the same input.
 *
 * Every function here only composes `src/lib/transformer/*`; none of the
 * maths lives in this file.
 */
import { causalMask, multiHeadAttention } from "@/lib/transformer/attention";
import { block } from "@/lib/transformer/block";
import {
  positionalEncoding,
  tokenEmbedding,
} from "@/lib/transformer/embeddings";
import { ffn } from "@/lib/transformer/ffn";
import { initModelWeights } from "@/lib/transformer/init";
import { layernormRows } from "@/lib/transformer/layernorm";
import {
  forwardTyped,
  type ModelConfig,
  type ModelWeights,
} from "@/lib/transformer/model";
import { addMat } from "@/lib/transformer/tensor";
import { ALPHABET, VOCAB_SIZE, encode } from "@/lib/transformer/tokenizer";
import {
  emptyAttentionTrace,
  emptyBlockTrace,
  type ForwardTrace,
} from "@/lib/transformer/trace";
import type { Matrix } from "@/lib/transformer/types";

/** Inputs shared by every compute route (already validated). */
export type ComputeParams = {
  text: string;
  seed: number;
  seqLen: number;
  dModel: number;
  nHeads: number;
  dFf: number;
  nBlocks: number;
};

/** The defaults the routes use (SPEC §4). */
export const DEFAULT_PARAMS: ComputeParams = {
  text: "hello!",
  seed: 42,
  seqLen: 8,
  dModel: 16,
  nHeads: 2,
  dFf: 32,
  nBlocks: 2,
};

/**
 * The model config for a request. Routes that run only part of the model
 * build a smaller one, exactly as they always have (the number of blocks
 * and `d_ff` change the order of the seeded weight draws, so they matter).
 */
export function configFor(
  p: ComputeParams,
  over: Partial<ModelConfig> = {},
): ModelConfig {
  return {
    seq_len: p.seqLen,
    d_model: p.dModel,
    n_heads: p.nHeads,
    d_ff: p.dFf,
    n_blocks: p.nBlocks,
    vocab_size: VOCAB_SIZE,
    seed: p.seed,
    ...over,
  };
}

/** Token ids and their characters, as every route returns them. */
function tokensOf(text: string, seqLen: number) {
  const tokenIds = encode(text, seqLen);
  return {
    alphabet: ALPHABET,
    tokenIds,
    tokens: tokenIds.map((id) => ALPHABET[id] ?? "?"),
  };
}

/** Token embedding + positional encoding: the input to block 0. */
function embedInput(tokenIds: number[], w: ModelWeights, p: ComputeParams) {
  const tokEmb = tokenEmbedding(tokenIds, w.tok_emb);
  const posEmb = positionalEncoding(p.seqLen, p.dModel);
  return { tokEmb, posEmb, x0: addMat(tokEmb, posEmb) };
}

/** `POST /api/compute/embed`: ids → token embeddings + positions. */
export function computeEmbed(p: ComputeParams) {
  const w = initModelWeights(
    configFor(p, { n_heads: 1, d_ff: 1, n_blocks: 0 }),
  );
  const t = tokensOf(p.text, p.seqLen);
  const { tokEmb, posEmb, x0 } = embedInput(t.tokenIds, w, p);
  return { ...t, tokEmb, posEmb, xAfterEmb: x0 };
}

/** `POST /api/compute/attention`: LN₁ then block 0's attention, traced. */
export function computeAttention(p: ComputeParams) {
  const w = initModelWeights(configFor(p, { d_ff: 1, n_blocks: 1 }));
  const t = tokensOf(p.text, p.seqLen);
  const { x0 } = embedInput(t.tokenIds, w, p);
  const block0 = w.blocks[0]!;
  // Pre-norm: attention reads the LayerNorm of the residual stream.
  const ln1 = layernormRows(x0, block0.ln1.gamma, block0.ln1.beta);
  const trace = emptyAttentionTrace();
  const output = multiHeadAttention(ln1, block0.attn, p.nHeads, trace);
  return {
    ...t,
    ln1,
    Q: trace.Q,
    K: trace.K,
    V: trace.V,
    mask: causalMask(p.seqLen),
    scores: trace.scores, // [n_heads, seq_len, seq_len]
    weights: trace.weights, // [n_heads, seq_len, seq_len]
    output,
    nHeads: p.nHeads,
  };
}

/** `POST /api/compute/ffn`: block 0 up to and including its FFN. */
export function computeFfn(p: ComputeParams) {
  const w = initModelWeights(configFor(p, { n_blocks: 1 }));
  const block0 = w.blocks[0]!;
  const t = tokensOf(p.text, p.seqLen);
  const { x0 } = embedInput(t.tokenIds, w, p);
  // Sub-layer 1: residual + attention.
  const ln1 = layernormRows(x0, block0.ln1.gamma, block0.ln1.beta);
  const h = addMat(x0, multiHeadAttention(ln1, block0.attn, p.nHeads));
  // Sub-layer 2: the FFN reads LN₂ of the stream.
  const ln2 = layernormRows(h, block0.ln2.gamma, block0.ln2.beta);
  const trace = { pre: [] as Matrix, act: [] as Matrix, out: [] as Matrix };
  const output = ffn(ln2, block0.ffn, trace);
  return {
    ...t,
    input: ln2,
    pre: trace.pre,
    act: trace.act,
    output,
    dModel: p.dModel,
    dFf: p.dFf,
  };
}

/** `POST /api/compute/block`: one full pre-norm block, traced. */
export function computeBlock(p: ComputeParams) {
  const w = initModelWeights(configFor(p, { n_blocks: 1 }));
  const t = tokensOf(p.text, p.seqLen);
  const { x0 } = embedInput(t.tokenIds, w, p);
  const trace = emptyBlockTrace();
  const output = block(x0, w.blocks[0]!, p.nHeads, trace);
  return {
    ...t,
    input: x0,
    ln1: trace.ln1,
    attnOut: trace.attnOut,
    ln2: trace.ln2,
    ffnOut: trace.ffnOut,
    output,
  };
}

/** A full forward pass with every block traced. */
export function tracedForward(
  tokenIds: number[],
  config: ModelConfig,
  w: ModelWeights,
): ForwardTrace {
  const trace: ForwardTrace = {
    tokenIds: [],
    tokEmb: [],
    posEmb: [],
    // emptyBlockTrace shares one empty matrix between sibling fields, and
    // forwardTyped overwrites them all; the attention trace gets its own.
    blocks: Array.from({ length: config.n_blocks }, () => ({
      ...emptyBlockTrace(),
      attn: emptyAttentionTrace(),
    })),
    xFinal: [],
    logits: [],
  };
  forwardTyped(tokenIds, config, w, trace);
  return trace;
}

/** `POST /api/compute/forward`: the whole model, every block traced. */
export function computeForward(p: ComputeParams) {
  const config = configFor(p);
  const w = initModelWeights(config);
  const t = tokensOf(p.text, p.seqLen);
  const trace = tracedForward(t.tokenIds, config, w);
  return {
    ...t,
    tokEmb: trace.tokEmb,
    posEmb: trace.posEmb,
    xFinal: trace.xFinal,
    logits: trace.logits,
    // Just the per-head softmax weights, for the stacking widget.
    perBlockAttention: trace.blocks.map((bt) => bt.attn.weights), // [nBlocks, nHeads, S, S]
    config,
  };
}
