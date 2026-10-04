/**
 * The attention weights behind the landing page's preview picture.
 *
 * Runs exactly the path `/api/compute/attention` runs — tokenise, embed,
 * add positional encoding, LayerNorm, multi-head causal attention — and
 * returns one head's softmax weights. The landing page renders them as a
 * static SVG on the server, so the picture is real output of
 * `lib/transformer`, not a drawing, and it costs the browser no JavaScript.
 */
import { multiHeadAttention } from "@/lib/transformer/attention";
import {
  positionalEncoding,
  tokenEmbedding,
} from "@/lib/transformer/embeddings";
import { initModelWeights } from "@/lib/transformer/init";
import { layernormRows } from "@/lib/transformer/layernorm";
import { addMat } from "@/lib/transformer/tensor";
import { ALPHABET, VOCAB_SIZE, encode } from "@/lib/transformer/tokenizer";
import { emptyAttentionTrace } from "@/lib/transformer/trace";
import type { Matrix } from "@/lib/transformer/types";

export type AttentionPreview = {
  /** One character per position (the toy tokenizer is character-level). */
  tokens: string[];
  /** Softmax weights for the chosen head, shape `[S, S]`; rows sum to 1. */
  weights: Matrix;
};

/**
 * Compute one head's attention weights for `text` with seeded weights.
 *
 * @param text    Input string; its length sets the sequence length.
 * @param seed    Weight-initialisation seed (the widgets default to 42).
 * @param dModel  Model width; must be divisible by `nHeads`.
 * @param nHeads  Number of heads.
 * @param head    Which head's weights to return.
 */
export function attentionPreview({
  text,
  seed = 42,
  dModel = 16,
  nHeads = 2,
  head = 0,
}: {
  text: string;
  seed?: number;
  dModel?: number;
  nHeads?: number;
  head?: number;
}): AttentionPreview {
  const seqLen = text.length;
  const weights = initModelWeights({
    seq_len: seqLen,
    d_model: dModel,
    n_heads: nHeads,
    d_ff: 1, // the FFN isn't run here
    n_blocks: 1,
    vocab_size: VOCAB_SIZE,
    seed,
  });
  const tokenIds = encode(text, seqLen);
  const x = addMat(
    tokenEmbedding(tokenIds, weights.tok_emb),
    positionalEncoding(seqLen, dModel),
  );
  const block0 = weights.blocks[0]!;
  const ln1 = layernormRows(x, block0.ln1.gamma, block0.ln1.beta);
  const trace = emptyAttentionTrace();
  multiHeadAttention(ln1, block0.attn, nHeads, trace);
  return {
    tokens: tokenIds.map((id) => ALPHABET[id] ?? "?"),
    weights: trace.weights[head] ?? [],
  };
}
