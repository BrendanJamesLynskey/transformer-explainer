/**
 * POST /api/compute/attention
 *
 * Body: { text, seed?, seqLen?, dModel?, nHeads? }
 *
 * Runs the same attention path the decoder uses on a single block's
 * worth of weights, with the standard pre-norm LayerNorm in front, and
 * returns the full trace. The visualisation renders these arrays
 * directly — there is no parallel "demo model".
 */
import { NextResponse } from "next/server";
import { z } from "zod";

import { causalMask, multiHeadAttention } from "@/lib/transformer/attention";
import {
  positionalEncoding,
  tokenEmbedding,
} from "@/lib/transformer/embeddings";
import { initModelWeights } from "@/lib/transformer/init";
import { layernormRows } from "@/lib/transformer/layernorm";
import { addMat } from "@/lib/transformer/tensor";
import { ALPHABET, VOCAB_SIZE, encode } from "@/lib/transformer/tokenizer";
import { emptyAttentionTrace } from "@/lib/transformer/trace";
import { env } from "@/lib/env";

export const runtime = "nodejs";

const requestSchema = z.object({
  text: z.string().max(200),
  seed: z.coerce.number().int().default(42),
  seqLen: z.coerce.number().int().min(1).max(env.MAX_SEQ_LEN).default(8),
  dModel: z.coerce.number().int().min(2).max(env.MAX_D_MODEL).default(16),
  nHeads: z.coerce.number().int().min(1).max(8).default(2),
});

export async function POST(req: Request): Promise<Response> {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json(
      { ok: false, error: "Invalid JSON body" },
      { status: 400 },
    );
  }

  const parsed = requestSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      { ok: false, error: parsed.error.issues[0]?.message ?? "Invalid input" },
      { status: 400 },
    );
  }
  const { text, seed, seqLen, dModel, nHeads } = parsed.data;
  if (dModel % nHeads !== 0) {
    return NextResponse.json(
      {
        ok: false,
        error: `dModel (${dModel}) must be divisible by nHeads (${nHeads})`,
      },
      { status: 400 },
    );
  }

  const config = {
    seq_len: seqLen,
    d_model: dModel,
    n_heads: nHeads,
    d_ff: 1, // unused at this stage but model.ts expects the field
    n_blocks: 1,
    vocab_size: VOCAB_SIZE,
    seed,
  };
  const weights = initModelWeights(config);

  const tokenIds = encode(text, seqLen);
  const tokEmb = tokenEmbedding(tokenIds, weights.tok_emb);
  const posEmb = positionalEncoding(seqLen, dModel);
  const xAfterEmb = addMat(tokEmb, posEmb);

  // Pre-norm: the input to attention is the LayerNorm of the residual
  // stream. SPEC §4.
  const block0 = weights.blocks[0]!;
  const ln1 = layernormRows(xAfterEmb, block0.ln1.gamma, block0.ln1.beta);

  const trace = emptyAttentionTrace();
  const output = multiHeadAttention(ln1, block0.attn, nHeads, trace);

  return NextResponse.json({
    ok: true,
    data: {
      alphabet: ALPHABET,
      tokenIds,
      tokens: tokenIds.map((id) => ALPHABET[id] ?? "?"),
      ln1,
      Q: trace.Q,
      K: trace.K,
      V: trace.V,
      mask: causalMask(seqLen),
      scores: trace.scores, // [n_heads, seq_len, seq_len]
      weights: trace.weights, // [n_heads, seq_len, seq_len]
      output,
      nHeads,
    },
  });
}
