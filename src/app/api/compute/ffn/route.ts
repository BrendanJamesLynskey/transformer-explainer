/**
 * POST /api/compute/ffn
 *
 * Body: { text, seed?, seqLen?, dModel?, dFf? }
 *
 * Runs token + positional embeddings → LN1 → attention residual stream →
 * LN2 → FFN, returning the per-stage matrices the widget visualises:
 *   - `input`   : the LN2 output (FFN input)
 *   - `pre`     : input · W1 + b1
 *   - `act`     : GELU(pre)
 *   - `output`  : act · W2 + b2
 */
import { NextResponse } from "next/server";
import { z } from "zod";

import { multiHeadAttention } from "@/lib/transformer/attention";
import {
  positionalEncoding,
  tokenEmbedding,
} from "@/lib/transformer/embeddings";
import { ffn } from "@/lib/transformer/ffn";
import { initModelWeights } from "@/lib/transformer/init";
import { layernormRows } from "@/lib/transformer/layernorm";
import { addMat } from "@/lib/transformer/tensor";
import { ALPHABET, VOCAB_SIZE, encode } from "@/lib/transformer/tokenizer";
import { env } from "@/lib/env";

export const runtime = "nodejs";

const requestSchema = z.object({
  text: z.string().max(200),
  seed: z.coerce.number().int().default(42),
  seqLen: z.coerce.number().int().min(1).max(env.MAX_SEQ_LEN).default(8),
  dModel: z.coerce.number().int().min(2).max(env.MAX_D_MODEL).default(16),
  dFf: z.coerce.number().int().min(2).max(256).default(32),
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
  const { text, seed, seqLen, dModel, dFf, nHeads } = parsed.data;
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
    d_ff: dFf,
    n_blocks: 1,
    vocab_size: VOCAB_SIZE,
    seed,
  };
  const w = initModelWeights(config);
  const block0 = w.blocks[0]!;

  const tokenIds = encode(text, seqLen);
  const tokEmb = tokenEmbedding(tokenIds, w.tok_emb);
  const posEmb = positionalEncoding(seqLen, dModel);
  const x0 = addMat(tokEmb, posEmb);

  // Pre-norm sub-layer 1: residual + attention.
  const ln1 = layernormRows(x0, block0.ln1.gamma, block0.ln1.beta);
  const attnOut = multiHeadAttention(ln1, block0.attn, nHeads);
  const h = addMat(x0, attnOut);

  // Pre-norm sub-layer 2: residual + FFN.
  const ln2 = layernormRows(h, block0.ln2.gamma, block0.ln2.beta);
  const trace = {
    pre: [] as number[][],
    act: [] as number[][],
    out: [] as number[][],
  };
  const ffnOut = ffn(ln2, block0.ffn, trace);

  return NextResponse.json({
    ok: true,
    data: {
      alphabet: ALPHABET,
      tokenIds,
      tokens: tokenIds.map((id) => ALPHABET[id] ?? "?"),
      input: ln2,
      pre: trace.pre,
      act: trace.act,
      output: ffnOut,
      dModel,
      dFf,
    },
  });
}
