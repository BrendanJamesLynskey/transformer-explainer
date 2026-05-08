/**
 * POST /api/compute/block
 *
 * Body: { text, seed?, seqLen?, dModel?, dFf?, nHeads? }
 *
 * Runs the full pre-norm decoder block on the user's input and returns
 * the complete `BlockTrace`. The widget uses it to show every sub-layer's
 * contribution and the residual stream additions.
 */
import { NextResponse } from "next/server";
import { z } from "zod";

import { block } from "@/lib/transformer/block";
import {
  positionalEncoding,
  tokenEmbedding,
} from "@/lib/transformer/embeddings";
import { initModelWeights } from "@/lib/transformer/init";
import { addMat } from "@/lib/transformer/tensor";
import { ALPHABET, VOCAB_SIZE, encode } from "@/lib/transformer/tokenizer";
import { emptyBlockTrace } from "@/lib/transformer/trace";
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

  const tokenIds = encode(text, seqLen);
  const x0 = addMat(
    tokenEmbedding(tokenIds, w.tok_emb),
    positionalEncoding(seqLen, dModel),
  );

  const trace = emptyBlockTrace();
  const out = block(x0, w.blocks[0]!, nHeads, trace);

  return NextResponse.json({
    ok: true,
    data: {
      alphabet: ALPHABET,
      tokenIds,
      tokens: tokenIds.map((id) => ALPHABET[id] ?? "?"),
      input: x0,
      ln1: trace.ln1,
      attnOut: trace.attnOut,
      ln2: trace.ln2,
      ffnOut: trace.ffnOut,
      output: out,
    },
  });
}
