/**
 * POST /api/compute/embed
 *
 * Body: { text: string; seed?: number; seqLen?: number; dModel?: number }
 *
 * Tokenises the input, looks up token embeddings from a deterministic
 * seed-based init, builds the sinusoidal positional encoding, and returns
 * everything as a trace the client visualisations can render.
 */
import { NextResponse } from "next/server";
import { z } from "zod";

import { initModelWeights } from "@/lib/transformer/init";
import {
  positionalEncoding,
  tokenEmbedding,
} from "@/lib/transformer/embeddings";
import { ALPHABET, VOCAB_SIZE, encode } from "@/lib/transformer/tokenizer";
import { addMat } from "@/lib/transformer/tensor";
import { env } from "@/lib/env";

export const runtime = "nodejs";

const requestSchema = z.object({
  text: z.string().max(200),
  // Defaults match SPEC §4 — kept aligned with the API surface in later phases.
  seed: z.coerce.number().int().default(42),
  seqLen: z.coerce.number().int().min(1).max(env.MAX_SEQ_LEN).default(8),
  dModel: z.coerce.number().int().min(2).max(env.MAX_D_MODEL).default(16),
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
  const { text, seed, seqLen, dModel } = parsed.data;

  const config = {
    seq_len: seqLen,
    d_model: dModel,
    n_heads: 1,
    d_ff: 1, // unused for embed-only init, but model.ts needs the field
    n_blocks: 0,
    vocab_size: VOCAB_SIZE,
    seed,
  };
  const weights = initModelWeights(config);

  const tokenIds = encode(text, seqLen);
  const tokEmb = tokenEmbedding(tokenIds, weights.tok_emb);
  const posEmb = positionalEncoding(seqLen, dModel);
  const xAfterEmb = addMat(tokEmb, posEmb);

  return NextResponse.json({
    ok: true,
    data: {
      alphabet: ALPHABET,
      tokenIds,
      tokens: tokenIds.map((id) => ALPHABET[id] ?? "?"),
      tokEmb,
      posEmb,
      xAfterEmb,
    },
  });
}
