/**
 * POST /api/compute/forward
 *
 * Body: { text, seed?, seqLen?, dModel?, dFf?, nHeads?, nBlocks? }
 *
 * Runs the full N-block forward pass and returns the complete trace —
 * per-block attention/FFN intermediates, the final residual stream, and
 * the next-token logits at every position. Used by the stacking widget,
 * the sampling widget, and the /playground page.
 */
import { NextResponse } from "next/server";
import { z } from "zod";

import { initModelWeights } from "@/lib/transformer/init";
import { forwardTyped } from "@/lib/transformer/model";
import { ALPHABET, VOCAB_SIZE, encode } from "@/lib/transformer/tokenizer";
import { emptyAttentionTrace, emptyBlockTrace } from "@/lib/transformer/trace";
import { env } from "@/lib/env";

export const runtime = "nodejs";

const requestSchema = z.object({
  text: z.string().max(200),
  seed: z.coerce.number().int().default(42),
  seqLen: z.coerce.number().int().min(1).max(env.MAX_SEQ_LEN).default(8),
  dModel: z.coerce.number().int().min(2).max(env.MAX_D_MODEL).default(16),
  dFf: z.coerce.number().int().min(2).max(256).default(32),
  nHeads: z.coerce.number().int().min(1).max(8).default(2),
  nBlocks: z.coerce.number().int().min(1).max(env.MAX_BLOCKS).default(2),
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
  const { text, seed, seqLen, dModel, dFf, nHeads, nBlocks } = parsed.data;
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
    n_blocks: nBlocks,
    vocab_size: VOCAB_SIZE,
    seed,
  };
  const w = initModelWeights(config);

  const tokenIds = encode(text, seqLen);
  const trace = {
    tokenIds: [] as number[],
    tokEmb: [] as number[][],
    posEmb: [] as number[][],
    blocks: Array.from({ length: nBlocks }, () => emptyBlockTrace()),
    xFinal: [] as number[][],
    logits: [] as number[][],
  };
  // Reset block traces — emptyBlockTrace shares an empty matrix between
  // sibling fields, but forwardTyped overwrites those fields anyway.
  trace.blocks.forEach((bt) => {
    bt.attn = emptyAttentionTrace();
  });

  const out = forwardTyped(tokenIds, config, w, trace);

  // Pull just the per-head softmax-weights matrices for the stacking widget;
  // the full block trace is also returned for callers that want it.
  const perBlockWeights = trace.blocks.map((bt) => bt.attn.weights);

  return NextResponse.json({
    ok: true,
    data: {
      alphabet: ALPHABET,
      tokenIds,
      tokens: tokenIds.map((id) => ALPHABET[id] ?? "?"),
      tokEmb: out.tokEmb,
      posEmb: out.posEmb,
      xFinal: out.xFinal,
      logits: out.logits,
      perBlockAttention: perBlockWeights, // [nBlocks, nHeads, S, S]
      config,
    },
  });
}
