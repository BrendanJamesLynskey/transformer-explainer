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

import { DEFAULT_PARAMS, computeBlock } from "@/lib/compute/traces";
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
  const params = { ...DEFAULT_PARAMS, ...parsed.data };
  if (params.dModel % params.nHeads !== 0) {
    return NextResponse.json(
      {
        ok: false,
        error: `dModel (${params.dModel}) must be divisible by nHeads (${params.nHeads})`,
      },
      { status: 400 },
    );
  }

  return NextResponse.json({ ok: true, data: computeBlock(params) });
}
