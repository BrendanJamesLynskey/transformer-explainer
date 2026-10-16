/**
 * POST /api/compute/embed
 *
 * Body: { text: string; seed?: number; seqLen?: number; dModel?: number;
 *        weights?: "random" | "trained" }
 *
 * Tokenises the input, looks up token embeddings from a deterministic
 * seed-based init, builds the sinusoidal positional encoding, and returns
 * everything as a trace the client visualisations can render.
 */
import { NextResponse } from "next/server";
import { z } from "zod";

import {
  DEFAULT_PARAMS,
  computeEmbed,
  trainedFits,
} from "@/lib/compute/traces";
import { env } from "@/lib/env";

export const runtime = "nodejs";

const requestSchema = z.object({
  text: z.string().max(200),
  // Defaults match SPEC §4 — kept aligned with the API surface in later phases.
  seed: z.coerce.number().int().default(42),
  seqLen: z.coerce.number().int().min(1).max(env.MAX_SEQ_LEN).default(8),
  dModel: z.coerce.number().int().min(2).max(env.MAX_D_MODEL).default(16),
  // "trained" runs Training's trained weights (they need the trained shapes)
  weights: z.enum(["random", "trained"]).default("random"),
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

  if (params.weights === "trained" && !trainedFits(params)) {
    return NextResponse.json(
      {
        ok: false,
        error:
          'weights "trained" needs the trained model\'s shapes: dModel 16, nHeads 2, dFf 32, nBlocks 2',
      },
      { status: 400 },
    );
  }
  return NextResponse.json({ ok: true, data: computeEmbed(params) });
}
