/**
 * POST /api/compute/sample
 *
 * Body: { logits: number[]; mode: "greedy" | "temperature" | "top-k" | "top-p";
 *         temperature?: number; k?: number; p?: number; seed?: number }
 *
 * Returns a single sampled token id under the requested mode. The optional
 * `seed` makes the sampling reproducible (mulberry32 RNG).
 */
import { NextResponse } from "next/server";
import { z } from "zod";

import { mulberry32 } from "@/lib/transformer/random";
import { sample, type SampleMode } from "@/lib/transformer/sampling";
import { ALPHABET } from "@/lib/transformer/tokenizer";

export const runtime = "nodejs";

const requestSchema = z.object({
  logits: z.array(z.number()).min(1).max(512),
  mode: z.enum(["greedy", "temperature", "top-k", "top-p"]),
  temperature: z.number().positive().max(10).default(1),
  k: z.number().int().positive().max(64).default(8),
  p: z.number().min(0).max(1).default(0.9),
  seed: z.number().int().optional(),
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
  const { logits, mode, temperature, k, p, seed } = parsed.data;

  let cfg: SampleMode;
  switch (mode) {
    case "greedy":
      cfg = { kind: "greedy" };
      break;
    case "temperature":
      cfg = { kind: "temperature", temperature };
      break;
    case "top-k":
      cfg = { kind: "top-k", k, temperature };
      break;
    case "top-p":
      cfg = { kind: "top-p", p, temperature };
      break;
  }

  const rng = seed === undefined ? Math.random : mulberry32(seed);
  const id = sample(logits, cfg, rng);
  return NextResponse.json({
    ok: true,
    data: { id, char: ALPHABET[id] ?? "?" },
  });
}
