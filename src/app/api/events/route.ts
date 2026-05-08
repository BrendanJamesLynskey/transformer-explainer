/**
 * POST /api/events
 *
 * Accepts a batch of analytics events from the client. Auth is optional —
 * anon visits are explicitly in scope (DAU is the most useful signal we
 * have). The session id in the body is what ties anon events together.
 */
import { NextResponse } from "next/server";

import { auth } from "@/lib/auth";
import { ingestSchema, rateLimitOk, recordEvents } from "@/lib/analytics";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: Request): Promise<Response> {
  // 60 events / minute / IP. Generous — a single page can emit ~10.
  const ip =
    req.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ??
    req.headers.get("x-real-ip") ??
    "anon";
  if (!rateLimitOk(`events:${ip}`, 60, 1)) {
    return NextResponse.json(
      { ok: false, error: "Slow down." },
      { status: 429 },
    );
  }

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json(
      { ok: false, error: "Invalid JSON body" },
      { status: 400 },
    );
  }
  const parsed = ingestSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      { ok: false, error: parsed.error.issues[0]?.message ?? "Invalid input" },
      { status: 400 },
    );
  }

  const session = await auth();
  const userId = (session?.user as { id?: string } | undefined)?.id ?? null;

  const written = await recordEvents(userId, parsed.data);
  return NextResponse.json({ ok: true, data: { written } });
}
