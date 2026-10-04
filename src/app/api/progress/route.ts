/**
 * /api/progress
 *
 *   GET  → current user's whole progress map (auth).
 *   POST → upsert one section's status (auth, monotonic), and record a
 *          `progress_update` analytics event.
 */
import { NextResponse } from "next/server";

import { recordServerEvent } from "@/lib/analytics";
import { auth } from "@/lib/auth";
import { listForUser, upsert, upsertProgressSchema } from "@/lib/progress";

export const runtime = "nodejs";

export async function GET(): Promise<Response> {
  const session = await auth();
  const userId = (session?.user as { id?: string } | undefined)?.id;
  if (!userId) {
    return NextResponse.json({ ok: false, error: "Sign in." }, { status: 401 });
  }
  const items = await listForUser(userId);
  return NextResponse.json({ ok: true, data: items });
}

export async function POST(req: Request): Promise<Response> {
  const session = await auth();
  const userId = (session?.user as { id?: string } | undefined)?.id;
  if (!userId) {
    return NextResponse.json({ ok: false, error: "Sign in." }, { status: 401 });
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
  const parsed = upsertProgressSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      { ok: false, error: parsed.error.issues[0]?.message ?? "Invalid input" },
      { status: 400 },
    );
  }
  const row = await upsert(userId, parsed.data.sectionSlug, parsed.data.status);
  // Into the analytics stream for /admin. `stored` can differ from the
  // requested status: the upsert never moves `completed` back.
  await recordServerEvent(
    req,
    userId,
    "progress_update",
    parsed.data.sectionSlug,
    { status: parsed.data.status, stored: row.status },
  );
  return NextResponse.json({ ok: true, data: row });
}
