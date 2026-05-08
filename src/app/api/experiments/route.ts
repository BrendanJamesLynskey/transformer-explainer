/**
 * /api/experiments
 *
 *   GET  → public list (paginated)
 *   POST → create (auth required)
 */
import { NextResponse } from "next/server";
import { z } from "zod";

import { auth } from "@/lib/auth";
import { create, createSchema, listPublic } from "@/lib/experiments";

export const runtime = "nodejs";

const listSchema = z.object({
  limit: z.coerce.number().int().min(1).max(50).default(20),
  offset: z.coerce.number().int().min(0).default(0),
});

export async function GET(req: Request): Promise<Response> {
  const url = new URL(req.url);
  const parsed = listSchema.safeParse({
    limit: url.searchParams.get("limit") ?? undefined,
    offset: url.searchParams.get("offset") ?? undefined,
  });
  if (!parsed.success) {
    return NextResponse.json(
      { ok: false, error: parsed.error.issues[0]?.message ?? "Invalid query" },
      { status: 400 },
    );
  }
  const items = await listPublic(parsed.data.limit, parsed.data.offset);
  return NextResponse.json({ ok: true, data: items });
}

export async function POST(req: Request): Promise<Response> {
  const session = await auth();
  if (!session?.user) {
    return NextResponse.json(
      { ok: false, error: "Sign in to save experiments." },
      { status: 401 },
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
  const parsed = createSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      { ok: false, error: parsed.error.issues[0]?.message ?? "Invalid input" },
      { status: 400 },
    );
  }
  const userId = (session.user as { id?: string }).id;
  if (!userId) {
    return NextResponse.json(
      { ok: false, error: "Session is missing user id." },
      { status: 401 },
    );
  }
  const exp = await create(userId, parsed.data);
  return NextResponse.json({ ok: true, data: exp }, { status: 201 });
}
