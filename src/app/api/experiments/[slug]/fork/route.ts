/**
 * POST /api/experiments/[slug]/fork
 *
 * Anyone signed in can fork a public or unlisted experiment to their
 * own account. Forking a private experiment returns 404 (the source's
 * visibility check matches `getBySlug` semantics — the forker has to be
 * able to *see* the source).
 */
import { NextResponse } from "next/server";

import { auth } from "@/lib/auth";
import { fork, getBySlug } from "@/lib/experiments";

export const runtime = "nodejs";

export async function POST(
  _req: Request,
  ctx: { params: { slug: string } },
): Promise<Response> {
  const session = await auth();
  const userId = (session?.user as { id?: string } | undefined)?.id;
  if (!userId) {
    return NextResponse.json({ ok: false, error: "Sign in." }, { status: 401 });
  }

  const source = await getBySlug(ctx.params.slug);
  if (!source || source.visibility === "private") {
    return NextResponse.json(
      { ok: false, error: "Cannot fork that experiment." },
      { status: 404 },
    );
  }

  const forked = await fork(userId, source);
  return NextResponse.json({ ok: true, data: forked }, { status: 201 });
}
