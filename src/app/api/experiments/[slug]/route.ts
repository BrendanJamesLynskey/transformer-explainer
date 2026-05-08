/**
 * /api/experiments/[slug]
 *
 *   GET    → read (public/unlisted: anyone; private: owner only)
 *   PATCH  → update (auth + owner)
 *   DELETE → remove (auth + owner)
 */
import { NextResponse } from "next/server";

import { auth } from "@/lib/auth";
import { getBySlug, remove, update, updateSchema } from "@/lib/experiments";

export const runtime = "nodejs";

type RouteContext = { params: { slug: string } };

export async function GET(_req: Request, ctx: RouteContext): Promise<Response> {
  const exp = await getBySlug(ctx.params.slug);
  if (!exp) {
    return NextResponse.json(
      { ok: false, error: "Not found" },
      { status: 404 },
    );
  }
  if (exp.visibility === "private") {
    const session = await auth();
    const userId = (session?.user as { id?: string } | undefined)?.id;
    if (userId !== exp.ownerId) {
      return NextResponse.json(
        { ok: false, error: "Not found" },
        { status: 404 },
      );
    }
  }
  return NextResponse.json({ ok: true, data: exp });
}

export async function PATCH(
  req: Request,
  ctx: RouteContext,
): Promise<Response> {
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
  const parsed = updateSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      { ok: false, error: parsed.error.issues[0]?.message ?? "Invalid input" },
      { status: 400 },
    );
  }
  const updated = await update(userId, ctx.params.slug, parsed.data);
  if (!updated) {
    return NextResponse.json(
      { ok: false, error: "Not found or not the owner." },
      { status: 404 },
    );
  }
  return NextResponse.json({ ok: true, data: updated });
}

export async function DELETE(
  _req: Request,
  ctx: RouteContext,
): Promise<Response> {
  const session = await auth();
  const userId = (session?.user as { id?: string } | undefined)?.id;
  if (!userId) {
    return NextResponse.json({ ok: false, error: "Sign in." }, { status: 401 });
  }
  const ok = await remove(userId, ctx.params.slug);
  if (!ok) {
    return NextResponse.json(
      { ok: false, error: "Not found or not the owner." },
      { status: 404 },
    );
  }
  return NextResponse.json({ ok: true, data: { slug: ctx.params.slug } });
}
