/**
 * /api/sections/[slug]/comments
 *
 *   GET  → list (anyone). Hidden comments are still returned but with a
 *          stripped body — the section UI renders a "[hidden]" placeholder.
 *   POST → create (auth required). One-level threading enforced.
 */
import { NextResponse } from "next/server";

import { auth } from "@/lib/auth";
import {
  create,
  createCommentSchema,
  isReplyOfReply,
  listForSection,
  renderCommentHtml,
} from "@/lib/comments";
import { runOrFallback } from "@/lib/db-fallback";
import { isValidSlug } from "@/lib/mdx/sections";

export const runtime = "nodejs";
// `isomorphic-dompurify` initialises jsdom at import time. Skip Next's
// build-time route data collection (which evaluates the module in a
// stripped environment that breaks jsdom's stylesheet lookup).
export const dynamic = "force-dynamic";

export async function GET(
  _req: Request,
  ctx: { params: { slug: string } },
): Promise<Response> {
  if (!isValidSlug(ctx.params.slug)) {
    return NextResponse.json(
      { ok: false, error: "Unknown section" },
      { status: 404 },
    );
  }
  // Empty list when the DB isn't reachable — the comments UI handles
  // this gracefully ("Be the first to leave a comment.") and the
  // explainer pages stay usable on a fresh clone.
  const rows = await runOrFallback(
    "comments:list",
    () => listForSection(ctx.params.slug),
    [],
  );
  const data = rows.map((c) => ({
    id: c.id,
    parentId: c.parentId,
    userId: c.userId,
    createdAt: c.createdAt,
    hidden: c.hidden,
    bodyHtml: c.hidden ? "" : renderCommentHtml(c.bodyMd),
  }));
  return NextResponse.json({ ok: true, data });
}

export async function POST(
  req: Request,
  ctx: { params: { slug: string } },
): Promise<Response> {
  const session = await auth();
  const userId = (session?.user as { id?: string } | undefined)?.id;
  if (!userId) {
    return NextResponse.json({ ok: false, error: "Sign in." }, { status: 401 });
  }
  if (!isValidSlug(ctx.params.slug)) {
    return NextResponse.json(
      { ok: false, error: "Unknown section" },
      { status: 404 },
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
  const parsed = createCommentSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      { ok: false, error: parsed.error.issues[0]?.message ?? "Invalid input" },
      { status: 400 },
    );
  }
  if (await isReplyOfReply(parsed.data.parentId)) {
    return NextResponse.json(
      {
        ok: false,
        error: "Threading is one level deep — reply to the top-level comment.",
      },
      { status: 400 },
    );
  }

  const created = await create(userId, ctx.params.slug, parsed.data);
  return NextResponse.json(
    {
      ok: true,
      data: {
        id: created.id,
        parentId: created.parentId,
        userId: created.userId,
        createdAt: created.createdAt,
        hidden: created.hidden,
        bodyHtml: renderCommentHtml(created.bodyMd),
      },
    },
    { status: 201 },
  );
}
