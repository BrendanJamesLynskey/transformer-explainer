/**
 * /api/sections/[slug]/comments
 *
 *   GET  → list (anyone). Hidden comments are still returned but with a
 *          stripped body — the section UI renders a "[hidden]" placeholder.
 *   POST → create (auth required). One-level threading enforced. The body
 *          is rendered *before* it is stored, so a rendering failure can
 *          never leave a saved comment behind a 500 (the client would show
 *          an error, the user would post again, and the section would fill
 *          with duplicates). Each stored comment is also recorded as a
 *          `comment_post` event.
 *
 * Both handlers answer `{ ok: false, error }` JSON on failure, never a bare
 * 500 page (CLAUDE.md §5 → Errors).
 */
import { NextResponse } from "next/server";

import { recordServerEvent } from "@/lib/analytics";
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
// The list changes whenever someone posts: never cache or prerender it.
export const dynamic = "force-dynamic";

const RENDER_FAILED = "Couldn't render the comments. Please try again later.";

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
  try {
    const data = rows.map((c) => ({
      id: c.id,
      parentId: c.parentId,
      userId: c.userId,
      createdAt: c.createdAt,
      hidden: c.hidden,
      bodyHtml: c.hidden ? "" : renderCommentHtml(c.bodyMd),
    }));
    return NextResponse.json({ ok: true, data });
  } catch (err) {
    console.error("[comments] render failed (list)", err);
    return NextResponse.json(
      { ok: false, error: RENDER_FAILED },
      { status: 500 },
    );
  }
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

  // Render first: if this throws, nothing has been stored yet.
  let bodyHtml: string;
  try {
    bodyHtml = renderCommentHtml(parsed.data.body);
  } catch (err) {
    console.error("[comments] render failed (post)", err);
    return NextResponse.json(
      { ok: false, error: "Couldn't render your comment, so it wasn't saved." },
      { status: 500 },
    );
  }

  const created = await create(userId, ctx.params.slug, parsed.data);
  // Into the analytics stream for /admin. Recorded here, not beaconed by the
  // client, so it counts exactly the comments that were actually stored.
  await recordServerEvent(req, userId, "comment_post", ctx.params.slug, {
    commentId: created.id,
    reply: created.parentId !== null,
  });
  return NextResponse.json(
    {
      ok: true,
      data: {
        id: created.id,
        parentId: created.parentId,
        userId: created.userId,
        createdAt: created.createdAt,
        hidden: created.hidden,
        bodyHtml,
      },
    },
    { status: 201 },
  );
}
