/**
 * PATCH /api/comments/[id]
 *
 * Owners may edit `body`. Admins may additionally toggle `hidden`.
 * A new body is rendered before the update, so a rendering failure leaves
 * the stored comment unchanged.
 */
import { NextResponse } from "next/server";

import { auth, isAdmin } from "@/lib/auth";
import { renderCommentHtml, update, updateCommentSchema } from "@/lib/comments";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function PATCH(
  req: Request,
  ctx: { params: { id: string } },
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
  const parsed = updateCommentSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      { ok: false, error: parsed.error.issues[0]?.message ?? "Invalid input" },
      { status: 400 },
    );
  }
  // Render the new body (if any) before touching the database.
  let newBodyHtml: string | null = null;
  try {
    if (parsed.data.body !== undefined) {
      newBodyHtml = renderCommentHtml(parsed.data.body);
    }
  } catch (err) {
    console.error("[comments] render failed (edit)", err);
    return NextResponse.json(
      { ok: false, error: "Couldn't render your edit, so it wasn't saved." },
      { status: 500 },
    );
  }

  const login = (session?.user as { githubLogin?: string | null } | undefined)
    ?.githubLogin;
  const updated = await update(
    userId,
    isAdmin(login),
    ctx.params.id,
    parsed.data,
  );
  if (!updated) {
    return NextResponse.json(
      {
        ok: false,
        error: "Not found, not the owner, or non-admin tried to hide.",
      },
      { status: 404 },
    );
  }
  let bodyHtml = "";
  if (!updated.hidden) {
    try {
      bodyHtml = newBodyHtml ?? renderCommentHtml(updated.bodyMd);
    } catch (err) {
      // Only reachable when an admin un-hides a comment whose stored body
      // no longer renders; the change is saved, so say so.
      console.error("[comments] render failed (unhide)", err);
      return NextResponse.json(
        { ok: false, error: "Saved, but the comment couldn't be rendered." },
        { status: 500 },
      );
    }
  }
  return NextResponse.json({
    ok: true,
    data: { id: updated.id, hidden: updated.hidden, bodyHtml },
  });
}
