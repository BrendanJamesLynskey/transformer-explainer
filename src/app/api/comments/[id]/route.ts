/**
 * PATCH /api/comments/[id]
 *
 * Owners may edit `body`. Admins may additionally toggle `hidden`.
 */
import { NextResponse } from "next/server";

import { auth, isAdmin } from "@/lib/auth";
import { renderCommentHtml, update, updateCommentSchema } from "@/lib/comments";

export const runtime = "nodejs";

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
  return NextResponse.json({
    ok: true,
    data: {
      id: updated.id,
      hidden: updated.hidden,
      bodyHtml: updated.hidden ? "" : renderCommentHtml(updated.bodyMd),
    },
  });
}
