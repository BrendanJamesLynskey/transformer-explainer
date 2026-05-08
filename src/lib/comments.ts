/**
 * Comments data layer.
 *
 * Two-level threading: top-level comments have `parent_id = null`; replies
 * have `parent_id = <top-level id>`. Replies-of-replies are not supported
 * — the API rejects them.
 */
import { and, eq, isNull, asc } from "drizzle-orm";
import { z } from "zod";

import { db } from "@/lib/db/client";
import { comments, type Comment, type NewComment } from "@/lib/db/schema";

export { renderCommentHtml } from "./comments-render";

export const createCommentSchema = z.object({
  body: z.string().min(1).max(2000),
  parentId: z.string().uuid().optional(),
});

export const updateCommentSchema = z.object({
  body: z.string().min(1).max(2000).optional(),
  hidden: z.boolean().optional(),
});

export type CreateComment = z.infer<typeof createCommentSchema>;
export type UpdateComment = z.infer<typeof updateCommentSchema>;

export async function listForSection(slug: string): Promise<Comment[]> {
  return db
    .select()
    .from(comments)
    .where(eq(comments.sectionSlug, slug))
    .orderBy(asc(comments.createdAt));
}

export async function listTopLevel(slug: string): Promise<Comment[]> {
  return db
    .select()
    .from(comments)
    .where(and(eq(comments.sectionSlug, slug), isNull(comments.parentId)))
    .orderBy(asc(comments.createdAt));
}

export async function create(
  userId: string,
  sectionSlug: string,
  input: CreateComment,
): Promise<Comment> {
  const row: NewComment = {
    sectionSlug,
    userId,
    parentId: input.parentId ?? null,
    bodyMd: input.body,
  };
  const inserted = await db.insert(comments).values(row).returning();
  return inserted[0]!;
}

export async function update(
  userId: string,
  isAdminUser: boolean,
  id: string,
  patch: UpdateComment,
): Promise<Comment | null> {
  // Owners may edit their body; admins additionally may hide.
  const next: Partial<NewComment> = {};
  if (patch.body !== undefined) next.bodyMd = patch.body;
  if (patch.hidden !== undefined) {
    if (!isAdminUser) return null; // hiding is admin-only
    next.hidden = patch.hidden;
  }
  if (Object.keys(next).length === 0) return null;

  const where = isAdminUser
    ? eq(comments.id, id)
    : and(eq(comments.id, id), eq(comments.userId, userId));
  const rows = await db.update(comments).set(next).where(where).returning();
  return rows[0] ?? null;
}

export async function isReplyOfReply(
  parentId: string | undefined,
): Promise<boolean> {
  if (!parentId) return false;
  const parent = await db
    .select({ parentId: comments.parentId })
    .from(comments)
    .where(eq(comments.id, parentId))
    .limit(1);
  return parent[0]?.parentId !== null && parent[0] !== undefined;
}
