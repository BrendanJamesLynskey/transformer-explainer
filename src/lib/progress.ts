/**
 * Per-user, per-section progress data layer.
 */
import { and, eq, sql } from "drizzle-orm";
import { z } from "zod";

import { db } from "@/lib/db/client";
import { progress, type Progress } from "@/lib/db/schema";

export const STATUSES = ["not_started", "in_progress", "completed"] as const;
export type ProgressStatus = (typeof STATUSES)[number];

export const upsertProgressSchema = z.object({
  sectionSlug: z.string().min(1).max(80),
  status: z.enum(STATUSES),
});

export async function listForUser(userId: string): Promise<Progress[]> {
  return db.select().from(progress).where(eq(progress.userId, userId));
}

export async function getOne(
  userId: string,
  sectionSlug: string,
): Promise<Progress | null> {
  const rows = await db
    .select()
    .from(progress)
    .where(
      and(eq(progress.userId, userId), eq(progress.sectionSlug, sectionSlug)),
    )
    .limit(1);
  return rows[0] ?? null;
}

const STATUS_RANK: Record<ProgressStatus, number> = {
  not_started: 0,
  in_progress: 1,
  completed: 2,
};

/**
 * Upsert: never *regress* the status (so an idle ProgressTracker doesn't
 * downgrade a completed section back to in_progress on revisit). The
 * `monotonic` SQL clause uses GREATEST against the rank lookup.
 */
export async function upsert(
  userId: string,
  sectionSlug: string,
  status: ProgressStatus,
): Promise<Progress> {
  const newRank = STATUS_RANK[status];

  await db
    .insert(progress)
    .values({ userId, sectionSlug, status })
    .onConflictDoUpdate({
      target: [progress.userId, progress.sectionSlug],
      set: {
        status: sql`CASE
          WHEN ${progress.status} = 'completed' THEN ${progress.status}
          WHEN ${progress.status} = 'in_progress' AND ${newRank} >= 1 THEN ${status}
          WHEN ${progress.status} = 'not_started' THEN ${status}
          ELSE ${progress.status}
        END`,
        updatedAt: new Date(),
      },
    });
  return (await getOne(userId, sectionSlug))!;
}
