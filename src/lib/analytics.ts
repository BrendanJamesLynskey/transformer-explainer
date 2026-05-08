/**
 * Analytics — DB-bound event ingestion and admin-dashboard aggregates.
 *
 * The pure pieces (schema, kinds, in-memory rate limit) live in
 * `analytics-shared.ts` and are re-exported here so callers only ever
 * reach for one module.
 */
import { and, count, desc, eq, gte, sql } from "drizzle-orm";

import { db } from "@/lib/db/client";
import { comments, events, experiments, users } from "@/lib/db/schema";

import type { Ingest } from "./analytics-shared";

export {
  EVENT_KIND,
  _resetRateLimitForTest,
  ingestSchema,
  rateLimitOk,
} from "./analytics-shared";
export type { EventKind, Ingest } from "./analytics-shared";

export async function recordEvents(
  userId: string | null,
  ingest: Ingest,
): Promise<number> {
  const rows = ingest.events.map((e) => ({
    userId,
    sessionId: ingest.sessionId,
    kind: e.kind,
    sectionSlug: e.sectionSlug ?? null,
    metaJson: e.meta ?? null,
  }));
  await db.insert(events).values(rows);
  return rows.length;
}

// ---------------------------------------------------------------------------
// Aggregates for the admin dashboard
// ---------------------------------------------------------------------------

export type DailyActive = { day: string; users: number };

/**
 * Distinct active sessions per UTC day for the last `days` days.
 *
 * We count *sessions* rather than userId because most events are anon. A
 * signed-in user with one tab counts as one session — same as anon.
 */
export async function dailyActive(days: number): Promise<DailyActive[]> {
  const since = new Date(Date.now() - days * 24 * 3600 * 1000);
  const rows = await db
    .select({
      day: sql<string>`to_char(date_trunc('day', ${events.createdAt}) at time zone 'UTC', 'YYYY-MM-DD')`,
      users: sql<number>`count(distinct ${events.sessionId})::int`,
    })
    .from(events)
    .where(gte(events.createdAt, since))
    .groupBy(sql`date_trunc('day', ${events.createdAt}) at time zone 'UTC'`)
    .orderBy(sql`date_trunc('day', ${events.createdAt}) at time zone 'UTC'`);
  return rows;
}

export type SectionFunnelRow = {
  sectionSlug: string;
  views: number;
  interacts: number;
  completes: number;
};

/**
 * Per-section funnel from raw events. `completes` comes from
 * `kind = 'section_complete'` so it includes anon visitors too (the
 * `progress` table is signed-in only).
 */
export async function sectionFunnel(): Promise<SectionFunnelRow[]> {
  const rows = await db
    .select({
      sectionSlug: events.sectionSlug,
      views: sql<number>`count(*) filter (where ${events.kind} = 'page_view')::int`,
      interacts: sql<number>`count(*) filter (where ${events.kind} = 'widget_interact')::int`,
      completes: sql<number>`count(*) filter (where ${events.kind} = 'section_complete')::int`,
    })
    .from(events)
    .where(sql`${events.sectionSlug} is not null`)
    .groupBy(events.sectionSlug)
    .orderBy(events.sectionSlug);
  return rows
    .filter((r): r is SectionFunnelRow => r.sectionSlug !== null)
    .map((r) => ({ ...r, sectionSlug: r.sectionSlug }));
}

export type TopExperiment = {
  slug: string;
  name: string;
  views: number;
};

export async function topExperiments(limit = 10): Promise<TopExperiment[]> {
  const rows = await db
    .select({
      slug: experiments.slug,
      name: experiments.name,
      views: count(events.id),
    })
    .from(experiments)
    .leftJoin(
      events,
      and(
        eq(events.kind, "exp_view"),
        sql`${events.metaJson}->>'slug' = ${experiments.slug}`,
      ),
    )
    .groupBy(experiments.slug, experiments.name)
    .orderBy(desc(count(events.id)))
    .limit(limit);
  return rows.map((r) => ({ ...r, views: Number(r.views) }));
}

export type RecentComment = {
  id: string;
  body: string;
  createdAt: Date;
  sectionSlug: string;
  authorName: string | null;
  authorLogin: string | null;
};

export async function recentComments(limit = 10): Promise<RecentComment[]> {
  const rows = await db
    .select({
      id: comments.id,
      body: comments.bodyMd,
      createdAt: comments.createdAt,
      sectionSlug: comments.sectionSlug,
      authorName: users.name,
      authorLogin: users.githubLogin,
    })
    .from(comments)
    .leftJoin(users, eq(users.id, comments.userId))
    .orderBy(desc(comments.createdAt))
    .limit(limit);
  return rows;
}
