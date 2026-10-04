/**
 * Pure analytics primitives — schema, kinds, rate-limit token bucket.
 *
 * Split out of `analytics.ts` so unit tests (and any code that doesn't
 * actually need to talk to Postgres) can import these without booting the
 * Drizzle client.
 */
import { z } from "zod";

import { SESSION_HEADER } from "./analytics-constants";

export { SESSION_HEADER };

export const EVENT_KIND = [
  "page_view",
  "widget_interact",
  "exp_view",
  "exp_save",
  "exp_fork",
  "comment_post",
  "section_complete",
  // Recorded server-side by POST /api/progress (Phase 11); meta.status is
  // the status the client asked for, meta.stored what the monotonic upsert kept.
  "progress_update",
] as const;
export type EventKind = (typeof EVENT_KIND)[number];

export const ingestSchema = z.object({
  // Stable per-tab id from the client (UUID v4 in localStorage). Required
  // even for signed-in users — gives DAU/WAU a denominator that doesn't
  // require auth.
  sessionId: z.string().min(8).max(64),
  events: z
    .array(
      z.object({
        kind: z.enum(EVENT_KIND),
        sectionSlug: z.string().min(1).max(80).optional(),
        meta: z.record(z.unknown()).optional(),
      }),
    )
    .min(1)
    .max(50),
});
export type Ingest = z.infer<typeof ingestSchema>;

// ---------------------------------------------------------------------------
// Server-recorded events (comments, progress).
// ---------------------------------------------------------------------------

const SESSION_ID_RE = /^[A-Za-z0-9_-]{8,64}$/;

/**
 * The session id to file a server-recorded event under: the client's id
 * from {@link SESSION_HEADER} when it looks like one, otherwise a stable
 * per-user id so the event still counts.
 */
export function serverEventSessionId(
  header: string | null,
  userId: string,
): string {
  if (header && SESSION_ID_RE.test(header)) return header;
  // Real user ids are UUIDs, but pad and cap anyway so the result always
  // satisfies ingestSchema's 8–64 characters.
  return `user:${userId}`.padEnd(8, "_").slice(0, 64);
}

/**
 * Build the ingest batch for one server-recorded event. Goes through
 * {@link ingestSchema} like a client batch, so the two paths can't drift.
 */
export function buildServerEvent(
  sessionId: string,
  kind: EventKind,
  sectionSlug: string,
  meta?: Record<string, unknown>,
): Ingest {
  return ingestSchema.parse({
    sessionId,
    events: [{ kind, sectionSlug, ...(meta ? { meta } : {}) }],
  });
}

// ---------------------------------------------------------------------------
// Tiny in-memory token bucket. Single-instance only — fine for v1 and
// flagged in SPEC.md §12.
// ---------------------------------------------------------------------------

const BUCKETS = new Map<string, { tokens: number; lastRefill: number }>();

export function rateLimitOk(
  key: string,
  capacity: number,
  refillPerSec: number,
  now: number = Date.now(),
): boolean {
  const b = BUCKETS.get(key) ?? { tokens: capacity, lastRefill: now };
  const dt = (now - b.lastRefill) / 1000;
  b.tokens = Math.min(capacity, b.tokens + dt * refillPerSec);
  b.lastRefill = now;
  if (b.tokens < 1) {
    BUCKETS.set(key, b);
    return false;
  }
  b.tokens -= 1;
  BUCKETS.set(key, b);
  return true;
}

/** Test-only: blow away the rate-limit state. */
export function _resetRateLimitForTest(): void {
  BUCKETS.clear();
}
