/**
 * Helpers for routes that *want to work* even if the database isn't reachable.
 *
 * The first-time experience for someone who's just cloned the repo is
 * `pnpm dev` with placeholder values in `.env.local` — they should be able
 * to read /learn and click around without seeing 500-cascades from the
 * EventTracker beacon and the comment-list fetch.
 *
 * `runOrFallback` returns a fallback value if the call throws, and prints
 * **one** warning per minute per `key` so the dev console stays useful.
 *
 * The same kindness is dangerous in production: it once hid a database
 * that had never been migrated. So every fallback is also *counted*, and
 * `/admin` shows the counts next to the `/api/health` result. The counts
 * live in this server instance's memory (a serverless instance starts
 * from zero), which is enough to say "this instance has been serving
 * fallbacks" without adding a table that would itself need the database.
 */
const lastWarnAt = new Map<string, number>();
const WARN_INTERVAL_MS = 60_000;

/** One row of the fallback tally, as `/admin` displays it. */
export type FallbackStat = {
  key: string;
  count: number;
  /** ISO timestamp of the most recent fallback for this key. */
  lastAt: string;
};

const tally = new Map<string, { count: number; lastAt: number }>();

function record(key: string, now: number) {
  const prev = tally.get(key);
  tally.set(key, { count: (prev?.count ?? 0) + 1, lastAt: now });
}

/**
 * Snapshot of how often each `key` has fallen back since this server
 * instance started, most recent first.
 */
export function fallbackStats(): FallbackStat[] {
  return [...tally.entries()]
    .map(([key, v]) => ({
      key,
      count: v.count,
      lastAt: new Date(v.lastAt).toISOString(),
    }))
    .sort((a, b) => b.lastAt.localeCompare(a.lastAt));
}

/** Test-only: forget every recorded fallback. */
export function _resetFallbackStatsForTest(): void {
  tally.clear();
}

function maybeWarn(key: string, err: unknown) {
  const now = Date.now();
  record(key, now);
  const prev = lastWarnAt.get(key) ?? 0;
  if (now - prev < WARN_INTERVAL_MS) return;
  lastWarnAt.set(key, now);
  // eslint-disable-next-line no-console
  console.warn(
    `[db-fallback] ${key}: returning fallback value. Underlying error:`,
    err instanceof Error ? err.message : err,
  );
}

/**
 * Run `fn`; if it throws, log (throttled), count the fallback, and return
 * `fallback` instead.
 */
export async function runOrFallback<T>(
  key: string,
  fn: () => Promise<T>,
  fallback: T,
): Promise<T> {
  try {
    return await fn();
  } catch (err) {
    maybeWarn(key, err);
    return fallback;
  }
}
