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
 */
const lastWarnAt = new Map<string, number>();
const WARN_INTERVAL_MS = 60_000;

function maybeWarn(key: string, err: unknown) {
  const now = Date.now();
  const prev = lastWarnAt.get(key) ?? 0;
  if (now - prev < WARN_INTERVAL_MS) return;
  lastWarnAt.set(key, now);
  // eslint-disable-next-line no-console
  console.warn(
    `[db-fallback] ${key}: returning fallback value. Underlying error:`,
    err instanceof Error ? err.message : err,
  );
}

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
