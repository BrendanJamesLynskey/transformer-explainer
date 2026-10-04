/**
 * Pure health-check logic: compare the migrations a database has applied
 * with the migrations this build ships, and shape the `/api/health` reply.
 *
 * Split out of `health.ts` (which talks to Postgres) so unit tests can
 * import it without booting the Drizzle client — the same pattern as
 * `analytics-shared.ts`.
 *
 * Why this exists: `lib/db-fallback.ts` deliberately hides database errors
 * so a fresh clone works without a database. In production that same
 * kindness hid a database that had never been migrated, for months. The
 * health check is the one place that refuses to fall back.
 */

/** One entry of `drizzle/meta/_journal.json`, as drizzle-kit writes it. */
export type JournalEntry = {
  idx: number;
  /** Milliseconds since the epoch; drizzle-kit stores it as `created_at`. */
  when: number;
  tag: string;
};

/**
 * Where the database's schema stands relative to this build.
 *
 * - `current` — the newest applied migration is the newest one shipped.
 * - `behind`  — this build ships migrations the database hasn't run.
 *               Queries against new columns or tables will fail.
 * - `ahead`   — the database has a migration this build doesn't know
 *               about (an older deploy after a newer migration). Usually
 *               harmless for additive migrations, so it is reported but
 *               not treated as a failure.
 * - `none`    — no migrations table at all: the database was never migrated.
 */
export type SchemaStatus = "current" | "behind" | "ahead" | "none";

export type SchemaReport = {
  status: SchemaStatus;
  /** Number of rows in `drizzle.__drizzle_migrations`. */
  applied: number;
  /** Number of migrations in this build's journal. */
  expected: number;
  /** Tag of the newest migration this build ships, e.g. `0000_superb_demogoblin`. */
  latestExpected: string | null;
  /** Tag matching the newest applied migration, or null if unknown to this build. */
  latestApplied: string | null;
};

export type HealthReport = {
  /** True when the database answers and its schema is not behind. */
  healthy: boolean;
  db: "up" | "down";
  /** Postgres / socket error code when `db` is `down` (never a message). */
  dbError: string | null;
  /** Null when the database could not be reached. */
  schema: SchemaReport | null;
  checkedAt: string;
};

/**
 * Compare applied migrations with the build's journal.
 *
 * drizzle-kit records each migration's journal `when` as the row's
 * `created_at`, so the newest timestamp on each side identifies the newest
 * migration. Comparing timestamps rather than counts means a database that
 * skipped a migration in the middle still reads as `behind`.
 *
 * @param appliedCreatedAt  `created_at` of every applied migration, or `null`
 *                          when the migrations table doesn't exist.
 * @param journal           The `entries` array from `_journal.json`.
 */
export function compareSchema(
  appliedCreatedAt: readonly number[] | null,
  journal: readonly JournalEntry[],
): SchemaReport {
  const sorted = [...journal].sort((a, b) => a.when - b.when);
  const newestShipped = sorted[sorted.length - 1] ?? null;
  const expected = sorted.length;

  if (appliedCreatedAt === null || appliedCreatedAt.length === 0) {
    return {
      status: expected === 0 ? "current" : "none",
      applied: 0,
      expected,
      latestExpected: newestShipped?.tag ?? null,
      latestApplied: null,
    };
  }

  const newestApplied = Math.max(...appliedCreatedAt);
  const appliedTag = sorted.find((e) => e.when === newestApplied)?.tag ?? null;
  const newestShippedWhen = newestShipped?.when ?? 0;

  let status: SchemaStatus = "current";
  if (newestApplied < newestShippedWhen) status = "behind";
  else if (newestApplied > newestShippedWhen) status = "ahead";

  return {
    status,
    applied: appliedCreatedAt.length,
    expected,
    latestExpected: newestShipped?.tag ?? null,
    latestApplied: appliedTag,
  };
}

/**
 * Reduce a thrown database error to a short code that is safe to publish.
 *
 * postgres-js errors carry a SQLSTATE (`28P01` = bad password, `3D000` =
 * no such database) and socket errors a Node code (`ECONNREFUSED`,
 * `ENOTFOUND`). Messages can contain user names or host names, so they are
 * never returned.
 */
export function safeErrorCode(err: unknown): string {
  if (err instanceof Error && err.message === "timeout") return "TIMEOUT";
  const code =
    typeof err === "object" && err !== null && "code" in err
      ? (err as { code: unknown }).code
      : undefined;
  if (typeof code === "string" && /^[A-Z0-9_]{2,24}$/.test(code)) return code;
  return "UNKNOWN";
}

/**
 * Build the final report. Healthy means "the database answers and every
 * migration this build needs has run".
 */
export function buildHealthReport(
  result:
    | { reachable: true; schema: SchemaReport }
    | { reachable: false; error: unknown },
  now: Date = new Date(),
): HealthReport {
  if (!result.reachable) {
    return {
      healthy: false,
      db: "down",
      dbError: safeErrorCode(result.error),
      schema: null,
      checkedAt: now.toISOString(),
    };
  }
  const healthy =
    result.schema.status === "current" || result.schema.status === "ahead";
  return {
    healthy,
    db: "up",
    dbError: null,
    schema: result.schema,
    checkedAt: now.toISOString(),
  };
}
