/**
 * Database health check — the DB-bound half of `health-shared.ts`.
 *
 * Asks Postgres for the migrations it has applied and compares them with
 * `drizzle/meta/_journal.json`, the list of migrations this build ships.
 * Unlike every other DB read in the app, this one does **not** go through
 * `runOrFallback`: its whole job is to notice when the database is missing.
 */
import { sql } from "drizzle-orm";

import journal from "../../drizzle/meta/_journal.json";

import { db } from "@/lib/db/client";

import {
  buildHealthReport,
  compareSchema,
  type HealthReport,
  type JournalEntry,
} from "./health-shared";

/** Give up on the database after this long; a health check must answer. */
const DB_TIMEOUT_MS = 5_000;

/** Postgres SQLSTATE for "relation does not exist". */
const UNDEFINED_TABLE = "42P01";

/**
 * Read `created_at` for every applied migration. Returns `null` when the
 * migrations table doesn't exist (the database was never migrated).
 */
async function readAppliedMigrations(): Promise<number[] | null> {
  try {
    const rows = await db.execute<{ created_at: string | number }>(
      sql`select created_at from drizzle.__drizzle_migrations`,
    );
    return Array.from(rows, (r) => Number(r.created_at));
  } catch (err) {
    const code = (err as { code?: unknown }).code;
    if (code === UNDEFINED_TABLE || code === "3F000") return null; // no table / no schema
    throw err;
  }
}

function withTimeout<T>(p: Promise<T>, ms: number): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("timeout")), ms);
    p.then(
      (v) => {
        clearTimeout(timer);
        resolve(v);
      },
      (e: unknown) => {
        clearTimeout(timer);
        reject(e);
      },
    );
  });
}

/**
 * Run the health check against the live database. Never throws: an
 * unreachable database is a result (`db: "down"`), not an exception.
 */
export async function checkHealth(): Promise<HealthReport> {
  try {
    const applied = await withTimeout(readAppliedMigrations(), DB_TIMEOUT_MS);
    const schema = compareSchema(
      applied,
      journal.entries as readonly JournalEntry[],
    );
    return buildHealthReport({ reachable: true, schema });
  } catch (error) {
    return buildHealthReport({ reachable: false, error });
  }
}
