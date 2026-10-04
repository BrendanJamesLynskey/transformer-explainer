/**
 * GET /api/health
 *
 * Reports whether the database answers and whether its schema has every
 * migration this build ships. Public and unauthenticated, so it returns
 * only statuses, counts, migration tags and an error *code* — never a
 * connection string, host name or error message.
 *
 *   200 → `{ ok: true,  data: HealthReport }`   healthy
 *   503 → `{ ok: false, error, data: HealthReport }`  DB down or schema behind
 *
 * `scripts/smoke-check.ts` calls this after every deploy (RUNBOOK.md).
 */
import { NextResponse } from "next/server";

import { checkHealth } from "@/lib/health";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(): Promise<Response> {
  const report = await checkHealth();
  const headers = { "cache-control": "no-store" };
  if (report.healthy) {
    return NextResponse.json({ ok: true, data: report }, { headers });
  }
  const error =
    report.db === "down"
      ? "Database unreachable."
      : report.schema?.status === "none"
        ? "Database has never been migrated: run `pnpm db:migrate` against it."
        : "Database schema is behind this build: run `pnpm db:migrate` against it.";
  return NextResponse.json(
    { ok: false, error, data: report },
    { status: 503, headers },
  );
}
