/**
 * scripts/smoke-check.ts
 *
 * Post-deploy smoke check. Fetches `/api/health` and every public page of a
 * deployed site, and exits non-zero if any of them fails. Run it straight
 * after every production deploy (RUNBOOK.md → "Deploying"):
 *
 *     pnpm smoke https://transformer-decoder-explained.vercel.app
 *
 * With no argument it checks http://localhost:3000.
 *
 * Fails when:
 *   - `/api/health` is not 200 with `healthy: true` (DB down, or the schema
 *     is behind — the deploy went out before `pnpm db:migrate`);
 *   - any page is not a 200 (redirects count as failures, so a page that
 *     bounces to sign-in is caught too).
 *
 * Needs no secrets and touches no database directly.
 */
import { SECTIONS } from "@/lib/mdx/sections";

const PAGES = [
  "/",
  "/learn",
  ...SECTIONS.map((s) => `/learn/${s.slug}`),
  "/playground",
  "/experiments",
  "/about",
  "/signin",
];

type Result = { path: string; ok: boolean; detail: string };

async function checkPage(base: string, path: string): Promise<Result> {
  try {
    const res = await fetch(base + path, { redirect: "manual" });
    return { path, ok: res.status === 200, detail: String(res.status) };
  } catch (err) {
    return { path, ok: false, detail: (err as Error).message };
  }
}

async function checkHealth(base: string): Promise<Result> {
  const path = "/api/health";
  try {
    const res = await fetch(base + path, { redirect: "manual" });
    const body = (await res.json()) as {
      data?: {
        healthy?: boolean;
        db?: string;
        dbError?: string | null;
        schema?: { status?: string; applied?: number; expected?: number };
      };
    };
    const d = body.data;
    const detail =
      `${res.status} db=${d?.db ?? "?"}` +
      (d?.dbError ? ` (${d.dbError})` : "") +
      (d?.schema
        ? ` schema=${d.schema.status} ${d.schema.applied}/${d.schema.expected}`
        : "");
    return { path, ok: res.status === 200 && d?.healthy === true, detail };
  } catch (err) {
    return { path, ok: false, detail: (err as Error).message };
  }
}

async function main(): Promise<void> {
  const base = (process.argv[2] ?? "http://localhost:3000").replace(/\/$/, "");
  console.log(`Smoke-checking ${base}`);

  const results = [await checkHealth(base)];
  for (const path of PAGES) results.push(await checkPage(base, path));

  for (const r of results) {
    console.log(
      `  ${r.ok ? "ok  " : "FAIL"}  ${r.path.padEnd(32)} ${r.detail}`,
    );
  }
  const failed = results.filter((r) => !r.ok);
  if (failed.length > 0) {
    console.error(`\n${failed.length} check(s) failed.`);
    process.exit(1);
  }
  console.log(`\nAll ${results.length} checks passed.`);
}

void main();
