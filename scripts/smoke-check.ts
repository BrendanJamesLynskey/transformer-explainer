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
 * It also fetches one section's comment list, which must hold at least one
 * visible comment, so the server-side Markdown sanitiser actually runs. A
 * list with no visible comments never calls it, and that is how a comment
 * renderer that couldn't load on Vercel (ERR_REQUIRE_ESM, 2026-10-04)
 * passed every check until someone posted. The section defaults to
 * `07-sampling`; choose another, or skip the check, with
 *
 *     pnpm smoke <url> --comments-section 01-overview
 *     pnpm smoke <url> --comments-section none
 *
 * Fails when:
 *   - `/api/health` is not 200 with `healthy: true` (DB down, or the schema
 *     is behind — the deploy went out before `pnpm db:migrate`);
 *   - any page is not a 200 (redirects count as failures, so a page that
 *     bounces to sign-in is caught too);
 *   - the comment list is not a 200 `{ ok: true }`, or has no visible
 *     comment with rendered HTML.
 *
 * Needs no secrets and touches no database directly.
 */
import { SECTIONS } from "@/lib/mdx/sections";

/** Section whose comment list is checked (it must have a visible comment). */
const DEFAULT_COMMENTS_SECTION = "07-sampling";

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

type CommentsBody = {
  ok?: boolean;
  data?: { hidden?: boolean; bodyHtml?: string }[];
};

async function checkComments(base: string, slug: string): Promise<Result> {
  const path = `/api/sections/${slug}/comments`;
  try {
    const res = await fetch(base + path, { redirect: "manual" });
    const text = await res.text();
    let body: CommentsBody | null;
    try {
      body = JSON.parse(text) as CommentsBody | null;
    } catch {
      return { path, ok: false, detail: `${res.status} (non-JSON body)` };
    }
    if (res.status !== 200 || body?.ok !== true || !Array.isArray(body.data)) {
      return { path, ok: false, detail: `${res.status} ok=${body?.ok}` };
    }
    const rendered = body.data.filter(
      (c) => !c.hidden && typeof c.bodyHtml === "string" && c.bodyHtml !== "",
    ).length;
    return {
      path,
      ok: rendered > 0,
      detail:
        rendered > 0
          ? `200 ${rendered}/${body.data.length} rendered`
          : `200 but no visible comment to render; post one or pass --comments-section`,
    };
  } catch (err) {
    return { path, ok: false, detail: (err as Error).message };
  }
}

/** Parse `[url] [--comments-section <slug|none>]`. */
function parseArgs(argv: string[]): { base: string; commentsSection: string } {
  let base = "http://localhost:3000";
  let commentsSection = DEFAULT_COMMENTS_SECTION;
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]!;
    if (arg === "--comments-section") {
      commentsSection = argv[++i] ?? "";
    } else if (arg.startsWith("--comments-section=")) {
      commentsSection = arg.slice("--comments-section=".length);
    } else {
      base = arg;
    }
  }
  if (
    commentsSection !== "none" &&
    !SECTIONS.some((s) => s.slug === commentsSection)
  ) {
    console.error(`Unknown section for --comments-section: ${commentsSection}`);
    process.exit(2);
  }
  return { base: base.replace(/\/$/, ""), commentsSection };
}

async function main(): Promise<void> {
  const { base, commentsSection } = parseArgs(process.argv.slice(2));
  console.log(`Smoke-checking ${base}`);

  const results = [await checkHealth(base)];
  for (const path of PAGES) results.push(await checkPage(base, path));
  if (commentsSection !== "none") {
    results.push(await checkComments(base, commentsSection));
  }

  for (const r of results) {
    console.log(
      `  ${r.ok ? "ok  " : "FAIL"}  ${r.path.padEnd(36)} ${r.detail}`,
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
