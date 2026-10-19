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
 * Part 2 (Training) has its own address (brief 29E, src/lib/site/hosts.ts).
 * Given the production decoder host, the check also covers the training
 * host (or name it with `--training <url>`): every Training page answers 200
 * there, with its canonical URL on the training host and a header naming the
 * Training site; the decoder host's `/training*` 308-redirects to it; the
 * training host's `/` goes to `/training` and its `/learn` back to the
 * decoder host. Against one local server (no `--training`), both parts are
 * checked on that one host, as before.
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
 *     comment with rendered HTML;
 *   - the home page's header lacks the four-track site switch (with
 *     Orchestration linked and Training on the training host), or links
 *     Compute in transit, which the switch leaves out (`unlisted`);
 *   - (two hosts) a redirect, canonical URL or header name is wrong.
 *
 * Needs no secrets and touches no database directly.
 */
import { SECTIONS, TRAINING_SECTIONS } from "@/lib/mdx/sections";
import { DECODER_ORIGIN, TRAINING_ORIGIN } from "@/lib/site/hosts";

/** Section whose comment list is checked (it must have a visible comment). */
const DEFAULT_COMMENTS_SECTION = "07-sampling";

/** Part 1 and the site's other pages: served by the decoder host. */
const DECODER_PAGES = [
  "/",
  "/foundations",
  "/learn",
  ...SECTIONS.map((s) => `/learn/${s.slug}`),
  "/playground",
  "/experiments",
  "/about",
  "/signin",
];

/** Part 2's pages: served by the training host (or the one local host). */
const TRAINING_PAGES = [
  "/training",
  // all 16 Part 2 chapters (29A: 1-5, 29B: 6-10, 29C: 11-16)
  ...TRAINING_SECTIONS.map((s) => `/training/${s.slug}`),
  "/training/corpus.txt",
  "/training/default-run-weights.json",
  "/training/attention-snapshots.json",
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

/** Markers the site switch must (and must not) put in the home page's HTML. */
const SWITCH_PRESENT = [
  'data-site-switch="menu"',
  'data-site-group="foundations"',
  'data-site-group="hardware"',
  'href="https://ai-racks-explained.vercel.app"',
  'href="https://ai-accelerators-explained.vercel.app"',
  'href="https://agent-orchestration-explained.vercel.app"',
  'href="https://agent-evals-explained.vercel.app"',
  'href="https://agent-security-explained.vercel.app"',
  `href="${TRAINING_ORIGIN}"`,
];
const SWITCH_ABSENT = ["compute-in-transit-results.vercel.app"];

async function checkSwitch(base: string, page = "/"): Promise<Result> {
  const path = `${page} (site switch)`;
  try {
    const res = await fetch(base + page, { redirect: "manual" });
    const html = await res.text();
    const missing = SWITCH_PRESENT.filter((m) => !html.includes(m));
    const present = SWITCH_ABSENT.filter((m) => html.includes(m));
    const ok =
      res.status === 200 && missing.length === 0 && present.length === 0;
    return {
      path,
      ok,
      detail: ok
        ? "200 four tracks, no Compute in transit"
        : `${res.status} missing [${missing.join(", ")}] unexpected [${present.join(", ")}]`,
    };
  } catch (err) {
    return { path, ok: false, detail: (err as Error).message };
  }
}

/** Expect a redirect with this status to exactly this location. */
async function checkRedirect(
  url: string,
  status: number,
  location: string,
): Promise<Result> {
  const path = `${url} → ${status}`;
  try {
    const res = await fetch(url, { redirect: "manual" });
    const got = res.headers.get("location") ?? "";
    // a relative Location (Vercel sends "/training" for a same-host redirect)
    // is resolved against the request, as a browser does
    const resolved = got ? new URL(got, url).href : "";
    const ok = res.status === status && resolved === new URL(location).href;
    return { path, ok, detail: `${res.status} → ${got || "(none)"}` };
  } catch (err) {
    return { path, ok: false, detail: (err as Error).message };
  }
}

/** "Track › Site" from the first `SitePlace` at or after `from` in the HTML. */
function placeAt(html: string, from: number): string {
  const m =
    /data-site-place="[a-z]+"><span[^>]*>([^<]+)<\/span>.*?<span class="font-medium">([^<]+)<\/span>/s.exec(
      html.slice(Math.max(from, 0)),
    );
  return m ? `${m[1]} › ${m[2]}` : "?";
}

/** Each Foundations site's own name, as its header's brand shows it (the title up to the colon). */
const SITE_NAMES: Record<string, string> = {
  decoder: "Transformer Decoder Explained",
  training: "Transformer Training Explained",
};

/**
 * The site the header names, the text of its brand (the site's own name),
 * the place the menu button shows ("Foundations › Training") and how many
 * times the header shows a place, in a page's HTML.
 */
function headerNames(html: string): {
  site: string;
  brand: string;
  label: string;
  places: number;
} {
  const m = /data-site-name="([a-z]+)"[^>]*>([^<]*)<\/a>/.exec(html);
  const site = m?.[1] ?? "?";
  const brand = m?.[2] ?? "?";
  const label = placeAt(html, html.indexOf("<summary"));
  const header = /<header[\s\S]*?<\/header>/.exec(html)?.[0] ?? "";
  const places = header.split("data-site-place=").length - 1;
  return { site, brand, label, places };
}

/**
 * A page's header names `site`: its brand shows the site's own name, the
 * menu button reads `label` ("Foundations › Training") and is the only place
 * shown, the header never says "Part 1/2", and
 * (when given) its canonical URL is `canonical`.
 */
async function checkHeader(
  url: string,
  site: string,
  label: string,
  canonical?: string,
): Promise<Result> {
  const path = `${url} (header)`;
  try {
    const res = await fetch(url, { redirect: "manual" });
    const html = await res.text();
    const names = headerNames(html);
    const canon =
      /<link rel="canonical" href="([^"]+)"/.exec(html)?.[1] ?? "(none)";
    const header = /<header[\s\S]*?<\/header>/.exec(html)?.[0] ?? "";
    const ok =
      res.status === 200 &&
      names.site === site &&
      names.label === label &&
      names.brand === SITE_NAMES[site] &&
      names.places === 1 &&
      !/Part [12]/.test(header) &&
      (canonical === undefined || canon === canonical);
    return {
      path,
      ok,
      detail:
        `${res.status} site=${names.site} brand="${names.brand}" menu="${names.label}" places=${names.places}` +
        (canonical === undefined ? "" : ` canonical=${canon}`),
    };
  } catch (err) {
    return { path, ok: false, detail: (err as Error).message };
  }
}

/** The author credit: the last line of /about (brief 35), which both hosts show. */
const CREDIT = [
  'data-author-credit="true">Built by',
  'href="https://github.com/BrendanJamesLynskey" class="',
  ">Brendan Lynskey</a>",
];

/** `url` (followed through redirects when `follow`) shows the author credit. */
async function checkCredit(url: string, follow = false): Promise<Result> {
  const path = `${url} (author credit)`;
  try {
    const res = await fetch(url, { redirect: follow ? "follow" : "manual" });
    const html = await res.text();
    const missing = CREDIT.filter((m) => !html.includes(m));
    const ok = res.status === 200 && missing.length === 0;
    return {
      path,
      ok,
      detail: ok
        ? `${res.status} at ${res.url}`
        : `${res.status} missing [${missing.join(", ")}]`,
    };
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

/** Parse `[url] [--training <url>] [--comments-section <slug|none>]`. */
function parseArgs(argv: string[]): {
  base: string;
  training: string | null;
  commentsSection: string;
} {
  let base = "http://localhost:3000";
  let training: string | null = null;
  let commentsSection = DEFAULT_COMMENTS_SECTION;
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]!;
    if (arg === "--comments-section") {
      commentsSection = argv[++i] ?? "";
    } else if (arg.startsWith("--comments-section=")) {
      commentsSection = arg.slice("--comments-section=".length);
    } else if (arg === "--training") {
      training = argv[++i] ?? "";
    } else if (arg.startsWith("--training=")) {
      training = arg.slice("--training=".length);
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
  base = base.replace(/\/$/, "");
  // the production decoder host implies the production training host
  if (training === null && base === DECODER_ORIGIN) training = TRAINING_ORIGIN;
  return {
    base,
    training: training?.replace(/\/$/, "") ?? null,
    commentsSection,
  };
}

async function main(): Promise<void> {
  const { base, training, commentsSection } = parseArgs(process.argv.slice(2));
  console.log(
    `Smoke-checking ${base}` + (training ? ` and ${training} (Training)` : ""),
  );

  const results = [await checkHealth(base)];
  for (const path of DECODER_PAGES) results.push(await checkPage(base, path));
  results.push(await checkSwitch(base));
  results.push(
    await checkHeader(`${base}/learn`, "decoder", "Foundations › Decoder"),
  );
  results.push(await checkCredit(`${base}/about`));
  const chapter = `/training/${TRAINING_SECTIONS[12]!.slug}`;
  if (training) {
    // Training on its own host; the old address redirects there
    results.push(await checkHealth(training));
    for (const path of TRAINING_PAGES)
      results.push(await checkPage(training, path));
    results.push(await checkPage(training, "/signin"));
    results.push(await checkSwitch(training, "/training"));
    results.push(
      await checkHeader(
        training + chapter,
        "training",
        "Foundations › Training",
        TRAINING_ORIGIN + chapter,
      ),
    );
    results.push(
      await checkRedirect(`${training}/`, 307, `${training}/training`),
    );
    results.push(
      await checkRedirect(`${training}/learn`, 308, `${base}/learn`),
    );
    // the one About page, reached from the training host too
    results.push(
      await checkRedirect(`${training}/about`, 308, `${base}/about`),
    );
    results.push(await checkCredit(`${training}/about`, true));
    for (const path of ["/training", chapter])
      results.push(
        await checkRedirect(`${base}${path}`, 308, `${training}${path}`),
      );
  } else {
    for (const path of TRAINING_PAGES)
      results.push(await checkPage(base, path));
    results.push(
      await checkHeader(
        base + chapter,
        "training",
        "Foundations › Training",
        TRAINING_ORIGIN + chapter,
      ),
    );
  }
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
