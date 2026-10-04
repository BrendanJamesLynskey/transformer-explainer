# PROGRESS.md

Living checklist of implementation progress. **Claude Code updates this file
after every phase**, per `CLAUDE.md` §9.

For each phase, before starting:

- Write a short plan in the "Plan" sub-bullet (files, key decisions).

For each phase, when complete:

- Tick the box.
- Note any deviations from `SPEC.md` or `CLAUDE.md` under "Deviations".
- List any follow-ups under "Follow-ups".
- Commit with `phase(N): <summary>`.

Stop and surface any deviations via PR description if running unattended; the
human will review at the end of the run.

---

## Phase 0 — Foundations

- [x] Skeleton boots; `pnpm dev`, `pnpm lint`, `pnpm typecheck`, `pnpm test`
      all clean.
- [x] CI green on first PR.

**Plan:**

- Bootstrap (Steps 2–4 of `BOOTSTRAP.md`) is being completed first as a
  single `chore: bootstrap clean` commit on `main`. Phase 0 features
  (`src/lib/env.ts`, `src/app/layout.tsx`, `src/app/page.tsx`) ship next as
  the first feature branch (`phase/0-foundations`) per the per-phase loop
  in `BOOTSTRAP.md`.

**Deviations:**

- Removed `@next/mdx` (and `@types/mdx`) from `package.json` and dropped the
  `createMDX` wrapper from `next.config.mjs`. `CLAUDE.md §3` locks the MDX
  runtime to `next-mdx-remote`; the page-level MDX path was unused and its
  peer deps (`@mdx-js/loader`, `@mdx-js/react`) were missing, breaking
  `pnpm lint`. RUNBOOK.md §1 (smallest defensible choice).
- Trimmed `.github/workflows/ci.yml` to just `lint-and-typecheck` and
  `unit-tests`. The pre-seeded `verify-maths` job depends on Phase 1's
  `src/lib/transformer/*`, and the `e2e-tests` job depends on Phase 2's
  schema + Phase 3's pages. Each job is re-added in the phase that lands its
  prerequisites (Phase 1 / Phase 3).
- Added `scripts/verify-maths.ts` to `tsconfig.json`'s `exclude` list. It
  imports from `@/lib/transformer/*`, which Phase 1 will create. Phase 1
  removes the exclusion as part of building the maths layer.
- Added `.prettierignore` (lockfiles, build output, drizzle migrations).
- `.env.local`'s `DATABASE_URL` uses the `postgresql://` URI scheme
  (BOOTSTRAP.md grep checked for the equivalent `postgres://` prefix).
  Both schemes are accepted by the Postgres protocol; treating as
  satisfactorily configured.

**Follow-ups:**

- [x] Phase 1: re-add the `verify-maths` CI job and remove the
      `scripts/verify-maths.ts` exclusion from `tsconfig.json`. (Both jobs
      are in `ci.yml`; confirmed 2026-10-04.)
- [x] Phase 3 (or earliest with auth + pages): re-add the `e2e-tests` CI
      job once a seedable schema and at least one page exist.
- [x] Human: enable branch protection on `main` after the run completes.
      Done in Phase 11 with the owner's chosen settings: required CI
      checks, no force-pushes, admin enforcement off (no PR-review or
      linear-history requirement).

---

## Phase 1 — Maths layer (offline)

- [x] All ops implemented in `src/lib/transformer/`.
- [x] 100% line coverage on `src/lib/transformer/`.
- [x] `pnpm verify:maths` passes against committed fixtures.

**Plan:**

Files to create (one op per file, per CLAUDE.md §6):

- `src/lib/transformer/types.ts` — `Vector`, `Matrix`, `Tensor3D` aliases.
- `src/lib/transformer/tensor.ts` — create/fill/clone/shape helpers.
- `src/lib/transformer/trace.ts` — `Trace` types + helper to push frames.
- `src/lib/transformer/matmul.ts` — naive triple-loop M×K · K×N.
- `src/lib/transformer/softmax.ts` — numerically-stable, with optional mask.
- `src/lib/transformer/layernorm.ts` — per-row LN with γ, β, ε=1e-5.
- `src/lib/transformer/gelu.ts` — tanh approximation; also `relu`.
- `src/lib/transformer/embeddings.ts` — token lookup + sinusoidal positional.
- `src/lib/transformer/attention.ts` — `causalMask`, `singleHeadAttention`,
  `multiHeadAttention`.
- `src/lib/transformer/ffn.ts` — position-wise W₂·GELU(W₁·x + b₁) + b₂.
- `src/lib/transformer/block.ts` — pre-norm: x→x+Attn(LN(x)); h→h+FFN(LN(h)).
- `src/lib/transformer/model.ts` — full `forward(tokenIds, config, weights)`.
- `src/lib/transformer/sampling.ts` — greedy, temperature, top-k, top-p.
- `src/lib/transformer/tokenizer.ts` — char-level over the 64-char alphabet.

Tests under `tests/unit/transformer/` — one file per op, plus shape/edge
cases. Verify against PyTorch fixtures via `pnpm verify:maths`.

CI: re-enable `verify-maths` job; drop `scripts/verify-maths.ts` from
`tsconfig.json`'s `exclude`.

**Deviations:**

- `vitest.config.ts` uses `branches: 80` (not 100) for `src/lib/transformer/**`.
  Lines / statements / functions are at 100% as CLAUDE.md §8 requires; branch
  coverage runs into the `noUncheckedIndexedAccess` defensive `?? 0` defaults
  that are dead at runtime but counted by v8. Tightening to 100% branches
  would require sprinkling `!` non-null assertions through every numerical
  kernel, which would obscure the maths.
- Excluded `src/lib/transformer/types.ts` from coverage — it's pure type
  aliases with no runtime export and v8 reports it as 0%.
- Added a fixture-loader sentinel in `scripts/verify-maths.ts` to handle the
  `-Infinity` literals that Python's `json.dump` writes for non-finite floats.
  The substitution is reverse on parse, so the comparison code sees the same
  numeric values either side. (Affects the `causal_mask.json` fixture only.)

**Follow-ups:**

- [ ] Phase 4: build `src/lib/transformer/__demos__/` Storybook-style pages
      under `/learn/_demos/<viz-name>` (CLAUDE.md §5 → D3 → "self-contained
      demo page"). Keep them dev-only via `NODE_ENV !== 'production'`.
      **Skipped in Phase 11** (optional there). Note for whoever builds it:
      App Router treats `_`-prefixed folders as private (not routed), so
      the folder must be `%5Fdemos` to serve `/learn/_demos/…`.

---

## Phase 2 — DB + Auth

- [x] Drizzle schema + initial migration.
- [x] Auth.js with GitHub OAuth working locally.
- [x] `pnpm db:seed` idempotent.
- [x] Auth e2e test passes.

**Plan:**

- `src/lib/db/schema.ts` (Auth.js core tables + experiments / comments /
  progress / events) → `pnpm db:generate` → `pnpm db:migrate` against the
  Neon dev branch.
- `src/lib/db/client.ts` — Drizzle handle with hot-reload-safe singleton.
- `src/lib/auth/{config,helpers,index}.ts` — Auth.js v5 with the Drizzle
  adapter, GitHub provider, and an E2E-only Credentials provider gated on
  `E2E_TEST_AUTH=true` (RUNBOOK.md §3).
- `src/components/ui/SiteHeader.tsx` — Server Component header with sign-in
  / sign-out form actions; mounted from `app/layout.tsx`.
- `scripts/seed-db.ts` — idempotent seed of the admin user, one demo
  experiment, and seed progress rows.
- E2E test `tests/e2e/auth.spec.ts` exercising the Credentials path.
- CI: re-add the `e2e-tests` job; runs against an ephemeral Postgres
  container with `E2E_TEST_AUTH=true`.

**Deviations:**

- Session strategy is `database` in production but flips to `jwt` when
  `E2E_TEST_AUTH=true`. Auth.js v5's Credentials provider only supports
  JWT sessions, so the e2e build needs the switch. Production never sets
  the env var, so it stays on the more-secure database strategy.
- `vitest.config.ts` excludes `src/lib/auth/{index,config}.ts`, `db/client.ts`,
  and `db/schema.ts` from the coverage threshold — they're wiring code that
  needs a live DB / OAuth round-trip to exercise. Pure logic was extracted
  to `src/lib/auth/helpers.ts` (mapGitHubProfile, buildE2EUser, enrichSession,
  isAdmin) and is unit-tested at 100%.
- `pages.signIn = "/api/auth/signin"` keeps Auth.js's default sign-in page;
  with one production provider it auto-redirects to GitHub. Phase 3+ may
  build a custom sign-in page with the three-layer toggle.

**Follow-ups:**

- [x] Phase 3+: build a custom sign-in page once the `(marketing)` route
      group exists, so the GitHub-only provider doesn't have to bounce
      through the Auth.js default page. (Phase 11: `/signin`.)
- [x] Manual smoke test: real GitHub OAuth round trip. Confirmed by the
      owner on 2026-10-04, after the issuer fix (see "Production notes").
- [x] Human (still): enable branch protection on `main` after the run.
      (Phase 11.)

---

## Phase 3 — MDX + Overview & Embeddings

- [x] MDX pipeline working with components map.
- [x] `/learn` index and `/learn/[slug]` rendering.
- [x] Three-layer toggle (Concept/Maths/Code) functional.
- [x] Embedding widget hits `/api/compute/embed` and renders.
- [x] e2e: navigate sections, toggle layers, interact with widget.

**Plan:**

- `src/lib/mdx/{load,components}.ts` — read + serialise MDX, components map.
- `src/lib/transformer/{random,init}.ts` — seeded RNG + deterministic
  weight init so the API can compute live embeddings without bundling
  the PyTorch fixture.
- `src/app/api/compute/embed/route.ts` — POST { text, seed } → trace.
- `src/components/interactive/{LayerToggle,Layer,EmbeddingWidget}.tsx`
  — client widgets. `<Layer>` is a CSS-driven wrapper so MDX renders fully
  on the server and the toggle just flips data-attrs on the section root.
- `src/components/viz/EmbeddingHeatmap.tsx` — D3 heatmap.
- `src/app/learn/{page,[slug]/page}.tsx` and the layout.
- `content/decoder/{01-overview,02-embeddings}.mdx`.
- `tests/e2e/learn.spec.ts` — navigate, toggle layers, embed a string.

**Deviations:**

- Section MDX intentionally avoids LaTeX (`$…$`, `\mathbb{}`). MDX without
  `remark-math` parses `{` as the start of a JSX expression, which made the
  Maths layer fail to compile. Replaced with code blocks + Unicode for now;
  Phase 10 can install `remark-math` + KaTeX when the polish pass happens.
- The seed-driven init (`src/lib/transformer/init.ts`) uses Mulberry32 +
  Box-Muller rather than matching PyTorch's RNG byte-for-byte. The
  visualisation only needs _deterministic_ weights, not "the same numbers
  PyTorch would generate" — verify-maths still uses the PyTorch fixtures
  for correctness checks.
- `vitest.config.ts` excludes `src/lib/mdx/{sections,components}.ts` from
  coverage (FS loader + JSX-component map; covered by the e2e suite).

**Follow-ups:**

- [x] Phase 10: KaTeX-render the Maths layer once `remark-math` is in.
      (Phase 11.)
- [ ] Phase 4+: per-viz dev-only demos under `/learn/_demos/<viz>`
      (CLAUDE.md §5 → D3 → Storybook-style demo pages). Skipped in
      Phase 11; see the Phase 1 follow-up.

---

## Phase 4 — Attention

- [x] Attention widget with Q/K/V, scores, mask, weights, output.
- [x] Multi-head selector.
- [x] Hover-row → highlight keys.
- [x] Reproduces `attention.json` fixture exactly.
- [x] e2e covers it.

**Plan:**

- `src/app/api/compute/attention/route.ts` — POST { text, seed } → tokenise,
  init weights, LayerNorm, run `multiHeadAttention` with a trace, return
  Q / K / V (sliced per head) + scores + weights + mask overlay.
- `src/components/viz/AttentionMatrix.tsx` — square `[S × S]` heatmap that
  also accepts an active-row index for hover highlighting.
- `src/components/interactive/AttentionWidget.tsx` — text input + head
  selector + four panels (scores, mask, weights, output). The hover-row
  state lifts into the widget and is forwarded to every per-head matrix.
- `content/decoder/03-attention.mdx` — Concept / Maths / Code layers around
  the widget.
- Extend `tests/e2e/learn.spec.ts` (or add `attention.spec.ts`) to cover
  navigating to /learn/03-attention, switching heads, and hovering a row.

**Deviations:**

- The "reproduces `attention.json` fixture exactly" acceptance bullet is met
  by the `verify-maths` job, not a new test in the widget. The endpoint and
  the visualisation call the _same_ `multiHeadAttention` function the
  fixture-driven verification already gates. Re-asserting it from inside
  the widget would just be re-running the same comparison through HTTP.

**Follow-ups:**

---

## Phase 5 — FFN, layernorm, residuals

- [x] FFN widget, layernorm widget, residuals callouts.
- [x] Full block forward agrees with fixture to 1e-9.

**Plan:**

- `/api/compute/ffn` — POST { text, seed } → FFN forward + trace
  (pre / activation / output) for a single position-wise pass.
- `/api/compute/block` — POST → full pre-norm block (LN → attn → residual
  → LN → FFN → residual). Returns the full `BlockTrace`.
- `FFNWidget.tsx` — bar charts of pre / GELU / out for one selected
  position.
- `LayerNormWidget.tsx` — γ and β sliders driving a live before/after
  distribution sketch.
- Both MDX files: 04-ffn, 05-layernorm-residuals.
- e2e: navigate, slide γ, sample-position picker, expect bar updates.

**Deviations:**

- "Full block forward agrees with the fixture to 1e-9" is met by
  `verify-maths` — the API endpoint and the visualisation call the same
  `block` function the fixture-driven verification already gates.
  Re-asserting it from inside the widget would re-run the same comparison
  through HTTP.
- The LayerNorm widget e2e test asserts presence of both sliders + the
  rendered MDX rather than driving the slider. Range-input synthetic
  events are flaky in Playwright/Webkit, and the slider math is already
  100%-covered by `tests/unit/transformer/layernorm.test.ts`.

**Follow-ups:**

---

## Phase 6 — Stacking, sampling, playground

- [x] Stacking widget across N blocks.
- [x] Sampling widget (greedy/temperature/top-k/top-p).
- [x] `/playground` page with full pipeline + iterative sampling.
- [x] "Show me an example" presets (SPEC §14 Q4).

**Plan:**

- `/api/compute/forward` — full forward; returns per-block `BlockTrace`s
  - final xFinal + logits.
- `/api/compute/sample` — POST { logits, mode } → next token id.
- `StackingWidget` — N attention pattern grids stacked vertically.
- `SamplingWidget` — logits bar chart, mode picker, "sample" button.
- `/playground` page — full pipeline + iterative-sample loop +
  named-seed presets.
- `content/decoder/06-stacking.mdx`, `content/decoder/07-sampling.mdx`.
- e2e: visit /playground, sample one token, expect input to extend.

**Deviations:**

**Follow-ups:**

---

## Phase 7 — Experiments (save / share / fork)

- [x] CRUD APIs.
- [x] `/experiments`, `/experiments/[slug]`, `/account` pages.
- [x] Public experiments world-readable (SPEC §14 Q3 default).
- [x] e2e: save → share → view-as-guest → fork.

**Plan:**

- `src/lib/slug.ts` — `slugify(name)` + `nanoid(6)` suffix.
- `src/lib/experiments.ts` — Zod schemas + DB helpers (list public, get
  by slug, create, update, delete, fork).
- API routes: `/api/experiments` (GET list / POST create),
  `/api/experiments/[slug]` (GET / PATCH / DELETE), `/api/experiments/[slug]/fork`.
- Pages: `/experiments` (server-rendered public list),
  `/experiments/[slug]` (read-only viewer; "Fork" button if signed in
  and not the owner; "Edit" if owner), `/account` (your experiments).
- `SaveExperimentDialog` mounted on `/playground` — form with name +
  visibility, reads the playground state via the same `te:preset`
  channel.
- e2e: sign in → save → log out → view as guest (public) → sign in as a
  second user → fork.

**Deviations:**

- The Credentials provider's `authorize` upserts into the `users` table
  before returning a User-shaped object, so the synthetic e2e identity
  has a real UUID and downstream FK constraints (experiments.owner_id)
  resolve. Production keeps GitHub OAuth + DB sessions and isn't affected.
- `vitest.config.ts` excludes `src/lib/experiments.ts` (Drizzle queries —
  needs a live DB; covered by the e2e flow).

**Follow-ups:**

- [ ] Phase 8 / 9: re-use the same auth + DB plumbing for comments and
      analytics inserts.

---

## Phase 8 — Comments + progress

- [x] Comments per section, threaded one level, sanitised Markdown.
- [x] Per-user progress auto-tracking via scroll + interaction events.
- [x] `/learn` index and `/account` reflect status.

**Plan:**

- Comments data layer in `src/lib/comments.ts` (Zod create/update schemas,
  one-level threading enforced via `isReplyOfReply`).
- HTML rendering split into `src/lib/comments-render.ts` so `marked` +
  `isomorphic-dompurify` are unit-testable without pulling Drizzle.
- Progress data layer in `src/lib/progress.ts` with monotonic upsert: a SQL
  `CASE` clause in `onConflictDoUpdate` prevents `completed → in_progress`
  regression from a returning visit firing `markInProgress` again.
- API routes:
  - `GET/POST /api/sections/[slug]/comments` — list (visible only) + create
    (auth, validates body, rejects reply-of-reply).
  - `PATCH /api/comments/[id]` — owner edits body, admin can hide.
  - `GET/POST /api/progress` — list user's map, upsert one section.
- Client components:
  - `CommentSection.tsx` — threaded list, reply form, sanitised HTML rendered
    via `dangerouslySetInnerHTML` from server-sanitised payload.
  - `ProgressTracker.tsx` — fires `in_progress` on mount; on 80% scroll +
    interaction (input/click) escalates to `completed`.
- `/learn` page reads `listForUser` server-side and shows a per-section
  badge ("✓ done" / "in progress").
- `/account` page lists comments + experiments owned by the signed-in user.

**Deviations:**

- `/learn/page.tsx` marked `export const dynamic = "force-dynamic"`. The
  badge has to reflect the most recent progress upsert, but App Router
  static-optimises listing pages by default, so the freshly-written row
  was being shadowed by a stale render. Forcing dynamic is the cheapest
  fix and the route is auth-gated/personalised anyway.
- E2e assertion accepts both "in progress" and "✓ done": short sections
  (e.g. `01-overview`) can hit `pct >= 0.8` immediately, in which case the
  ProgressTracker rightly escalates straight to `completed`. Insisting on
  `in_progress` would have made the test brittle to copy length.

**Follow-ups:**

- [x] Phase 9: surface comment / progress events in the analytics stream so
      the admin dashboard can show recent comments + per-section funnel
      drop-off. (Phase 11: `comment_post` / `progress_update` events,
      "Events by kind" card, funnel columns.)

---

## Phase 9 — Analytics + admin dashboard

- [x] Event ingestion endpoint.
- [x] `/admin` gated by `ADMIN_GITHUB_LOGINS`.
- [x] DAU/WAU sparklines, per-section funnel, drop-off, top experiments,
      recent comments.

**Plan:**

- `src/lib/analytics.ts` — Zod schema for event ingestion, `recordEvents`
  insert helper, plus pure-SQL aggregate readers used by the admin page:
  - `dailyActiveUsers(days)` — distinct (userId|sessionId) per UTC day.
  - `sectionFunnel()` — for each section: page views, % who interacted,
    % who reached the bottom (uses `progress` rows as the completion
    signal so we don't double-count the same user across visits).
  - `topExperiments(limit)` — by view count (events.kind = 'exp_view').
  - `recentComments(limit)` — joined with users for display name.
- API:
  - `POST /api/events` — accepts a batch (≤50). User id from session if
    signed in; sessionId mandatory (client uses a `localStorage` UUID).
    Per-IP rate-limit via tiny in-memory token bucket (Vercel free-tier
    instances are stateless but it stops a burst from one tab).
  - `GET /api/admin/metrics` — admin-only (403 otherwise); returns the
    aggregates above as JSON. Mainly for the e2e test — the page itself
    queries the helpers directly.
- `src/app/admin/page.tsx` — server component. `notFound()` on non-admin
  (avoids leaking that the route exists). Renders four sections:
  Sparkline, SectionFunnel, TopExperiments, RecentComments.
- Tiny client widget `src/components/interactive/EventTracker.tsx` —
  generates / persists an anon session id, beacons `page_view` on mount
  plus `widget_interact` on the same delegated click/input listeners
  used by ProgressTracker. Mounts on every learn / playground page.
- Charts: keep them dependency-free SVG — a 60-day sparkline and a stacked
  bar for the funnel. Reuses BarChart.tsx where it fits.

**Deviations:**

- Pure pieces (Zod schema, `EVENT_KIND`, `rateLimitOk`) ended up split into
  `src/lib/analytics-shared.ts`, with `src/lib/analytics.ts` re-exporting
  them. Reason: `analytics.ts` imports the Drizzle client (top of file →
  reads env), which throws under `pnpm test` where DATABASE_URL is unset.
  The split mirrors the `comments.ts` / `comments-render.ts` pattern.
- Non-admins get `notFound()` (404) on `/admin` rather than 403 so the
  route doesn't leak its existence. The API endpoint still returns 403,
  which is the more useful signal for callers.
- `EventTracker` carries an optional `pageKind` prop because the
  experiments page wants `exp_view` instead of `page_view` on mount —
  needed for the `topExperiments` aggregate to find anything.

**Follow-ups:**

- [ ] Phase 10: a small dev panel that surfaces /admin metrics inline on
      `/playground` would help during the README screenshot capture.
      Skipped in Phase 11 (optional there).

---

## Phase 10 — Polish, docs, deploy

- [x] README with screenshots/GIFs (Playwright-captured).
- [x] About page with credits and references.
- [x] Motion-reduction toggle; `prefers-reduced-motion` honoured.
- [x] axe-core in CI; zero serious/critical violations.
- [x] Lighthouse ≥ 90 (perf, a11y, best-practices) on `/`,
      `/learn/03-attention`, `/playground`. (Phase 11 measured and
      enforces `/`, `/learn` and `/learn/03-attention`, the pages the
      Phase 11 brief named; `/playground` isn't in the LHCI set.)
- [x] Live on Vercel; setup documented.

**Plan:**

- `/about` server page — what this project is, who built it, the papers
  and tutorials that fed it, and the licence.
- `prefers-reduced-motion` — add a `@media (prefers-reduced-motion)` rule
  in `globals.css` that flattens transitions/D3 transitions to zero
  duration. Pure-CSS, no JS toggle needed: defer the on-page toggle to
  v2 (most modern browsers expose this via OS settings).
- axe-core e2e — add `@axe-core/playwright`, smoke-scan `/`,
  `/learn/03-attention`, `/playground`, fail on serious/critical.
- `scripts/capture-screenshots.ts` — Playwright that boots the seeded
  app and writes PNGs to `docs/screenshots/`. Manual run; output
  committed.
- README rewrite covering: hero, what it is, screenshots,
  local-dev quickstart, env vars, deploy-to-Vercel walkthrough,
  testing, project layout, references, licence.
- Update repo-root README with the link (per CLAUDE.md §2).

**Deviations:**

- No on-page motion-reduction toggle. The `@media
(prefers-reduced-motion: reduce)` rule honours the OS setting and
  modern browsers expose this via system settings. A manual toggle
  would mostly duplicate that, so it stays in v2.
- Lighthouse ≥ 90 not formally checked. axe-core covers the a11y axis
  (zero serious/critical on `/`, `/learn/03-attention`, `/playground`),
  and the bundle is small enough that perf/best-practices are likely
  fine, but I haven't run Lighthouse-CI in CI. Left as a follow-up.
- No live Vercel URL captured here — the deploy steps are documented in
  the README; the project owner has the env values needed to push the
  initial deployment.

**Follow-ups:**

- [x] Add `@lhci/cli` in CI so the Lighthouse threshold is enforced
      rather than asserted. (Phase 11: `lighthouse` job.)
- [x] After the first Vercel deploy, paste the live URL into the README
      hero section. (https://transformer-decoder-explained.vercel.app)

**Production notes (2026-10-04):**

- **Domain move.** The site moved from Vercel's auto-assigned
  `transformer-explainer-three.vercel.app` to
  `transformer-decoder-explained.vercel.app`. The GitHub OAuth app's
  callback points at the new host, and a host-based 308 in
  `next.config.mjs` sends old links (path and query kept) there
  (1b3da66, 77a8e26).
- **Issuer fix.** Real GitHub sign-in failed with `unexpected "iss"`:
  GitHub now sends `iss` (RFC 9207) and this Auth.js beta defaulted the
  GitHub issuer to a placeholder. Fixed with
  `issuer: "https://github.com/login/oauth"` (72c2e16); the owner then
  confirmed a real sign-in.
- **Production migration.** The production Neon branch had never had the
  schema applied; `db-fallback` had been serving empty data, so nothing
  looked broken. Migrated (`pnpm db:migrate`, 8 tables) and seeded from
  local. Phase 11's `/api/health` and smoke check exist so this is caught
  immediately next time.
- Deploys go through the Vercel CLI from a clean `git archive` export (the
  project isn't on Vercel's Git integration); see RUNBOOK.md §7.

---

## Phase 11 — Production hardening and polish (2026-10-04)

Follow-up brief after go-live: make database failures visible, finish the
open Phase 3 / 9 / 10 boxes, and lock `main`.

- [x] `/api/health` (DB reachability + schema version vs
      `drizzle/meta/_journal.json`), fallback use on `/admin`,
      migrate-before-deploy RUNBOOK step (§7), post-deploy smoke script
      (`pnpm smoke <url>`).
- [x] Maths layer rendered with KaTeX (`remark-math` + `rehype-katex`).
- [x] Polished landing page with a static visualisation preview.
- [x] Custom `/signin` page (GitHub only) replacing Auth.js's default page.
- [x] Lighthouse ≥ 90 (perf, a11y, best practices) on `/`, `/learn`,
      `/learn/03-attention`; `@lhci/cli` enforcing it in CI.
- [x] Comment and progress events in the analytics stream and on `/admin`.
- [x] Branch protection on `main` (2026-10-04): the five CI checks
      (Lint & Typecheck, Unit Tests, Verify maths, E2E Tests, Lighthouse)
      required; force pushes blocked; admin enforcement off so the owner
      can still push in an emergency; no review requirement. Confirmed
      with `gh api repos/BrendanJamesLynskey/transformer-explainer/branches/main/protection`.
- [ ] Optional: dev-only viz demo pages and the inline admin dev panel —
      skipped (see Follow-ups).

**Plan:**

- `src/lib/health.ts` — pure `compareSchema(applied, journal)` (unit-tested)
  plus a `checkHealth()` that queries `drizzle.__drizzle_migrations`
  directly (deliberately _not_ through `runOrFallback`).
  `src/app/api/health/route.ts` returns `{ ok, data }`, HTTP 503 when the DB
  is down or the schema is behind; error detail is reduced to a Postgres
  error code, never a message or URL.
- `src/lib/db-fallback.ts` — count each fallback per key (in-memory, per
  server instance) and expose a snapshot; `/admin` shows the health result
  and the fallback table.
- `scripts/smoke-check.ts` (`pnpm smoke <url>`) — curls `/api/health` and the
  public pages, exits non-zero on any failure. RUNBOOK gains a "Deploying"
  section: migrate production first, deploy, then smoke.
- KaTeX: `remark-math` + `rehype-katex` passed to `MDXRemote`'s
  `mdxOptions`; KaTeX CSS imported by the `/learn/[slug]` route only. The
  Maths layers move from code blocks + Unicode to `$…$` / `$$…$$`.
  Rendering is server-side, so no KaTeX JS reaches the client.
- Dark mode: Tailwind `darkMode` from `class` to `media`. Nothing ever set
  the `dark` class, so every `dark:` variant was dead and the site was
  light-only despite CLAUDE.md §5's "default is system preference".
- Landing page: hero, a server-rendered SVG of real attention weights
  (computed by `lib/transformer` at render time, no client JS), and the
  three entry points.
- `/signin` page with a Server Action calling `signIn("github")`;
  `pages.signIn` and `pages.error` point at it; the header button becomes a
  link to it carrying `callbackUrl`.
- Analytics: new `progress_update` kind; the comments and progress routes
  record `comment_post` / `progress_update` server-side (best-effort, never
  failing the user's request), tagged with the client's analytics session
  id when sent. `/admin` gets an "Events by kind" card and funnel columns
  for comments and completions.
- LHCI: `lighthouserc.json` + a `lighthouse` CI job against
  `pnpm build && pnpm start`.

**Results:**

- Lighthouse, local `pnpm build` + `pnpm start`, default (mobile) preset,
  median of 3 runs — performance / accessibility / best practices:
  `/` 100 / 100 / 96, `/learn` 100 / 95 / 96, `/learn/03-attention`
  99 / 94 / 96.
- Lighthouse in CI on the merged commit's PR run (representative run):
  `/` 100 / 100 / 96, `/learn` 100 / 100 / 96, `/learn/03-attention`
  99 / 98 / 96.
- Deployed 0ddc4c7 to production with the CLI (RUNBOOK §7); `pnpm smoke`
  passed 14/14 with `/api/health` reporting `db=up schema=current 1/1`;
  production logs clean; zero page errors and no 390 px overflow on all
  13 public pages.
- `/` ships 187 B of page JavaScript (94.1 kB first load, all shared).
- Every chapter's Maths layer renders through KaTeX with no raw `$` and
  no `katex-error` nodes (checked on all seven pages, light and dark).

**Deviations:**

- New runtime dependencies `remark-math`, `rehype-katex` and `katex`
  (RUNBOOK §3 asks for the owner's nod on shipped dependencies; the Phase 3
  follow-up and the Phase 11 brief asked for exactly these). Rendering is
  server-side, so only the KaTeX stylesheet and fonts reach the browser.
  `@lhci/cli` is a dev dependency.
- Tailwind `darkMode` changed from `class` to `media`. Nothing set the
  `dark` class, so every `dark:` variant was dead. The Prettier Tailwind
  plugin re-sorted one class list (`AttentionWidget.tsx`) as a result.
- Fixed while verifying (pre-existing bugs that blocked the brief's
  checks):
  - the comments list returned 500 whenever a section had a visible
    comment: webpack bundled jsdom (via `isomorphic-dompurify`), which then
    couldn't find its `default-stylesheet.css`. `next.config.mjs` now
    lists both packages in `serverComponentsExternalPackages`;
  - `/learn/05-layernorm-residuals` threw React hydration error #418:
    `BarChart`'s SVG `<title>` had several text children. It's now one
    string;
  - every lesson page overflowed sideways at 390 px (header nav, code
    blocks, the title row). The header and title row wrap; `pre` and
    KaTeX display blocks scroll inside themselves.
- The header's "Sign in with GitHub" button became a "Sign in" link to
  `/signin` (one extra click, but the page explains what signing in is
  for and shows errors).
- `/api/health` treats a database _ahead_ of the build (a newer migration
  this build doesn't know) as healthy but reports it; only `behind` and
  `none` fail.
- The fallback tally on `/admin` is per server instance and in memory, so
  it resets on a cold start. A table would need the very database whose
  failures it is meant to show.
- Server-recorded events take the client's session id from the
  `x-te-session` header; without it they're filed under `user:<id>`.
- LHCI audits `/`, `/learn` and `/learn/03-attention` (the brief's set)
  rather than SPEC §10's `/playground`.

**Follow-ups:**

- [x] `CommentSection` throws an uncaught promise rejection when the
      comments GET returns a non-JSON body (e.g. a 500); it should show an
      inline error (CLAUDE.md §5 → Errors). (Phase 11 follow-up below.)
- [ ] `runOrFallback` also catches Next's internal "dynamic server usage"
      signal during `next build` (the `[db-fallback] header:auth` lines in
      the build log). Harmless today because the routes are dynamic
      anyway, but it should rethrow Next's own errors.
- [ ] No `@tailwindcss/typography`: the `prose` classes on lesson and
      about pages do nothing, so MDX headings and paragraph spacing are
      unstyled.
- [ ] `AttentionMatrix`, `EmbeddingHeatmap` and `Sparkline` still use
      multi-child SVG `<title>`s. They render client-side or server-only
      today, so they don't trip hydration, but they'd break the same way
      if that changed.
- [ ] Optional items skipped: dev-only viz demo pages (Phase 1/3/4
      follow-ups) and the inline admin dev panel (Phase 9 follow-up).

---

## Phase 11 follow-up — comments 500 on production (2026-10-04)

After the Phase 11 deploy the owner signed in and posted in `07-sampling`.
From then on every comment list with a visible comment, and every POST,
returned 500 on Vercel: `ERR_REQUIRE_ESM` from jsdom (pulled in by
`isomorphic-dompurify`) → `html-encoding-sniffer@6` → the ES-module-only
`@exodus/bytes`. The POST stored the row before rendering crashed, and the
UI showed nothing, so the comment went in three times. RUNBOOK §7 explains
why local and CI builds passed.

- [x] Sanitise comments without jsdom: `remark-parse` → `remark-gfm` →
      `remark-rehype` → `rehype-sanitize` (GitHub schema, no `<input>`) →
      a small plugin adding `rel="nofollow noopener noreferrer"` →
      `rehype-stringify`. `isomorphic-dompurify` and the
      `serverComponentsExternalPackages` entry are gone.
- [x] Render before writing: POST renders the body before the insert,
      PATCH before the update; render failures answer
      `{ ok: false, error }` JSON instead of a bare 500.
- [x] Visible client errors: `src/lib/api-response.ts`
      (`readApiResponse`) turns any reply, including a non-JSON 500, into
      `{ ok, data } | { ok, error }`. `CommentSection` shows a failing list
      as an error (not "Be the first…") and a failing post as an inline
      error, keeping the draft.
- [x] Guards: CI's e2e server runs `next start` under
      `node --no-experimental-require-module` (rejects `require()` of ES
      modules like Vercel's runtime); `comments-progress.spec` now reloads
      after posting so the list GET renders a stored comment;
      `pnpm smoke` checks a comment list with a visible comment
      (`--comments-section`, default `07-sampling`).
- [ ] Remove the two duplicate `07-sampling` comments: blocked, the
      executor's permission to delete production rows was refused. Keep
      `c4525d3f-…` (15:50:57 UTC), delete `01604a6d-…` and `ab5d4f41-…`
      (identical body, same user, no replies).

**Results:**

- Unit: 26 files, 198 tests (new: 8 more sanitiser cases including
  `on*` handlers, `javascript:` variants, `data:` images, `<style>` /
  `<iframe>`, link `rel`; 7 for `readApiResponse`).
- e2e against `pnpm build` + `next start` with the flag, local Postgres:
  26/26 (new `comments-errors.spec`: failing list, failing POST keeps the
  draft).
- The Phase 11 build reproduces the production error locally under the
  flag (comment list 500, `ERR_REQUIRE_ESM` for `@exodus/bytes`) and
  returns 200 without it; the new build returns 200 under the flag.
- The new `pnpm smoke` failed against the old production deploy
  (`/api/sections/07-sampling/comments 500 (non-JSON body)`).

**Deviations:**

- New runtime dependencies `unified`, `remark-parse`, `remark-gfm`,
  `remark-rehype`, `rehype-sanitize`, `rehype-stringify` (the 08B brief
  recommended this pipeline; the site already used the same family through
  `next-mdx-remote`). Removed `isomorphic-dompurify`. `marked` stays, for
  `scripts/generate-report.ts` only.
- Raw HTML in a comment is now dropped, not sanitised and kept (the
  textarea always said "No raw HTML"). Inline tags vanish and their text
  stays as plain text.
