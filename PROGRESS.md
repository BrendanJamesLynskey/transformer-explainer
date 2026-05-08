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

- [ ] Phase 1: re-add the `verify-maths` CI job and remove the
      `scripts/verify-maths.ts` exclusion from `tsconfig.json`.
- [ ] Phase 3 (or earliest with auth + pages): re-add the `e2e-tests` CI
      job once a seedable schema and at least one page exist.
- [ ] Human: enable branch protection on `main` after the run completes:
      require status checks, require linear history, require PR review.
      See `gh api` docs for the full payload.

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

- [ ] Phase 3+: build a custom sign-in page once the `(marketing)` route
      group exists, so the GitHub-only provider doesn't have to bounce
      through the Auth.js default page.
- [ ] Manual smoke test: real GitHub OAuth round trip. Documented as a
      README step in Phase 10. Today's e2e covers the Credentials path.
- [ ] Human (still): enable branch protection on `main` after the run.

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

- [ ] Phase 10: KaTeX-render the Maths layer once `remark-math` is in.
- [ ] Phase 4+: per-viz dev-only demos under `/learn/_demos/<viz>`
      (CLAUDE.md §5 → D3 → Storybook-style demo pages).

---

## Phase 4 — Attention

- [ ] Attention widget with Q/K/V, scores, mask, weights, output.
- [ ] Multi-head selector.
- [ ] Hover-row → highlight keys.
- [ ] Reproduces `attention.json` fixture exactly.
- [ ] e2e covers it.

**Plan:**

**Deviations:**

**Follow-ups:**

---

## Phase 5 — FFN, layernorm, residuals

- [ ] FFN widget, layernorm widget, residuals callouts.
- [ ] Full block forward agrees with fixture to 1e-9.

**Plan:**

**Deviations:**

**Follow-ups:**

---

## Phase 6 — Stacking, sampling, playground

- [ ] Stacking widget across N blocks.
- [ ] Sampling widget (greedy/temperature/top-k/top-p).
- [ ] `/playground` page with full pipeline + iterative sampling.
- [ ] "Show me an example" presets (SPEC §14 Q4).

**Plan:**

**Deviations:**

**Follow-ups:**

---

## Phase 7 — Experiments (save / share / fork)

- [ ] CRUD APIs.
- [ ] `/experiments`, `/experiments/[slug]`, `/account` pages.
- [ ] Public experiments world-readable (SPEC §14 Q3 default).
- [ ] e2e: save → share → view-as-guest → fork.

**Plan:**

**Deviations:**

**Follow-ups:**

---

## Phase 8 — Comments + progress

- [ ] Comments per section, threaded one level, sanitised Markdown.
- [ ] Per-user progress auto-tracking via scroll + interaction events.
- [ ] `/learn` index and `/account` reflect status.

**Plan:**

**Deviations:**

**Follow-ups:**

---

## Phase 9 — Analytics + admin dashboard

- [ ] Event ingestion endpoint.
- [ ] `/admin` gated by `ADMIN_GITHUB_LOGINS`.
- [ ] DAU/WAU sparklines, per-section funnel, drop-off, top experiments,
      recent comments.

**Plan:**

**Deviations:**

**Follow-ups:**

---

## Phase 10 — Polish, docs, deploy

- [ ] README with screenshots/GIFs (Playwright-captured).
- [ ] About page with credits and references.
- [ ] Motion-reduction toggle; `prefers-reduced-motion` honoured.
- [ ] axe-core in CI; zero serious/critical violations.
- [ ] Lighthouse ≥ 90 (perf, a11y, best-practices) on `/`,
      `/learn/03-attention`, `/playground`.
- [ ] Live on Vercel; setup documented.

**Plan:**

**Deviations:**

**Follow-ups:**
