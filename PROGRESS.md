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

- [ ] Skeleton boots; `pnpm dev`, `pnpm lint`, `pnpm typecheck`, `pnpm test`
      all clean.
- [ ] CI green on first PR.

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

- [ ] All ops implemented in `src/lib/transformer/`.
- [ ] 100% line coverage on `src/lib/transformer/`.
- [ ] `pnpm verify:maths` passes against committed fixtures.

**Plan:**

**Deviations:**

**Follow-ups:**

---

## Phase 2 — DB + Auth

- [ ] Drizzle schema + initial migration.
- [ ] Auth.js with GitHub OAuth working locally.
- [ ] `pnpm db:seed` idempotent.
- [ ] Auth e2e test passes.

**Plan:**

**Deviations:**

**Follow-ups:**

---

## Phase 3 — MDX + Overview & Embeddings

- [ ] MDX pipeline working with components map.
- [ ] `/learn` index and `/learn/[slug]` rendering.
- [ ] Three-layer toggle (Concept/Maths/Code) functional.
- [ ] Embedding widget hits `/api/compute/embed` and renders.
- [ ] e2e: navigate sections, toggle layers, interact with widget.

**Plan:**

**Deviations:**

**Follow-ups:**

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
