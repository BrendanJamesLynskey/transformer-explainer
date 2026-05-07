# CLAUDE.md — Transformer Decoder Explainer

This file tells Claude Code how to work in this repository. Read it fully before
starting any task. The companion `SPEC.md` describes **what** to build; this file
describes **how** to build it.

---

## 1. Project mission

Build a **graphical, interactive web-based explainer of the Transformer decoder**,
designed for learners. The reader should be able to feed in their own tokens and
watch every operation — embeddings, masked self-attention, FFN, layernorm,
residuals, multi-block stacking, and final next-token sampling — execute on the
server, with the results visualised step-by-step in the browser.

The codebase is itself a **teaching artefact**. Code clarity is a first-class
requirement, on a par with correctness. A motivated learner reading the source
should be able to understand it without reaching for external references.

---

## 2. Audience for the code

Two readers, both important:

1. **The end user** — a learner who interacts with the deployed site.
2. **The code reader** — a learner (or the project owner) reading the source on
   GitHub to understand both the maths _and_ a real-world Next.js full-stack
   codebase.

Optimise for both. Where they conflict, prefer (2): a slightly slower or more
verbose implementation that teaches well beats a clever one-liner.

---

## 3. Tech stack (locked-in)

Do not substitute these without explicit approval.

| Layer         | Choice                                                  |
| ------------- | ------------------------------------------------------- |
| Framework     | Next.js 14 (App Router) + TypeScript (strict)           |
| Styling       | Tailwind CSS                                            |
| Database      | Postgres on Neon (free tier)                            |
| ORM           | Drizzle ORM                                             |
| Auth          | Auth.js (NextAuth v5) with GitHub OAuth only            |
| Content       | MDX files in `/content`, rendered via `next-mdx-remote` |
| Maths runtime | Pure TypeScript — no PyTorch, no ONNX, no WASM          |
| Visualisation | D3.js (with React refs; no `react-d3` wrappers)         |
| Testing       | Vitest (unit), Playwright (e2e)                         |
| Lint/format   | ESLint + Prettier (Next.js defaults + Tailwind plugin)  |
| Deployment    | Vercel free tier; Neon free tier; GitHub Actions CI     |

---

## 4. Repository layout

```
.
├── CLAUDE.md                     ← this file
├── SPEC.md                       ← what to build
├── README.md                     ← user-facing overview, generated last
├── PROGRESS.md                   ← living checklist; update as phases complete
├── package.json
├── tsconfig.json
├── next.config.mjs
├── tailwind.config.ts
├── drizzle.config.ts
├── .env.example                  ← committed; never commit .env.local
├── .github/
│   └── workflows/
│       └── ci.yml                ← lint, typecheck, test on every PR
├── content/
│   └── decoder/
│       ├── 01-overview.mdx
│       ├── 02-embeddings.mdx
│       ├── 03-attention.mdx
│       ├── 04-ffn.mdx
│       ├── 05-layernorm-residuals.mdx
│       ├── 06-stacking.mdx
│       └── 07-sampling.mdx
├── src/
│   ├── app/                      ← Next.js App Router
│   │   ├── (marketing)/          ← public landing page
│   │   ├── learn/[slug]/         ← MDX-rendered explainer pages
│   │   ├── experiments/          ← saved-experiment views
│   │   ├── admin/                ← analytics dashboard (auth-gated)
│   │   ├── api/
│   │   │   ├── compute/          ← server-side maths endpoints
│   │   │   ├── experiments/      ← CRUD for saved experiments
│   │   │   ├── progress/         ← per-section progress tracking
│   │   │   ├── events/           ← analytics ingestion
│   │   │   └── auth/             ← Auth.js handlers
│   │   ├── layout.tsx
│   │   └── page.tsx
│   ├── components/
│   │   ├── ui/                   ← generic reusable bits (Button, Card, …)
│   │   ├── viz/                  ← D3 visualisations (one file per concept)
│   │   └── interactive/          ← interactive widgets used inside MDX
│   ├── lib/
│   │   ├── transformer/          ← THE maths — see §6
│   │   ├── db/                   ← Drizzle schema + client
│   │   ├── auth/                 ← Auth.js config
│   │   ├── analytics/            ← event tracking helpers
│   │   └── mdx/                  ← MDX loading, components map
│   └── types/                    ← shared TS types
├── tests/
│   ├── unit/                     ← Vitest for lib/transformer + lib/*
│   └── e2e/                      ← Playwright user-journey tests
└── scripts/
    ├── seed-db.ts                ← seed sample experiments + admin user
    └── verify-maths.ts           ← cross-check our ops against reference values
```

---

## 5. Coding conventions

### TypeScript

- `strict: true`. No `any`. If you genuinely need an escape hatch, use `unknown`
  and narrow it.
- Prefer `type` aliases for data shapes; `interface` only for things that are
  meant to be extended (rare here).
- All exported functions and components have JSDoc with at least a one-line
  summary. For `lib/transformer/*`, JSDoc is **mandatory and detailed**:
  include shapes (e.g. `@param x  Input tensor of shape [seq_len, d_model]`),
  the formula being computed, and a citation (paper section or standard ref).
- No barrel `index.ts` files re-exporting everything; learners benefit from
  explicit import paths.

### React / Next.js

- App Router only. Server Components by default; mark `"use client"` explicitly
  and only when needed (interactive widgets, D3 viz, anything using hooks).
- Data fetching for pages: server-side via Drizzle directly. No client-side
  fetching for first paint.
- Client-side fetching uses `fetch` against `/api/*` routes; no SWR/React Query
  for v1 (one fewer dependency to learn).
- Component files: one component per file, named after the file. Co-locate the
  D3 chart logic in the same file as its React wrapper, separated by a
  `// ─── d3 logic ──────────────` banner comment.

### D3

- Use D3 for **maths and layout** (scales, axes, paths, transitions).
- Use React for **structure** (which `<g>` exists, keyed by data).
- Pattern: render SVG structure in JSX; in a `useEffect`, select the ref and
  apply D3 transitions/updates. Never have D3 create or remove DOM nodes that
  React owns.
- Each viz component must have a self-contained Storybook-style demo page under
  `/learn/_demos/<viz-name>` (dev-only, gated by `NODE_ENV !== 'production'`)
  for visual regression checks.

### CSS

- Tailwind utility classes inline. No CSS modules, no styled-components.
- For repeated patterns (e.g. matrix-cell), define a Tailwind `@layer
components` class in `globals.css` rather than copy-pasting class strings.
- Dark mode: use Tailwind's `dark:` variant; default is system preference.

### Comments

- **Why, not what.** A loop counter doesn't need a comment; a non-obvious
  numerical-stability trick does.
- Any maths code references the equation it implements, e.g.
  `// softmax: y_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))`.
- Every file in `lib/transformer/` opens with a header comment giving (a) the
  operation, (b) the input/output shapes, (c) a one-line intuition, (d) a
  pointer to the MDX section that explains it.

### Errors

- API routes return shape `{ ok: true, data } | { ok: false, error }`.
  Never throw across the network boundary.
- Validate every API input with Zod schemas. Zod schemas live next to the
  route handler that uses them.
- Client-side: surface errors in the UI with a small toast component, never
  `alert()`, never silent.

---

## 6. The maths layer (`src/lib/transformer/`)

This is the heart of the project. Treat it as a small, well-documented teaching
library.

### File-per-operation rule

Each fundamental op gets its own file. Suggested decomposition:

```
lib/transformer/
├── types.ts              ← Tensor type alias, shape helpers
├── tensor.ts             ← create, fill, clone, shape utilities
├── matmul.ts             ← 2-D matrix multiply (naive triple loop)
├── softmax.ts            ← numerically-stable softmax with mask support
├── layernorm.ts          ← layer normalisation with γ, β
├── gelu.ts               ← GELU (and ReLU as a comparison point)
├── embeddings.ts         ← token + positional embedding lookup
├── attention.ts          ← single-head and multi-head causal attention
├── ffn.ts                ← position-wise feed-forward
├── block.ts              ← one decoder block composing the above
├── model.ts              ← stacked blocks + final projection
├── sampling.ts           ← greedy, temperature, top-k, top-p
├── tokenizer.ts          ← tiny BPE-ish or char tokenizer (see SPEC §4)
└── trace.ts              ← Trace type + helpers — see below
```

### Tensor representation

Use `number[][]` for matrices and `number[]` for vectors. **Do not** introduce a
class hierarchy or a fancy `Tensor` wrapper. The point is for the maths to be
visible. Performance is not a goal — keep inputs small (seq_len ≤ 16,
d_model ≤ 64 by default) and let the user see every multiply.

### Trace objects

Every op accepts an optional `trace?: Trace` argument. If present, the op
records intermediate tensors into it (e.g. attention writes Q, K, V, scores,
mask, softmax-weights, output). The API endpoints return the full trace; the
client uses it to drive the visualisations. This means **the visualisation is
not a separate model — it shows the actual computed values**.

### Verification

`scripts/verify-maths.ts` runs each op on a fixed seed and compares against
hand-computed reference values committed in `tests/unit/fixtures/`. This runs in
CI. If you change an op, regenerate the fixture deliberately and explain why in
the commit message.

---

## 7. Database & migrations

- Schema lives in `src/lib/db/schema.ts`.
- Migrations are generated with `drizzle-kit generate` and committed to
  `drizzle/` — never hand-edit a migration after it has been applied to a
  shared environment.
- `scripts/seed-db.ts` is idempotent: safe to re-run.
- Local dev uses a Neon dev branch (free) — document the setup in the README.

---

## 8. Testing strategy

| Layer                    | Tool       | What we test                                          |
| ------------------------ | ---------- | ----------------------------------------------------- |
| `lib/transformer/*`      | Vitest     | Numerical correctness against known reference values  |
| `lib/db/*`, `lib/auth/*` | Vitest     | Logic with a test Postgres (or pglite for unit speed) |
| API routes               | Vitest     | Request/response shape, auth, validation              |
| Critical user journeys   | Playwright | Sign in, run experiment, save, view as guest, admin   |

**Coverage targets** (enforced in CI):

- `lib/transformer/`: 100% line coverage. No exceptions.
- Other `lib/`: 80%.
- API routes: every route has at least one happy-path and one auth/validation
  failure test.

Playwright runs against a real `next dev` server with a seeded test database.

---

## 9. Workflow — phase-by-phase

**Build in the order given by `SPEC.md` §10 (Implementation Phases).** Do not
skip ahead. After each phase:

1. All tests pass; coverage targets met for any new `lib/transformer/` code.
2. `pnpm lint && pnpm typecheck` clean.
3. Update `PROGRESS.md` — tick the phase, note any deviations, list any
   follow-ups.
4. Commit with a message of the form
   `phase(N): <short summary>` (e.g. `phase(3): masked attention with trace`).
5. **Stop and ask** before starting the next phase if any of:
   - You had to deviate from `SPEC.md` materially.
   - A dependency choice came up that isn't covered here or in `SPEC.md`.
   - A test had to be skipped.

### Plan/act mode

Use plan mode at the start of each phase: write a short plan to
`PROGRESS.md` under that phase's heading (files you'll create/edit, key
decisions), then act. Do not produce a giant unreviewable diff.

### Subagents

Use the Opus-orchestrator / Sonnet-executor pattern that the project owner is
familiar with:

- **Orchestrator** owns the plan and reviews each subagent's diff.
- **Executor subagents** are dispatched per file or per small file-cluster.
- A subagent's brief always includes (a) the file(s) it owns, (b) the relevant
  section of `SPEC.md` and `CLAUDE.md`, (c) the acceptance test it must pass.

---

## 10. Environment variables

`.env.example` lists every variable. Key ones:

```
# Database
DATABASE_URL=postgres://...

# Auth.js
AUTH_SECRET=                # `openssl rand -base64 32`
AUTH_GITHUB_ID=
AUTH_GITHUB_SECRET=

# App
NEXT_PUBLIC_SITE_URL=http://localhost:3000
ADMIN_GITHUB_LOGINS=BrendanJamesLynskey   # comma-separated GH usernames

# Compute limits (server-side enforcement)
MAX_SEQ_LEN=16
MAX_D_MODEL=64
MAX_BLOCKS=4
```

Never read `process.env` outside `src/lib/env.ts`, which validates with Zod at
boot and exports a typed `env` object.

---

## 11. Definition of done (whole project)

- All phases in `SPEC.md` §10 complete and ticked in `PROGRESS.md`.
- `pnpm build` succeeds locally.
- CI green on `main` (lint, typecheck, unit, e2e).
- Site deploys to Vercel from `main` with no manual steps beyond setting env
  vars and connecting Neon — documented in `README.md`.
- README covers: what it is, screenshots/GIFs, local setup, deploy guide,
  contributing, licence (MIT).
- `verify-maths.ts` agrees with a small reference PyTorch script
  (`scripts/reference.py`, run manually, output committed to
  `tests/unit/fixtures/`) to within 1e-5.
