# SPEC.md — Transformer Decoder Explainer

The product spec. Read `CLAUDE.md` first for conventions and workflow.

---

## 1. One-line description

A graphical, interactive web explainer of the Transformer decoder, where users
type tokens and watch every operation — embeddings, masked self-attention, FFN,
layernorm, residuals, multi-block stacking, sampling — execute on the server and
visualise step-by-step in the browser.

---

## 2. Goals & non-goals

### Goals

1. Teach the Transformer decoder rigorously across **three layers**:
   conceptual, mathematical, and implementation. Users toggle layers per
   section.
2. Let users supply their own input and **see the actual numbers** flow through.
3. Be a deployable, real-world full-stack app (auth, DB, dynamic content,
   analytics, admin) — i.e. a portfolio-grade artefact.
4. Be its own best teaching example: a learner reading the source on GitHub
   should learn Next.js full-stack patterns alongside Transformer internals.

### Non-goals

- Performance / scale. The maths runs in pure TS. Inputs are tiny.
- Training. We only do forward passes with hand-set or seeded weights.
- Tokeniser fidelity. A toy char-level or tiny-BPE tokeniser is fine.
- Mobile-first design. Desktop / tablet first; mobile graceful degradation.
- Internationalisation. English only.

---

## 3. User personas

- **Curious learner** — knows some Python and basic linear algebra; wants to
  understand attention beyond hand-wavy blog posts.
- **Practitioner refresher** — has used Transformers but wants to revisit the
  internals visually.
- **Code reader** — clones the repo and reads it to learn full-stack Next.js.
- **Admin (project owner)** — adds new explainers, monitors usage.

---

## 4. Core domain model — the toy Transformer

Defaults (configurable up to the limits in `.env`):

| Param        | Default | Max |
| ------------ | ------- | --- |
| `seq_len`    | 8       | 16  |
| `d_model`    | 16      | 64  |
| `n_heads`    | 2       | 4   |
| `d_ff`       | 32      | 128 |
| `n_blocks`   | 2       | 4   |
| `vocab_size` | 64      | 128 |

Weights: deterministic from a seed (default 42). The user can change the seed.
We provide 2-3 named "characters" — fixed seeds with curated qualitative
behaviour (e.g. one that makes attention focus on the most recent token, one
that spreads attention).

Tokeniser: char-level over a 64-character alphabet (lowercase letters, digits,
basic punctuation, space). Documented as a deliberate simplification.

---

## 5. Information architecture

```
/                       Landing page — what it is, "Try it now" CTA
/learn                  Index of explainer sections
/learn/[slug]           Individual explainer (MDX + interactive widgets)
/playground             Free-form playground: full pipeline, all controls
/experiments            Browse public experiments
/experiments/[slug]     View a saved experiment (read-only if not owner)
/account                Signed-in user's experiments, progress, notes
/admin                  Analytics dashboard (auth + admin-list gated)
/about                  Credits, references, methodology
```

### Explainer sections (order matters — pedagogical progression)

1. `01-overview` — What a decoder does; the big picture; tokens in, logits out.
2. `02-embeddings` — Token embedding lookup; positional encoding (sinusoidal).
3. `03-attention` — The headline act: Q/K/V, scores, mask, softmax, output.
   Single-head first, then multi-head.
4. `04-ffn` — Position-wise feed-forward; GELU.
5. `05-layernorm-residuals` — Why residuals; why layernorm; pre-norm vs post-norm.
6. `06-stacking` — Stacking blocks; what changes layer-to-layer.
7. `07-sampling` — From logits to next token; greedy / temperature / top-k / top-p.

---

## 6. Feature catalogue

### 6.1 Three-layer toggle

Every explainer page has a layer selector (sticky, top-right):

- **Concept** — animations, intuitions, no maths shown.
- **Maths** — show the actual matrices, dot products, softmax row-by-row.
- **Code** — show the corresponding TypeScript snippet from
  `lib/transformer/`, plus optional PyTorch and the user's RTL reference for
  comparison (`@brendan/rtl-comparison` info boxes).

Layers are independently togglable; the user can have all three on. Default:
Concept on, Maths off, Code off.

### 6.2 Interactive widgets (MDX-embedded)

Each section embeds one or more interactive widgets that share the same
underlying compute path as the playground.

- **Embedding lookup widget** — type a string; see each char map to a token id
  and then a vector; visualise the vectors as small heatmaps.
- **Attention widget** — type ≤16 chars; pick a head; see Q/K/V matrices, the
  raw score grid, the causal mask overlay, the softmax-normalised weights, and
  the output vectors. Hover a query row to highlight the keys it attends to.
- **FFN widget** — show input vector → up-projection → GELU → down-projection,
  with bar charts of activations at each stage.
- **Layernorm widget** — slider for γ and β; see how the distribution changes.
- **Stacking widget** — show the same input passing through N blocks; per-block
  attention pattern grid.
- **Sampling widget** — show final logits as a bar chart; controls for
  temperature, top-k, top-p; clicking "sample" picks a token and appends it,
  iterating up to a chosen length.

### 6.3 Playground (`/playground`)

Everything in one page: input box, full hyperparameter panel, all visualisations
stacked, with the option to "Save as experiment" (named, optionally public).

### 6.4 Experiments

- **Save**: persists `{ input, config, seed, layout_state, name, visibility }`.
- **Share**: every experiment has a slug-based URL; public experiments are
  listed at `/experiments`.
- **Fork**: anyone (signed in) can fork a public experiment to their account.
- **Comments**: per-section commenting on `/learn/[slug]` pages. Threaded one
  level deep. Markdown with sanitisation. Owner can pin or hide.

### 6.5 Progress tracking

Per-user, per-section: not started / in progress / completed (auto-marked when
the user has interacted with every widget on the page and scrolled past 80%).
Visible on `/learn` index and `/account`.

### 6.6 Admin analytics dashboard (`/admin`)

Gated by `ADMIN_GITHUB_LOGINS` env. Shows:

- DAU/WAU/MAU sparklines.
- Per-section funnel: visits → 80% scroll → all widgets interacted →
  marked complete.
- Drop-off heatmap by section.
- Top public experiments by views.
- Recent comments queue (with hide action).

All metrics computed from rows in the `events` table — no third-party
analytics service.

---

## 7. API surface

All under `/api`. Validate inputs with Zod. Return `{ ok, data | error }`.

| Method | Path                            | Purpose                                |
| ------ | ------------------------------- | -------------------------------------- |
| POST   | `/api/compute/embed`            | Tokens → embeddings + positional       |
| POST   | `/api/compute/attention`        | Run masked multi-head attention; trace |
| POST   | `/api/compute/ffn`              | Run FFN; trace                         |
| POST   | `/api/compute/block`            | Run one full block; trace              |
| POST   | `/api/compute/forward`          | Full forward pass; trace               |
| POST   | `/api/compute/sample`           | Logits → next token (modes)            |
| GET    | `/api/experiments`              | List public experiments (paginated)    |
| POST   | `/api/experiments`              | Create (auth)                          |
| GET    | `/api/experiments/[slug]`       | Read                                   |
| PATCH  | `/api/experiments/[slug]`       | Update (auth, owner)                   |
| DELETE | `/api/experiments/[slug]`       | Delete (auth, owner)                   |
| POST   | `/api/experiments/[slug]/fork`  | Fork to current user (auth)            |
| GET    | `/api/sections/[slug]/comments` | List                                   |
| POST   | `/api/sections/[slug]/comments` | Create (auth)                          |
| PATCH  | `/api/comments/[id]`            | Edit (auth, author) / hide (admin)     |
| GET    | `/api/progress`                 | Current user's progress map (auth)     |
| POST   | `/api/progress`                 | Upsert section progress (auth)         |
| POST   | `/api/events`                   | Ingest analytics event                 |
| GET    | `/api/admin/metrics`            | Admin dashboard data (auth, admin)     |

Compute endpoints enforce `MAX_SEQ_LEN`, `MAX_D_MODEL`, `MAX_BLOCKS` from env.

---

## 8. Database schema (Drizzle / Postgres)

```ts
users {
  id              uuid pk
  github_id       text unique
  github_login    text
  display_name    text
  avatar_url      text
  created_at      timestamptz
}

sessions { …Auth.js standard… }
accounts { …Auth.js standard… }

experiments {
  id              uuid pk
  slug            text unique               // human-readable + nanoid suffix
  owner_id        uuid fk → users.id
  name            text
  visibility      text check in ('public','unlisted','private')
  input_text      text
  config_json     jsonb                     // hyperparams, seed
  layout_json     jsonb                     // which panels open, layer toggles
  forked_from     uuid null fk → experiments.id
  view_count      int default 0
  created_at      timestamptz
  updated_at      timestamptz
  index (owner_id), index (visibility, created_at)
}

comments {
  id              uuid pk
  section_slug    text                      // e.g. '03-attention'
  user_id         uuid fk → users.id
  parent_id       uuid null fk → comments.id  // 1-level threading
  body_md         text
  hidden          boolean default false
  created_at      timestamptz
  index (section_slug, created_at)
}

progress {
  user_id         uuid fk → users.id
  section_slug    text
  status          text check in ('not_started','in_progress','completed')
  updated_at      timestamptz
  pk (user_id, section_slug)
}

events {                                    // analytics
  id              bigserial pk
  user_id         uuid null fk → users.id   // null = anonymous
  session_id      text                      // cookie-scoped
  kind            text                      // 'page_view','widget_interact',…
  section_slug    text null
  meta_json       jsonb
  created_at      timestamptz
  index (created_at), index (kind, created_at)
}
```

---

## 9. Visual design

- Type: Inter (UI), JetBrains Mono (code & matrices).
- Palette: neutral greys + a single accent (project owner's pick — default
  `indigo-600`). Heatmaps use D3 `interpolateViridis`.
- Light / dark mode both first-class.
- Density: matrices are the star — give them room. Avoid decorative gradients.
- Motion: every widget update is a transition (200–400ms ease). The user can
  disable motion in a settings dropdown (respects `prefers-reduced-motion`).

---

## 10. Implementation phases

Each phase is a complete, shippable increment. Tests pass and CI is green at
the end of every phase. Update `PROGRESS.md` after each.

### Phase 0 — Foundations (skeleton, no features)

Files: `package.json`, `tsconfig.json`, `next.config.mjs`, `tailwind.config.ts`,
`drizzle.config.ts`, `.env.example`, `.github/workflows/ci.yml`,
`src/app/layout.tsx`, `src/app/page.tsx` (placeholder landing),
`src/lib/env.ts`.

Acceptance: `pnpm dev` boots; `pnpm lint && pnpm typecheck && pnpm test` all
clean (with one trivial smoke test); CI green on a placeholder PR.

### Phase 1 — Maths layer (offline)

Files: everything under `src/lib/transformer/`. No UI yet.

Acceptance: full unit test suite for matmul, softmax, layernorm, gelu,
embeddings, single-head attention, multi-head attention, ffn, block, model,
sampling, tokenizer. 100% line coverage. `verify-maths.ts` passes against
committed PyTorch reference fixtures.

### Phase 2 — DB + Auth

Files: `src/lib/db/schema.ts`, generated migrations, `src/lib/auth/*`,
`/api/auth/*`, `scripts/seed-db.ts`. Sign-in / sign-out UI on the layout.

Acceptance: GitHub OAuth works locally with a dev OAuth app; users row
created on first sign-in; `pnpm db:seed` is idempotent; auth e2e test passes.

### Phase 3 — MDX content + first explainer (Overview & Embeddings)

Files: `src/lib/mdx/*`, `src/components/interactive/Embedding*`,
`content/decoder/01-overview.mdx`, `content/decoder/02-embeddings.mdx`,
`/learn` index, `/learn/[slug]` page, three-layer toggle.

Acceptance: pages render; embedding widget shows real values from the
maths layer via `/api/compute/embed`; layer toggle works; e2e test reads
section 1 → section 2 → toggles layers → interacts with widget.

### Phase 4 — Attention (the headline)

Files: `src/components/viz/AttentionMatrix.tsx`,
`src/components/viz/AttentionHeads.tsx`, `src/components/interactive/AttentionWidget.tsx`,
`/api/compute/attention`, `content/decoder/03-attention.mdx`.

Acceptance: widget reproduces a known fixture exactly; hover interactions
work; multi-head selector works; mask is visibly applied; e2e test covers it.

### Phase 5 — FFN, layernorm, residuals

Files: matching widgets and MDX for sections 04 and 05, plus
`/api/compute/ffn`, `/api/compute/block`.

Acceptance: full block forward through the widget agrees with the
direct-call fixture to 1e-9.

### Phase 6 — Stacking + sampling + playground

Files: `content/decoder/06-stacking.mdx`, `content/decoder/07-sampling.mdx`,
`/api/compute/forward`, `/api/compute/sample`, `/playground` page.

Acceptance: end-to-end "type text → see all stages → sample next token"
works in the playground; iterative sampling appends and re-runs.

### Phase 7 — Experiments (save / share / fork)

Files: experiments CRUD APIs, `/experiments`, `/experiments/[slug]`,
`/account`. "Save as experiment" button on playground.

Acceptance: a signed-in user can save, share a public URL with a logged-out
viewer, fork someone else's experiment. e2e test covers the full lifecycle.

### Phase 8 — Comments + progress

Files: comments APIs and UI in MDX layout; progress APIs and tracking
hook driven by scroll + widget-interaction events.

Acceptance: comments post, edit, hide; progress auto-advances; `/learn`
and `/account` reflect it; e2e test covers it.

### Phase 9 — Analytics + admin dashboard

Files: `src/lib/analytics/*`, `/api/events`, `/api/admin/metrics`,
`/admin` page.

Acceptance: events flow on real interactions; admin page shows non-zero
charts after a scripted Playwright flow; non-admins get a 403.

### Phase 10 — Polish, docs, deploy

Files: README.md (with screenshots/GIFs captured by a Playwright script),
about page, motion-reduction toggle, dark mode pass, accessibility audit
fixes (axe-core in CI), Vercel deploy guide, post-deploy smoke-test script.

Acceptance: project is live on Vercel; README walks a stranger through
local setup and deploy in under 15 minutes; Lighthouse ≥ 90 on
performance, accessibility, best-practices for `/`, `/learn/03-attention`,
`/playground`.

---

## 11. Accessibility

- All interactive widgets keyboard-navigable; focus rings visible.
- Heatmaps include numeric tooltips and an alt text summary.
- Colour is never the sole conveyor of information (use shape/label too).
- `prefers-reduced-motion` honoured; manual override too.
- axe-core runs in Playwright; zero serious/critical violations gates the
  Phase 10 acceptance.

---

## 12. Security & abuse

- All compute endpoints enforce strict input bounds (Zod + env limits).
- Rate-limit `/api/compute/*` and `/api/events` per IP (simple in-memory
  token bucket on Vercel edge is fine for v1; document the trade-off).
- Comments: server-side Markdown rendering with a sanitiser; no raw HTML.
- CSRF: Auth.js defaults; same-site cookies.
- Never log PII; events store hashed IPs only if at all.

---

## 13. Out of scope (parking lot for v2)

- Encoder-decoder cross-attention (you noted v1 is decoder-only).
- Real BPE tokeniser.
- WebGPU/WASM acceleration.
- Trained weights (download + load a real tiny-GPT checkpoint).
- Email digest of new comments / followed experiments.
- Mobile-optimised layout for matrices.

---

## 14. Open questions for the project owner

These are things to decide before, or during, the relevant phase. Default
behaviour is given in brackets; flag in `PROGRESS.md` if you change.

1. Accent colour. [`indigo-600`]
2. Site name / domain. [`transformer-explainer.vercel.app` placeholder]
3. Should public experiments be world-readable without sign-in, or only
   listed but require sign-in to view? [world-readable]
4. Do we want a "Show me an example" button that fills the playground with
   one of the curated character-seeds? [yes — adds in Phase 6]
5. Licence. [MIT]
