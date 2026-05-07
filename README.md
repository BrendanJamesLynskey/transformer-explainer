# Transformer Decoder Explainer

A graphical, interactive web explainer of the Transformer decoder. Type tokens,
watch every operation — embeddings, masked self-attention, FFN, layernorm,
residuals, multi-block stacking, sampling — execute on the server and visualise
step-by-step in the browser.

> **Status:** under construction. See `PROGRESS.md` for current phase.
> The full plan lives in `SPEC.md`; conventions in `CLAUDE.md`.

## Stack

Next.js 14 (App Router) · TypeScript (strict) · Tailwind · Postgres (Neon)
· Drizzle ORM · Auth.js (GitHub OAuth) · MDX content · D3.js viz · pure-TS
transformer maths · Vitest · Playwright.

## Quick start

```bash
git clone <repo>
cd transformer-explainer
pnpm install
cp .env.example .env.local        # fill in DATABASE_URL, AUTH_*
pnpm db:push                      # apply schema to your Neon dev branch
pnpm db:seed                      # idempotent seed
pnpm dev                          # http://localhost:3000
```

See the README's Setup section (filled in by Phase 10) for full instructions
including GitHub OAuth app creation and Vercel deployment.

## Development

```bash
pnpm lint                # ESLint + Tailwind plugin
pnpm typecheck           # tsc --noEmit
pnpm test                # Vitest unit tests
pnpm test:coverage       # with coverage thresholds enforced
pnpm test:e2e            # Playwright (requires running app)
pnpm verify:maths        # cross-check TS ops against PyTorch fixtures
```

## Regenerating numerical fixtures

The TypeScript transformer implementation is verified against PyTorch
fixtures committed in `tests/unit/fixtures/`. To regenerate:

```bash
pip install torch numpy
python scripts/reference.py
git add tests/unit/fixtures/ && git commit -m "fixtures: regenerate"
```

## Licence

MIT — see `LICENSE`.
