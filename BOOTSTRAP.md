# BOOTSTRAP.md — How Claude Code should start

**Read this file first when you begin work in this repo.** Then read, in order:

1. `CLAUDE.md` — workflow and conventions (how to build).
2. `SPEC.md` — product spec (what to build).
3. `PROGRESS.md` — phase tracker (where you are).
4. `RUNBOOK.md` — how to handle ambiguity, blockers, and decisions when the
   human is asleep.

After reading, follow the procedure below. **Do not skip to coding.**

---

## Initial procedure (run once, at the start of the overnight session)

### Step 0 — Verify the human's pre-flight setup

The human is expected to have completed a small pre-flight setup before
kicking off this run. Verify these are in place; the rest of the procedure
assumes they are.

**(a)** A populated `.env.local` in the project root with:

```bash
test -f .env.local || echo "MISSING: .env.local"
grep -q "^DATABASE_URL=postgres://" .env.local || echo "MISSING: DATABASE_URL"
grep -q "^AUTH_SECRET=." .env.local || echo "MISSING: AUTH_SECRET"
grep -q "^AUTH_GITHUB_ID=." .env.local || echo "MISSING: AUTH_GITHUB_ID"
grep -q "^AUTH_GITHUB_SECRET=." .env.local || echo "MISSING: AUTH_GITHUB_SECRET"
```

**(b)** Authenticated CLIs:

```bash
gh auth status      # expect logged in
vercel whoami       # expect a username (skip if vercel not installed; OK)
```

**(c)** The Neon database is reachable:

```bash
# Quick connectivity check using the URL from .env.local
( set -a; source .env.local; set +a; \
  pnpm dlx postgres "$DATABASE_URL" -c "select 1 as ok;" 2>/dev/null \
  || echo "WARN: cannot reach Neon DB; will retry in Phase 2" )
```

If any **MISSING** values appear from (a), or `gh auth status` fails, stop
and surface the issue clearly to the human. These are one-time setup steps
(see `RUNBOOK.md → §3 → Blocker: GitHub OAuth not configured`).

If `vercel whoami` fails, that's fine for now — note it and continue. The
deploy step in Phase 10 will handle a missing Vercel CLI gracefully.

If the Neon connectivity check fails but `.env.local` looks well-formed,
continue — the network may be flaky. Phase 2 will retry properly with full
error reporting.

### Step 1 — Sanity check the environment

```bash
node --version    # expect >= 20.11
pnpm --version    # expect >= 9
git status        # expect either a clean working tree, or only .env.local untracked
```

If any of these fail, stop and write a note to `PROGRESS.md` under "Phase 0 →
Plan" describing what you'd need.

### Step 2 — Install and verify the skeleton

```bash
pnpm install
pnpm format:check
pnpm lint
pnpm typecheck
pnpm test
```

The repository has been pre-seeded with config, `package.json`, a smoke test,
and CI. All five commands above must pass before you start writing features.
Fix anything that doesn't, then commit:
`chore: bootstrap clean (passes lint/typecheck/test)`.

### Step 3 — Generate the numerical fixtures

The Python reference script `scripts/reference.py` produces the fixtures that
the TS implementation is verified against. **You must generate these before
phase 1.** They are not pre-committed because they would need to be regenerated
from the Python source anyway; doing it yourself makes the dependency
explicit.

```bash
python3 -m venv .venv-fixtures
source .venv-fixtures/bin/activate
pip install --quiet torch numpy
python scripts/reference.py
deactivate
git add tests/unit/fixtures/
git commit -m "fixtures: generate from scripts/reference.py"
```

If `pip install torch` fails (e.g. offline or restricted environment), see
`RUNBOOK.md → Blocker: cannot install PyTorch`. Do not proceed to phase 1
without fixtures — phase 1's acceptance test depends on them.

### Step 4 — Initialise git and create the GitHub repo

You have access to the `gh` CLI and authenticated GitHub access. **Create the
remote repo yourself** so the human doesn't need to do it before kicking off
the run.

First, confirm git and `gh` are available and authenticated:

```bash
git --version
gh --version
gh auth status
```

If `gh auth status` reports not logged in, stop and surface this — there's no
defensible workaround.

If the working directory is not yet a git repo (no `.git/`), initialise it
and commit the bootstrap files:

```bash
git init -b main
git add .
git commit -m "chore: bootstrap from starter bundle

Includes CLAUDE.md, SPEC.md, BOOTSTRAP.md, RUNBOOK.md, PROGRESS.md, full
config (Next.js/TS/Tailwind/Drizzle/Auth.js/Vitest/Playwright), CI workflow,
PyTorch reference fixture generator, and verify-maths script."
```

Then create the remote repo. Use these settings:

- **Name:** `transformer-explainer` (matches the directory name)
- **Visibility:** public — this is a portfolio piece
- **Description:** "A graphical, interactive explainer of the Transformer decoder. Built as a learning project."
- **No** README/license/gitignore from GitHub's templates — the bundle has them.

```bash
gh repo create transformer-explainer \
  --public \
  --description "A graphical, interactive explainer of the Transformer decoder. Built as a learning project." \
  --source=. \
  --remote=origin \
  --push
```

That single command creates the remote, sets `origin`, and pushes `main` in
one go. CI will run on this first push — let it. If CI fails on the bootstrap
commit, fix the failure before proceeding to Phase 0 features.

If `gh repo create` fails because the repo already exists under your account
(e.g. you're re-running after a partial run):

```bash
gh repo view BrendanJamesLynskey/transformer-explainer >/dev/null 2>&1 \
  && git remote add origin https://github.com/BrendanJamesLynskey/transformer-explainer.git \
  && git push -u origin main
```

For all subsequent pushes during the overnight run, use a feature branch per
phase and open a PR — see `RUNBOOK.md §4 (Commit hygiene)` and `§5 (PR
description template)`. Phases merge into `main` only after CI is green.

Set up the per-phase branching pattern now:

```bash
git config --local pull.rebase true
git config --local rebase.autoStash true
```

#### Repo settings to apply via `gh`

After the repo exists, configure the basics. **Do not enable strict branch
protection** — Claude Code is the only "developer" during this run and would
deadlock against a "review required" rule. The human can tighten things up
after the run.

```bash
gh repo edit --enable-issues --enable-wiki=false --delete-branch-on-merge
```

Skip `gh api branches/main/protection` for now; add a follow-up to
`PROGRESS.md → Phase 0 → Follow-ups`:

```
[ ] Human: enable branch protection on `main` after the run completes:
    require status checks, require linear history, require PR review.
    See gh api docs for the full payload.
```

Confirm everything is wired up:

```bash
git remote -v        # expect origin pointing at the new repo
gh repo view --web   # only run if you want to open it; safe to skip in headless mode
gh run list --limit 1  # expect a CI run in progress or completed
```

### Step 5 — Begin Phase 0

Read `SPEC.md §10` Phase 0 acceptance criteria and `PROGRESS.md`'s Phase 0
section. Most of Phase 0 is already done (config, CI, smoke test). What
remains:

- Create `src/lib/env.ts` with Zod-validated env access.
- Create the placeholder `src/app/layout.tsx` and `src/app/page.tsx`
  (a minimal landing page is enough — Phase 10 polishes it).
- Confirm `pnpm dev` boots and serves the placeholder.
- Commit: `phase(0): foundations skeleton`.
- Tick the box in `PROGRESS.md`.

Then proceed to Phase 1.

---

## Per-phase loop (every phase)

Repeat for phases 1 through 10. Each phase is a feature branch + PR.

1. **Branch.** From a clean `main`:
   ```bash
   git checkout main && git pull --ff-only
   git checkout -b phase/N-<short-name>
   ```
2. **Plan.** Read the matching section of `SPEC.md`. Write a short plan into
   the phase's "Plan" subsection in `PROGRESS.md` — files you'll create or
   edit, key decisions, the acceptance test you'll satisfy.
3. **Act.** Implement. Use the orchestrator + executor-subagent pattern from
   `CLAUDE.md §9`.
4. **Verify locally.** `pnpm format:check && pnpm lint && pnpm typecheck &&
pnpm test:coverage && pnpm verify:maths` (verify:maths only after phase 1).
5. **e2e** (where applicable per phase). `pnpm test:e2e`.
6. **Update.** Tick the phase box in `PROGRESS.md`. Fill "Deviations" and
   "Follow-ups" honestly even if empty.
7. **Commit.** `phase(N): <short summary>`.
8. **Push and open a PR.**
   ```bash
   git push -u origin phase/N-<short-name>
   gh pr create \
     --title "phase(N): <short summary>" \
     --body-file <(printf "%s\n" "<filled-in PR body — see RUNBOOK.md §5>") \
     --base main
   ```
9. **Wait for CI green.** Poll with `gh pr checks --watch`. If CI fails, fix
   on the same branch (don't open a new PR).
10. **Self-merge.** Once CI is green:
    ```bash
    gh pr merge --squash --delete-branch
    git checkout main && git pull --ff-only
    ```
    Self-merging is appropriate here because this is a solo project and the
    human will review the merged history in the morning. Use squash so each
    phase becomes a single tidy commit on `main`.

If anything blocks you, consult `RUNBOOK.md` rather than the human. The human
is asleep.

---

## What "done" looks like

When you reach the bottom of `PROGRESS.md` with every box ticked and CI green
on `main`:

- Make sure the README (Phase 10) walks a stranger through local setup and
  Vercel deploy.
- Attempt the Vercel deployment yourself if the Vercel CLI is available and
  authenticated (`vercel whoami` succeeds). See `RUNBOOK.md §3 → Deployment`.
  If it isn't, leave a clear "human, do this" block in the final summary
  with the exact `vercel` commands to run.
- Print a final summary to the conversation log: phases completed, total
  commits, list of merged PR numbers, any items collected from
  `PROGRESS.md → Follow-ups` that need human review, and the deploy status.
- Stop. Do not continue extrapolating new features.

---

## Things only the human can do

Some setup steps need a human in front of github.com or vercel.com because
they require interactive login flows that the GitHub API cannot bypass. If
any of these are required for a phase and haven't been done, **don't try to
fake them** — implement everything else for that phase, write a clear
follow-up note, and continue.

| Action                                   | Why human-only                                                                                                    | Where it's needed  |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------ |
| Create a GitHub OAuth app + secrets      | The OAuth-app create endpoint is not in the GitHub API; only the web UI offers it.                                | Phase 2 (auth).    |
| Add OAuth credentials as repo secrets    | You can do this via `gh secret set` once the human has the values.                                                | Phase 2 onward.    |
| Create a Neon project + get DATABASE_URL | Neon's public API requires a Neon-issued token, not a GitHub one.                                                 | Phase 2 onward.    |
| Connect the GitHub repo to Vercel        | Vercel's project-create flow can be done via `vercel link --yes` but the first-ever Vercel login needs a browser. | Phase 10 (deploy). |
| Set production env vars in Vercel        | `vercel env add` works once `vercel` is logged in.                                                                | Phase 10.          |

For each of these, the workflow is: implement everything that doesn't depend
on the secret; mock or stub the dependency in tests; record a "needs human"
follow-up in `PROGRESS.md`. The human will run a small batch of these in the
morning, then re-run any affected phase's verification.
