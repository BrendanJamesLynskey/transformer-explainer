# RUNBOOK.md — Operating policy for unattended runs

This file tells Claude Code how to behave when something is ambiguous, broken,
or under-specified during an overnight run. The principle is:
**make a defensible, narrow choice, document it, and keep moving.** Surface
everything in PR descriptions and `PROGRESS.md` so the human can review in the
morning.

---

## 1. Decision-making policy

When you encounter a question not answered by `SPEC.md` or `CLAUDE.md`, follow
this hierarchy, in order:

1. **Smallest defensible choice.** Pick the option that adds the least
   surface area, least new dependency, least clever code. "Make it work,
   keep it boring."
2. **Reversibility wins.** If choice A is hard to undo and choice B is
   easy, pick B even if A looks marginally better.
3. **Mirror an existing pattern.** If the repo already does X somewhere,
   do X here too — even if you'd start fresh differently.
4. **Document the choice** in `PROGRESS.md` under the current phase's
   "Deviations" section, with one sentence of rationale.
5. **Never invent product scope.** If the question is "should we add Y?",
   the answer is "no" unless `SPEC.md` already includes Y.

### Example calls — settle these without asking

| Question                                   | Default                                                            |
| ------------------------------------------ | ------------------------------------------------------------------ |
| Where does this util go?                   | `src/lib/<closest-domain>/`                                        |
| Should this component be client or server? | Server, unless it uses hooks/D3/event handlers.                    |
| New dependency for a small problem?        | Solve it without one, even if more code.                           |
| Should a viz animate on first paint?       | Yes, 250ms ease-out, respect reduced motion.                       |
| Empty state for a list?                    | One short sentence, neutral grey, no illustration.                 |
| Error state for a failed fetch?            | Inline `<ErrorBox>` with retry; never `alert()`.                   |
| Loading state?                             | Skeleton with the same shape as the loaded content.                |
| Test missing for code I'm changing?        | Add it before the change. Don't skip.                              |
| Two ways to phrase a Zod schema?           | The shorter one. Don't refactor existing schemas.                  |
| Component prop count growing past ~6?      | Group related props into a single object prop.                     |
| MDX page needs a one-off styled element?   | Put it inline with Tailwind, don't make a new component.           |
| You spotted an unrelated bug?              | Note it in `PROGRESS.md` follow-ups; don't fix in the same commit. |

---

## 2. When to actually stop

Stop and leave a summary message **only** for these conditions:

- A required external service (Neon, GitHub OAuth, Vercel) is unreachable
  AND the work cannot proceed without it.
- A test failure indicates a fundamental conflict between `SPEC.md` and
  `CLAUDE.md` that no narrow choice resolves.
- A phase's acceptance criterion cannot be met within reason — e.g. the
  PyTorch reference and the TS implementation diverge by > 1e-3 and you've
  spent more than ~30 minutes investigating.
- You would need to make a change that affects multiple already-completed
  phases (architectural backtrack).

In all other cases, decide, document, move on. The human will rather review
a debatable decision in the morning than wake up to a stalled run.

---

## 3. Common blockers and how to resolve them

### Blocker: cannot install PyTorch

If `pip install torch` fails (offline / restricted index / disk space):

1. Try CPU-only wheel: `pip install --index-url https://download.pytorch.org/whl/cpu torch`.
2. If still failing, install `numpy` only and use the NumPy reference path:
   - Modify `scripts/reference.py` to use NumPy throughout (it already uses
     NumPy semantics; replace `torch.empty(...).normal_(...)` with
     `np.random.default_rng(seed).normal(0, 0.02, shape)`, etc).
   - Note this in `PROGRESS.md → Phase 0 Deviations`.
3. If neither works, generate fixtures manually using the TS implementation
   itself (self-consistency only, **lower confidence**), commit with the
   message `fixtures: bootstrap from TS (TODO: replace with PyTorch run)`,
   and add a follow-up to `PROGRESS.md`. The human will regenerate properly
   on first review.

### Blocker: GitHub OAuth not configured for local dev

The human is expected to have completed the pre-flight setup described in
`BOOTSTRAP.md → Step 0 (pre-flight)` before kicking off the run. That means
`AUTH_GITHUB_ID`, `AUTH_GITHUB_SECRET`, `AUTH_SECRET`, and `DATABASE_URL`
should all be present in `.env.local` (and inheritable by your shell). Check
this at the start of Phase 2:

```bash
test -n "$AUTH_GITHUB_ID" && test -n "$AUTH_GITHUB_SECRET" \
  && test -n "$AUTH_SECRET" && test -n "$DATABASE_URL" \
  && echo "auth env OK" || echo "auth env MISSING — see fallback below"
```

**Happy path** (env present): implement Phase 2 normally. The "GitHub OAuth
works locally" e2e test should run against the real provider — Playwright can
drive the GitHub OAuth flow, but it's brittle. Prefer this pattern: write the
e2e test using a local Auth.js Credentials provider that's gated on
`process.env.E2E_TEST_AUTH === "true"`, set that flag only when running e2e
tests, and have a separate **manual smoke test** documented in the README for
the real OAuth path. Both paths exercise the same auth.ts plumbing.

**Fallback path** (env missing or placeholder): the OAuth-app create endpoint
is **not exposed by GitHub's REST/GraphQL API** — only the web UI offers it.
`gh` cannot create one for you.

1. Implement everything else in Phase 2 (schema, Drizzle adapter, `auth.ts`
   config wired up, sign-in/sign-out UI components).
2. Write the Credentials-provider-based e2e test as above.
3. Add to `PROGRESS.md → Phase 2 Follow-ups`:
   ```
   [ ] Human: create GitHub OAuth app at
       https://github.com/settings/developers
       Callback URL: http://localhost:3000/api/auth/callback/github
       Add to .env.local AND to GitHub repo secrets:
           gh secret set AUTH_GITHUB_ID --body "<id>"
           gh secret set AUTH_GITHUB_SECRET --body "<secret>"
           gh secret set AUTH_SECRET --body "$(openssl rand -base64 32)"
   ```
4. Mark the "OAuth works locally" acceptance criterion as deferred (don't
   tick it). Continue to Phase 3.

### Blocker: cannot create the GitHub repo

If `gh repo create` in Step 4 of `BOOTSTRAP.md` fails:

1. Check `gh auth status`. If not authenticated, stop and surface — there is
   no defensible workaround.
2. If authenticated but the create still fails (rate limit, name collision,
   or 403):
   - Try with a suffix: `transformer-explainer-v2`, etc. Update local
     references accordingly.
   - If still failing, work locally with no remote. Commit normally to
     `main`. At end-of-run, note in the final summary that the human needs
     to push to a remote of their choice. **Do not stop the run** — the
     value of the bundled work doesn't depend on the remote existing.

### Blocker: Neon connection failing

If `DATABASE_URL` points to a Neon instance you can't reach:

1. Fall back to local Postgres via Docker for development:
   `docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=dev postgres:16`.
   Update `.env.local` (NOT `.env.example`) to point at it.
2. Migrations and seed should still work identically.
3. Note the workaround in `PROGRESS.md`.

### Blocker: a flaky e2e test

1. Run it three times. If it passes 3/3, it wasn't flaky.
2. If it fails 1+ times, find the race (usually a missing `await
page.waitFor*`) and fix it. Do **not** add `test.retry()`.
3. If you genuinely can't fix it in a reasonable time, mark it `.skip()` with
   a comment containing `// TODO(flaky): <one-line description>` and add a
   follow-up. Do not delete the test.

### Blocker: dependency you'd like vs the locked stack

Don't add it. The stack in `CLAUDE.md §3` is locked. If you're sure something
is missing (e.g. you discover Vitest needs `happy-dom` for a particular
component test), add it as a **dev dependency only**, document the reason in
the commit message, and add a follow-up note. Anything that appears in the
shipped bundle requires the human's nod — leave a TODO and find a workaround.

### Blocker: typecheck failure you can't immediately resolve

Use `// @ts-expect-error <reason>` for at most one such case per phase, with a
specific reason. Add it to the phase's follow-ups. Never use `// @ts-ignore`.
Never widen `tsconfig.json` strictness.

### Deployment (Phase 10)

The end-of-project Vercel deployment can be partially automated. Try, in
order:

1. **Vercel CLI path.** If `vercel` is installed and `vercel whoami` returns
   a logged-in user:
   ```bash
   vercel link --yes --project transformer-explainer
   # Set production env vars (use the same names as .env.example).
   # DATABASE_URL must be the production Neon branch — ask the human if it
   # isn't already in your environment.
   for v in DATABASE_URL AUTH_SECRET AUTH_GITHUB_ID AUTH_GITHUB_SECRET \
            NEXT_PUBLIC_SITE_URL ADMIN_GITHUB_LOGINS \
            MAX_SEQ_LEN MAX_D_MODEL MAX_BLOCKS; do
     val=$(printenv "$v")
     if [ -n "$val" ]; then
       printf "%s" "$val" | vercel env add "$v" production --force
     else
       echo "MISSING: $v — add manually before first deploy."
     fi
   done
   vercel deploy --prod
   ```
2. **Vercel git integration path.** If the CLI isn't available, the human
   can connect the GitHub repo via the Vercel web UI (one click) and Vercel
   will auto-deploy on every push to `main`. Document this in the README's
   deploy section as the recommended path for first-time setup.
3. **Neither works.** Document everything Phase 10's README needs in the
   final summary. The site doesn't have to be live for the project to be
   "done" — a green CI on `main` and clear deploy docs are sufficient.

If `vercel` deploys succeed, capture the live URL and put it in the README
and in the final summary. Run a Playwright smoke test against the live URL:

```bash
PLAYWRIGHT_BASE_URL=https://<live-url> pnpm test:e2e --grep @smoke
```

Tests tagged `@smoke` should be a small subset that hits the landing page,
one explainer page, and the playground without auth. Add the tag to those
specific tests during Phase 10.

---

## 4. Commit hygiene

- One phase per commit (or a small number of commits within a phase, all
  prefixed `phase(N):`).
- No "wip" commits on `main`. Use a feature branch and squash-merge by PR
  if you want intermediate checkpoints.
- Commit message body: one paragraph on the _why_. The diff explains the
  _what_.
- Never amend or force-push a commit that's been pushed (unless you push to
  a never-merged feature branch).

---

## 5. PR description template (when pushing)

```
## Phase N — <name>

### What
Bullet list of the main pieces.

### How
Anything non-obvious about the approach.

### Deviations from SPEC/CLAUDE
- (or "none")

### Follow-ups
- (or "none")

### Verification
- pnpm lint ✓
- pnpm typecheck ✓
- pnpm test:coverage ✓ (lib/transformer/ at 100%, others ≥ 80%)
- pnpm verify:maths ✓
- pnpm test:e2e ✓ (N tests)

Closes Phase N in PROGRESS.md.
```

---

## 6. End-of-run summary

When all phases are complete, write a final message to the conversation log:

```
Run complete. <N> phases shipped across <M> commits, <P> PRs merged.

Repository: https://github.com/<owner>/transformer-explainer
Latest CI: <green/red> on main (<commit sha>)
Deployment: <live URL | "not deployed — see Human Actions below">

Phases shipped (with merged PR numbers):
  Phase 0  — foundations                        #<n>
  Phase 1  — maths layer                        #<n>
  Phase 2  — DB + auth                          #<n>
  ... (etc)

Highlights:
- <one bullet per non-trivial phase, especially anything novel or tricky>

Lighthouse scores (Phase 10): perf <n>, a11y <n>, best-practices <n>.

==========  HUMAN ACTIONS REQUIRED  ==========
[ ] Create GitHub OAuth app at https://github.com/settings/developers
    Callback (dev):   http://localhost:3000/api/auth/callback/github
    Callback (prod):  https://<vercel-domain>/api/auth/callback/github
    Then run:
        gh secret set AUTH_GITHUB_ID --body "<id>"
        gh secret set AUTH_GITHUB_SECRET --body "<secret>"
        gh secret set AUTH_SECRET --body "$(openssl rand -base64 32)"

[ ] Provision Neon project, get DATABASE_URL, run:
        gh secret set DATABASE_URL --body "<connection string>"

[ ] Connect repo to Vercel (if I couldn't: <reason>) and add the same env
    vars there. See README "Deploy" section for the click-through.

[ ] Re-run Phase 2 e2e tests once OAuth is configured:
        pnpm test:e2e -- tests/e2e/auth.spec.ts

==========  OPEN FOLLOW-UPS (deferred items)  ==========
<bullet list collected from every PROGRESS.md "Follow-ups" section>

==========  KNOWN DEVIATIONS  ==========
<bullet list collected from every PROGRESS.md "Deviations" section>
```

Then stop. Do not start a new phase. Do not refactor.
