/**
 * Landing page (placeholder).
 *
 * Phase 0 ships a minimal "what this is + start here" page so `pnpm dev`
 * has something to render. Phase 10 will replace this with the polished
 * marketing landing page (animated hero, mini playground preview, etc.).
 */
export default function HomePage(): JSX.Element {
  return (
    <main className="mx-auto max-w-3xl px-6 py-16 sm:py-24">
      <p className="font-mono text-xs uppercase tracking-widest text-accent">
        Transformer Explainer
      </p>
      <h1 className="mt-4 text-4xl font-semibold tracking-tight sm:text-5xl">
        Watch a decoder think, one matrix at a time.
      </h1>
      <p className="mt-6 text-lg text-neutral-600 dark:text-neutral-300">
        A graphical, interactive walk-through of the Transformer decoder. Type
        in tokens, then see every operation — embeddings, masked self-attention,
        FFN, layernorm, residuals, and final next-token sampling — execute on
        the server and visualised step-by-step in your browser.
      </p>

      <section className="mt-12 grid gap-4 text-sm text-neutral-700 dark:text-neutral-300 sm:grid-cols-2">
        <div className="rounded-lg border border-neutral-200 p-4 dark:border-neutral-800">
          <h2 className="font-semibold">For learners</h2>
          <p className="mt-2">
            Three layers — concept, maths, code — togglable per section, so you
            can stay at the level you find useful.
          </p>
        </div>
        <div className="rounded-lg border border-neutral-200 p-4 dark:border-neutral-800">
          <h2 className="font-semibold">For code readers</h2>
          <p className="mt-2">
            The codebase is itself a teaching artefact. Browse the source on{" "}
            <a
              href="https://github.com/BrendanJamesLynskey/transformer-explainer"
              className="focus-ring underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
            >
              GitHub
            </a>{" "}
            to see Next.js full-stack patterns alongside the maths.
          </p>
        </div>
      </section>

      <p className="mt-12 font-mono text-xs text-neutral-500 dark:text-neutral-500">
        Phase 0 placeholder · /learn and /playground come online in later
        phases.
      </p>
    </main>
  );
}
