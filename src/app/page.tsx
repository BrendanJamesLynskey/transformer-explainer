import Link from "next/link";

/**
 * Landing page: what the explainer is, and the three ways in (the guided
 * lessons, the free-form playground, and saved experiments).
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

      <nav aria-label="Get started" className="mt-8 flex flex-wrap gap-3">
        <Link
          href="/learn"
          className="focus-ring rounded bg-accent px-4 py-2 text-sm font-medium text-accent-fg hover:opacity-90"
        >
          Start the lessons →
        </Link>
        <Link
          href="/playground"
          className="focus-ring rounded border border-neutral-300 px-4 py-2 text-sm font-medium hover:border-accent dark:border-neutral-700"
        >
          Open the playground
        </Link>
        <Link
          href="/experiments"
          className="focus-ring rounded border border-neutral-300 px-4 py-2 text-sm font-medium hover:border-accent dark:border-neutral-700"
        >
          Browse saved experiments
        </Link>
      </nav>

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
    </main>
  );
}
