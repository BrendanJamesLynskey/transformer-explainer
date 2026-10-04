import Link from "next/link";

import { AttentionPreviewSvg } from "@/components/viz/AttentionPreviewSvg";
import { attentionPreview } from "@/lib/attention-preview";

/** The sentence the preview runs through the model. */
const PREVIEW_TEXT = "the cat sat";

const ENTRY_POINTS = [
  {
    href: "/learn",
    title: "Lessons",
    blurb:
      "Seven short sections, from token embeddings to sampling. Toggle the Concept, Maths and Code layers to choose your depth.",
    cta: "Start the lessons →",
  },
  {
    href: "/playground",
    title: "Playground",
    blurb:
      "The whole decoder on one page. Change the input, the model size and the sampler, and generate text one token at a time.",
    cta: "Open the playground →",
  },
  {
    href: "/experiments",
    title: "Experiments",
    blurb:
      "Configurations other readers saved and shared. Sign in with GitHub to save your own or fork theirs.",
    cta: "Browse experiments →",
  },
] as const;

/**
 * Landing page: what the explainer is, one real visualisation, and the
 * three ways in (lessons, playground, saved experiments).
 *
 * Server Component with no client JavaScript of its own: the attention
 * preview is computed on the server by the same maths library the lessons
 * use, and drawn as static SVG.
 */
export default function HomePage(): JSX.Element {
  const preview = attentionPreview({ text: PREVIEW_TEXT });

  return (
    <main className="mx-auto max-w-5xl px-6 py-12 sm:py-20">
      <section className="grid items-center gap-10 lg:grid-cols-[1fr_auto]">
        <div>
          <p className="font-mono text-xs uppercase tracking-widest text-accent dark:text-indigo-300">
            Transformer Decoder Explainer
          </p>
          <h1 className="mt-4 text-4xl font-semibold tracking-tight sm:text-5xl">
            Watch a decoder think, one matrix at a time.
          </h1>
          <p className="mt-6 max-w-2xl text-lg text-neutral-600 dark:text-neutral-300">
            Type in your own text and follow it through every operation of a
            GPT-style decoder: embeddings, masked self-attention, the
            feed-forward network, LayerNorm and residuals, stacked blocks, and
            next-token sampling. The numbers you see are computed on the server
            by a small, readable TypeScript implementation.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
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
          </div>
        </div>

        <figure className="mx-auto w-full max-w-[280px] rounded-lg border border-neutral-200 p-4 dark:border-neutral-800">
          <AttentionPreviewSvg preview={preview} />
          <figcaption className="mt-3 text-xs text-neutral-600 dark:text-neutral-400">
            Real attention weights for &ldquo;{PREVIEW_TEXT}&rdquo; (head 0,
            seed 42). Each row is a letter looking back at itself and earlier
            letters; grey cells are masked. The model&rsquo;s weights are
            random, not trained, so each row spreads its attention almost
            evenly: the last row gives each of its 11 letters about 1/11.{" "}
            <Link
              href="/learn/03-attention"
              className="focus-ring rounded text-accent underline underline-offset-2 dark:text-indigo-300"
            >
              Try your own text
            </Link>
          </figcaption>
        </figure>
      </section>

      <nav aria-label="Ways in" className="mt-16 grid gap-4 sm:grid-cols-3">
        {ENTRY_POINTS.map((e) => (
          <Link
            key={e.href}
            href={e.href}
            className="focus-ring group rounded-lg border border-neutral-200 p-5 hover:border-accent dark:border-neutral-800 dark:hover:border-indigo-400"
          >
            <h2 className="font-semibold">{e.title}</h2>
            <p className="mt-2 text-sm text-neutral-600 dark:text-neutral-400">
              {e.blurb}
            </p>
            <p className="mt-4 text-sm font-medium text-accent dark:text-indigo-300">
              {e.cta}
            </p>
          </Link>
        ))}
      </nav>

      <p className="mt-12 text-sm text-neutral-600 dark:text-neutral-400">
        The code is a teaching artefact too: read the source on{" "}
        <a
          href="https://github.com/BrendanJamesLynskey/transformer-explainer"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          GitHub
        </a>{" "}
        for Next.js full-stack patterns alongside the maths.
      </p>
    </main>
  );
}
