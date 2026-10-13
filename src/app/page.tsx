import Link from "next/link";

import { AttentionPreviewSvg } from "@/components/viz/AttentionPreviewSvg";
import { attentionPreview } from "@/lib/attention-preview";
import { getSiteContext } from "@/lib/site/context";
import { siteHref } from "@/lib/site/hosts";

/** The sentence the preview runs through the model. */
const PREVIEW_TEXT = "the cat sat";

const ENTRY_POINTS = [
  {
    href: "/learn",
    title: "Decoder",
    blurb:
      "How the decoder runs: seven short chapters, from token embeddings to sampling. Toggle the Concept, Maths and Code layers to choose your depth.",
    cta: "Start the Decoder chapters →",
  },
  {
    href: "/training",
    title: "Training",
    blurb:
      "The next site: how the same model is trained, from the loss, backprop and optimisers to many GPUs, training in your browser, checked step by step against a PyTorch-verified reference.",
    cta: "Go to Training →",
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
 * Landing page of the Decoder site: what it is (how a decoder-only
 * transformer runs; its sibling in the Foundations track, Training, shows
 * how it is trained), one real visualisation, and the ways in (the Decoder
 * chapters, Training, the playground, saved experiments).
 *
 * Server Component with no client JavaScript of its own: the attention
 * preview is computed on the server by the same maths library the lessons
 * use, and drawn as static SVG.
 */
export default function HomePage(): JSX.Element {
  const ctx = getSiteContext();
  const preview = attentionPreview({ text: PREVIEW_TEXT });

  return (
    <main className="mx-auto max-w-5xl px-6 py-12 sm:py-20">
      <section className="grid items-center gap-10 lg:grid-cols-[1fr_auto]">
        <div>
          <p className="font-mono text-xs uppercase tracking-widest text-accent dark:text-indigo-300">
            Transformer Decoder Explainer
          </p>
          <h1 className="mt-4 text-4xl font-semibold tracking-tight sm:text-5xl">
            How a decoder-only transformer runs, and how it is trained.
          </h1>
          <p className="mt-6 max-w-2xl text-lg text-neutral-600 dark:text-neutral-300">
            This site, Decoder, follows your own text through every operation of
            a GPT-style decoder: embeddings, masked self-attention, the
            feed-forward network, LayerNorm and residuals, stacked blocks, and
            next-token sampling. Its sibling site, Training, trains that same
            model: the loss, backpropagation and the optimiser, step by step, in
            your browser. Every number comes from a small, readable TypeScript
            implementation, checked against PyTorch.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link
              href="/learn"
              className="focus-ring rounded bg-accent px-4 py-2 text-sm font-medium text-accent-fg hover:opacity-90"
            >
              Start the Decoder chapters →
            </Link>
            <Link
              href={siteHref("/training", ctx)}
              className="focus-ring rounded border border-neutral-300 px-4 py-2 text-sm font-medium hover:border-accent dark:border-neutral-700"
            >
              Next: Training →
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

      <nav
        aria-label="Ways in"
        className="mt-16 grid gap-4 sm:grid-cols-2 lg:grid-cols-4"
      >
        {ENTRY_POINTS.map((e) => (
          <Link
            key={e.href}
            href={siteHref(e.href, ctx)}
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
