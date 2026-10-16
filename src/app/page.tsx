import type { Route } from "next";
import Link from "next/link";

import { CheckedStrip, StartButton } from "@/components/ui/Landing";
import { TrainedHeadsSvg } from "@/components/viz/TrainedHeadsSvg";
import heads from "@/data/decoder/trained-heads.json";
import { attentionPreview } from "@/lib/attention-preview";
import { SECTIONS } from "@/lib/mdx/sections";
import { getSiteContext } from "@/lib/site/context";
import { siteHref } from "@/lib/site/hosts";

/** The text the hero runs through the model: 8 characters, the trained context. */
const PREVIEW_TEXT = "the cat ";

const pct = (v: number) => `${Math.round(v * 100)}%`;
const at = (b: number, h: number) =>
  heads.trained.find((s) => s.block === b && s.head === h)!;

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
  const trained = attentionPreview({ text: PREVIEW_TEXT });
  const random = attentionPreview({ text: PREVIEW_TEXT, kind: "random" });
  const local = (b: number, h: number) => pct(at(b, h).self + at(b, h).prev);

  return (
    <main className="mx-auto max-w-5xl px-6 py-12 sm:py-20">
      <p className="font-mono text-xs uppercase tracking-widest text-accent dark:text-indigo-300">
        Transformer Decoder Explained
      </p>
      <h1 className="mt-4 text-4xl font-semibold tracking-tight sm:text-5xl">
        How a decoder-only transformer runs, and how it is trained.
      </h1>
      <div className="mt-6 grid items-start gap-8 md:grid-cols-[1fr_auto]">
        <div>
          <p className="max-w-2xl text-lg text-neutral-600 dark:text-neutral-300">
            An interactive guide that follows your own text through every
            operation of a GPT-style decoder, computed live in your browser.
          </p>
          <div className="mt-6">
            <StartButton href={`/learn/${SECTIONS[0]!.slug}` as Route} />
          </div>
          <p className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-sm text-neutral-600 dark:text-neutral-400">
            <a
              href={siteHref("/training", ctx)}
              className="focus-ring rounded font-medium underline decoration-neutral-400 underline-offset-4 hover:decoration-current"
            >
              Next: Training →
            </a>
            <Link
              href="/playground"
              className="focus-ring rounded font-medium underline decoration-neutral-400 underline-offset-4 hover:decoration-current"
            >
              Open the playground
            </Link>
          </p>
        </div>
        <figure
          data-hero-figure
          className="mx-auto w-full max-w-[340px] rounded-lg border border-neutral-200 p-4 dark:border-neutral-800"
        >
          <TrainedHeadsSvg
            trained={trained}
            random={random}
            stats={heads.trained}
            randomStats={heads.random}
          />
          <figcaption className="mt-3 text-xs text-neutral-600 dark:text-neutral-400">
            The trained model&rsquo;s four attention heads on &ldquo;the
            cat&nbsp;&rdquo;: Training&rsquo;s default run, 2,000 steps on
            <em> Alice&rsquo;s Adventures in Wonderland</em>. Across{" "}
            {heads.trained[0]!.windows.toLocaleString("en-GB")} windows of the
            book, block 1&rsquo;s heads put {local(0, 0)} and {local(0, 1)} of
            each row on the current letter and the one before it; block
            2&rsquo;s head 1 sends {pct(at(1, 1).spaceStart)} of a space&rsquo;s
            attention to the window&rsquo;s first two letters. Before training
            (bottom) every row is nearly even.{" "}
            <Link
              href="/learn/03-attention"
              className="focus-ring rounded text-accent underline underline-offset-2 dark:text-indigo-300"
            >
              Try your own text
            </Link>
          </figcaption>
        </figure>
      </div>
      <CheckedStrip href="/about#checked">
        The TypeScript transformer in your browser is checked against PyTorch,
        operation by operation, to within 1e-5, and the animations&apos; frames
        against a float64 PyTorch run of the same weights, trained and at random
        init.
      </CheckedStrip>
      <section aria-labelledby="learn-heading" className="mt-12">
        <h2 id="learn-heading" className="text-xl font-semibold tracking-tight">
          What you&apos;ll learn
        </h2>
        <p className="mt-4 max-w-2xl text-neutral-600 dark:text-neutral-300">
          This site, Decoder, follows your own text through every operation of a
          GPT-style decoder: embeddings, masked self-attention, the feed-forward
          network, LayerNorm and residuals, stacked blocks, and next-token
          sampling. Its sibling site, Training, trains that same model: the
          loss, backpropagation and the optimiser, step by step, in your
          browser. Every number comes from a small, readable TypeScript
          implementation, checked against PyTorch.
        </p>
      </section>

      <nav
        aria-label="Ways in"
        className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-4"
      >
        {ENTRY_POINTS.map((e) => {
          const content = (
            <>
              <h2 className="font-semibold">{e.title}</h2>
              <p className="mt-2 text-sm text-neutral-600 dark:text-neutral-400">
                {e.blurb}
              </p>
              <p className="mt-4 text-sm font-medium text-accent dark:text-indigo-300">
                {e.cta}
              </p>
            </>
          );
          // Training is the other Foundations site: a full page load
          return e.href === "/training" ? (
            <a
              key={e.href}
              href={siteHref(e.href, ctx)}
              className="focus-ring group rounded-lg border border-neutral-200 p-5 hover:border-accent dark:border-neutral-800 dark:hover:border-indigo-400"
            >
              {content}
            </a>
          ) : (
            <Link
              key={e.href}
              href={siteHref(e.href, ctx)}
              className="focus-ring group rounded-lg border border-neutral-200 p-5 hover:border-accent dark:border-neutral-800 dark:hover:border-indigo-400"
            >
              {content}
            </Link>
          );
        })}
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
