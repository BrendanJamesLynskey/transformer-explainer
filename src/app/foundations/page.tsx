/**
 * /foundations — the Foundations track's start page (brief 44): the two
 * sites in reading order, the prerequisites stated plainly, how a chapter
 * works (layers, self-checks, what to remember), and a short glossary whose
 * terms link to the chapter section that defines each one.
 *
 * Served by the decoder host (the training host redirects here, like every
 * page that is not Training's). Server Component, no client JavaScript.
 */
import { ReadingOrder, type TrackSite } from "@/components/learn/ReadingOrder";
import { FOUNDATIONS_GLOSSARY as GLOSSARY } from "@/lib/learn/foundations";
import { TRAINING_DECK_HUB } from "@/lib/mdx/sections";
import { withShare } from "@/lib/share";
import { getSiteContext } from "@/lib/site/context";
import { siteHref } from "@/lib/site/hosts";

export const metadata = withShare("/foundations", {
  title: "Foundations: start here",
  description:
    "The Foundations track's start page: the Decoder and Training sites in reading order, what you need to know first, and a glossary of the terms they define.",
});

const LINK =
  "focus-ring rounded text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent dark:text-indigo-300";

/** A refresher for the linear algebra (a deck series on the same GitHub). */
const LINEAR_ALGEBRA =
  "https://brendanjameslynskey.github.io/LLM_Hub_Linear_Algebra/";

export default function FoundationsPage(): JSX.Element {
  const ctx = getSiteContext();
  const sites: TrackSite[] = [
    {
      key: "decoder",
      name: "Decoder",
      url: siteHref("/learn", ctx),
      why: "one forward pass of a small GPT-style model, operation by operation, from embeddings and attention to sampling; assumes only the prerequisites below",
    },
    {
      key: "training",
      name: "Training",
      url: siteHref("/training", ctx),
      why: "how that same model is trained, from the loss and backprop to optimisers, precision, memory and many GPUs; assumes the Decoder",
    },
  ];
  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <p className="font-mono text-xs uppercase tracking-widest text-accent dark:text-indigo-300">
        Foundations track
      </p>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">
        Start here: how a transformer runs, and how it is trained
      </h1>
      <p className="mt-4 text-neutral-600 dark:text-neutral-300">
        Two sites, one model. The Decoder follows your own text through every
        operation of a small decoder; Training trains that same model, in your
        browser, and then scales the ideas up to thousands of GPUs. Read them in
        order: every later site in the family (serving, hardware, agents)
        assumes these two.
      </p>

      <ReadingOrder
        track="Foundations"
        intro="Each chapter builds on the one before it. Start at the Decoder's chapter 1; the Training site picks up where the Decoder ends."
        sites={sites}
      />

      <section aria-labelledby="prereq-heading" className="mt-12">
        <h2
          id="prereq-heading"
          className="text-xl font-semibold tracking-tight"
        >
          What you need first
        </h2>
        <ul className="mt-4 list-disc space-y-2 pl-6 text-neutral-700 dark:text-neutral-300">
          <li>
            <strong>Vectors and matrices.</strong> A dot product, a matrix times
            a vector, and which shapes can be multiplied. Every operation in the
            Decoder is one of these.{" "}
            <a href={LINEAR_ALGEBRA} className={LINK}>
              A linear algebra refresher
            </a>{" "}
            (slides) covers it.
          </li>
          <li>
            <strong>Exponentials and logarithms.</strong> What <em>e</em>
            <sup>x</sup> and ln&nbsp;x do, and that a log turns a product into a
            sum. The softmax and the loss are built from them.
          </li>
          <li>
            <strong>A little probability.</strong> A distribution&rsquo;s
            probabilities add up to 1, and sampling draws from it.
          </li>
          <li>
            <strong>Derivatives and the chain rule</strong>, for Training
            chapters 4 and 5 only. The Concept layer explains them in words; the
            Maths layer gives the formulas.
          </li>
          <li>
            <strong>Reading code helps but is optional.</strong> Each
            chapter&rsquo;s Code layer shows the TypeScript that runs; hide it
            with the layer switch at the top of the chapter.
          </li>
        </ul>
        <p className="mt-4 text-sm text-neutral-600 dark:text-neutral-400">
          You don&rsquo;t need PyTorch, a GPU or an account: everything runs in
          the browser or on this site&rsquo;s server, and signing in only keeps
          your comments and progress.
        </p>
      </section>

      <section aria-labelledby="how-heading" className="mt-12">
        <h2 id="how-heading" className="text-xl font-semibold tracking-tight">
          How a chapter works
        </h2>
        <p className="mt-4 text-neutral-700 dark:text-neutral-300">
          A chapter opens with an animation of its mechanism, computed by the
          site&rsquo;s tested model, then explains it in three layers you can
          switch on and off: Concept (words and pictures), Maths (the formulas)
          and Code (the code that runs). It ends with{" "}
          <strong>Check yourself</strong>, three questions to commit to before
          you reveal the answers (the numbers come from the same model), and{" "}
          <strong>What to remember</strong>, the chapter in a few lines.
        </p>
      </section>

      <section aria-labelledby="glossary-heading" className="mt-12">
        <h2
          id="glossary-heading"
          className="text-xl font-semibold tracking-tight"
        >
          Glossary
        </h2>
        <p className="mt-2 text-sm text-neutral-600 dark:text-neutral-400">
          Each term links to the section that defines it. The{" "}
          <a href={`${TRAINING_DECK_HUB}#glossary`} className={LINK}>
            Transformer Training slide series&rsquo; glossary
          </a>{" "}
          has more.
        </p>
        <dl
          data-foundations-glossary
          className="mt-4 divide-y divide-neutral-200 text-sm dark:divide-neutral-800"
        >
          {GLOSSARY.map((g) => (
            <div key={g.term} className="py-2 sm:flex sm:gap-4">
              <dt className="font-medium sm:w-56 sm:shrink-0">
                <a href={siteHref(g.href, ctx)} className={LINK}>
                  {g.term}
                </a>
              </dt>
              <dd className="text-neutral-600 dark:text-neutral-400">
                {g.what}
              </dd>
            </div>
          ))}
        </dl>
      </section>
    </main>
  );
}
