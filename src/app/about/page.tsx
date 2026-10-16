/**
 * /about — what this project is and where it came from.
 */
import Link from "next/link";

import { getSiteContext } from "@/lib/site/context";
import { siteHref } from "@/lib/site/hosts";

export const metadata = {
  title: "About",
  description:
    "How a decoder-only transformer runs, and how it is trained: why this project exists, the references it draws on, and the licence.",
};

/** The author, credited in one quiet line at the foot of the page (brief 35). */
const AUTHOR = {
  name: "Brendan Lynskey",
  url: "https://github.com/BrendanJamesLynskey",
} as const;

/**
 * The planned GPU replications (brief 38): each Training chapter that links a
 * notebook points here instead of carrying its own "pending" notice.
 */
const GPU_RUNS = [
  {
    chapter: "07-init-norm-mup",
    title: "Chapter 7",
    what: "a µP width sweep, widths 128 to 1,024 under both parameterisations",
    notebook: "07_mup_width_sweep.ipynb",
    time: "about an hour on an A100",
  },
  {
    chapter: "08-stability",
    title: "Chapter 8",
    what: "baseline, QK-norm and z-loss on a 6-layer, width-512 model",
    notebook: "08_attention_logit_growth.ipynb",
    time: "about 45 minutes on an A100",
  },
  {
    chapter: "09-mixed-precision",
    title: "Chapter 9",
    what: "FP32, BF16, FP16 with and without a GradScaler, and FP8 emulation on a 60M-parameter model",
    notebook: "09_mixed_precision.ipynb",
    time: "about 40 minutes on an A100",
  },
  {
    chapter: "10-memory",
    title: "Chapter 10",
    what: "peak memory and step time with and without activation checkpointing",
    notebook: "10_activation_checkpointing.ipynb",
    time: "about 10 minutes on any GPU",
  },
  {
    chapter: "14-scaling-laws",
    title: "Chapter 14",
    what: "an IsoFLOP sweep on GPTs of 1–40M parameters",
    notebook: "14_isoflop_sweep.ipynb",
    time: "about an hour on an A100",
  },
  {
    chapter: "15-running-a-real-job",
    title: "Chapter 15",
    what: "throughput, MFU and an exact checkpoint resume",
    notebook: "15_throughput_mfu.ipynb",
    time: "about 10 minutes on any GPU",
  },
] as const;

export default function AboutPage(): JSX.Element {
  const ctx = getSiteContext();
  return (
    <main className="mx-auto max-w-3xl space-y-10 px-6 py-12">
      <header>
        <p className="font-mono text-xs uppercase tracking-widest text-accent dark:text-indigo-300">
          /about
        </p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight">
          About this project
        </h1>
      </header>

      <section className="mdx-content">
        <p>
          This is a graphical, interactive explainer of how a decoder-only
          transformer runs, and how it is trained, in two sites of the
          Foundations track. Decoder (<code>/learn</code>): type your own
          tokens, watch every operation execute, and sample the next token
          end-to-end. Training (<code>/training</code>, at its own address):
          train that same model, from the same initial weights, on a
          public-domain book in your browser, and follow the loss, the gradients
          and the optimiser step by step. Every matrix you see is the actual
          value the model computed — the visualisations are not a separate
          model.
        </p>
        <p>
          Training&rsquo;s trainer has a hand-written backward pass for every
          operation. A NumPy reference of it matches PyTorch autograd and finite
          differences, its optimisers match <code>torch.optim</code>, and the
          TypeScript trainer matches the reference step for step (the checks and
          their tolerances are in{" "}
          <a
            href="https://github.com/BrendanJamesLynskey/transformer-explainer/blob/main/results.md"
            className="focus-ring rounded text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent dark:text-indigo-300"
          >
            results.md
          </a>
          ).
        </p>
        <p>
          It exists for two readers. End users learning the Transformer get a
          step-by-step walk-through with three layers of depth (concept, maths,
          code) they can toggle inside any section. Source-readers get a
          real-world Next.js full-stack codebase that reads like a textbook,
          with shapes, formulas, and references in the JSDoc of every operation.
        </p>
      </section>

      <section id="runs-to-come" className="scroll-mt-20 space-y-3">
        <h2 className="text-lg font-medium">Runs to come</h2>
        <p className="text-sm">
          Every number in the Training chapters comes from the model that runs
          in your browser, or from its Python reference. Six chapters also link
          a notebook that repeats the experiment at a larger scale on a GPU.
          Those runs are planned; when one is done, its output goes into the
          repository and its numbers into the chapter.
        </p>
        <ul className="space-y-1 text-sm">
          {GPU_RUNS.map((run) => (
            <li key={run.notebook}>
              <a
                href={siteHref(`/training/${run.chapter}`, ctx)}
                className="focus-ring rounded text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent dark:text-indigo-300"
              >
                {run.title}
              </a>
              : {run.what} (
              <a
                href={`https://colab.research.google.com/github/BrendanJamesLynskey/transformer-explainer/blob/main/notebooks/${run.notebook}`}
                className="focus-ring rounded text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent dark:text-indigo-300"
                rel="noopener noreferrer"
                target="_blank"
              >
                notebook
              </a>
              , {run.time}).
            </li>
          ))}
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-medium">Built with</h2>
        <ul className="grid grid-cols-2 gap-x-6 gap-y-1 font-mono text-sm">
          <li>Next.js 14 (App Router)</li>
          <li>TypeScript (strict)</li>
          <li>Tailwind CSS</li>
          <li>D3.js</li>
          <li>Drizzle ORM + Postgres</li>
          <li>Auth.js v5</li>
          <li>next-mdx-remote</li>
          <li>Vitest + Playwright</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-medium">References</h2>
        <ul className="space-y-2 text-sm">
          <li>
            Vaswani et al., 2017 —{" "}
            <a
              href="https://arxiv.org/abs/1706.03762"
              className="focus-ring rounded text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent dark:text-indigo-300"
              rel="noopener noreferrer"
              target="_blank"
            >
              Attention Is All You Need
            </a>
            . The original Transformer paper. The pre-norm decoder block,
            sinusoidal positional encoding, and scaled dot-product attention in
            this codebase all follow this paper directly.
          </li>
          <li>
            Radford et al., 2019 —{" "}
            <a
              href="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf"
              className="focus-ring rounded text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent dark:text-indigo-300"
              rel="noopener noreferrer"
              target="_blank"
            >
              Language Models are Unsupervised Multitask Learners (GPT-2)
            </a>
            . The decoder-only architecture and pre-LayerNorm pattern follow
            this work.
          </li>
          <li>
            Hendrycks &amp; Gimpel, 2016 —{" "}
            <a
              href="https://arxiv.org/abs/1606.08415"
              className="focus-ring rounded text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent dark:text-indigo-300"
              rel="noopener noreferrer"
              target="_blank"
            >
              Gaussian Error Linear Units (GELUs)
            </a>
            . The tanh approximation used in `lib/transformer/gelu.ts`.
          </li>
          <li>
            Andrej Karpathy —{" "}
            <a
              href="https://github.com/karpathy/nanoGPT"
              className="focus-ring rounded text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent dark:text-indigo-300"
              rel="noopener noreferrer"
              target="_blank"
            >
              nanoGPT
            </a>{" "}
            and{" "}
            <a
              href="https://www.youtube.com/watch?v=kCc8FmEb1nY"
              className="focus-ring rounded text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent dark:text-indigo-300"
              rel="noopener noreferrer"
              target="_blank"
            >
              &ldquo;Let&apos;s build GPT&rdquo;
            </a>
            . Reference implementations checked against during development.
          </li>
          <li>
            Kingma &amp; Ba, 2014 —{" "}
            <a
              href="https://arxiv.org/abs/1412.6980"
              className="focus-ring rounded text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent dark:text-indigo-300"
              rel="noopener noreferrer"
              target="_blank"
            >
              Adam: A Method for Stochastic Optimization
            </a>
            , and Loshchilov &amp; Hutter, 2017 —{" "}
            <a
              href="https://arxiv.org/abs/1711.05101"
              className="focus-ring rounded text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent dark:text-indigo-300"
              rel="noopener noreferrer"
              target="_blank"
            >
              Decoupled Weight Decay Regularization
            </a>{" "}
            (AdamW). The optimisers in `lib/training/optimizers.ts`.
          </li>
          <li>
            Sennrich, Haddow &amp; Birch, 2015 —{" "}
            <a
              href="https://arxiv.org/abs/1508.07909"
              className="focus-ring rounded text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent dark:text-indigo-300"
              rel="noopener noreferrer"
              target="_blank"
            >
              Neural Machine Translation of Rare Words with Subword Units
            </a>
            . Byte-pair encoding, in `lib/training/bpe.ts`.
          </li>
          <li>
            Lewis Carroll, 1865 —{" "}
            <a
              href="https://www.gutenberg.org/ebooks/11"
              className="focus-ring rounded text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent dark:text-indigo-300"
              rel="noopener noreferrer"
              target="_blank"
            >
              Alice&rsquo;s Adventures in Wonderland
            </a>{" "}
            (Project Gutenberg eBook #11, public domain). The Training
            site&rsquo;s text, lower-cased to the 64-character alphabet.
          </li>
          <li>
            Narayanan et al., 2021 —{" "}
            <a
              href="https://arxiv.org/abs/2104.04473"
              className="focus-ring rounded text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent dark:text-indigo-300"
              rel="noopener noreferrer"
              target="_blank"
            >
              Efficient Large-Scale Language Model Training on GPU Clusters
              Using Megatron-LM
            </a>
            , and Grattafiori et al., 2024 —{" "}
            <a
              href="https://arxiv.org/abs/2407.21783"
              className="focus-ring rounded text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent dark:text-indigo-300"
              rel="noopener noreferrer"
              target="_blank"
            >
              The Llama 3 Herd of Models
            </a>
            . The published runs the training-systems model
            (`lib/training/systems.ts`) is checked against.
          </li>
          <li>
            Peng et al., 2024 —{" "}
            <a
              href="https://arxiv.org/abs/2411.19870"
              className="focus-ring rounded text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent dark:text-indigo-300"
              rel="noopener noreferrer"
              target="_blank"
            >
              DeMo: Decoupled Momentum Optimization
            </a>
            , and Douillard et al., 2023 —{" "}
            <a
              href="https://arxiv.org/abs/2311.08105"
              className="focus-ring rounded text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent dark:text-indigo-300"
              rel="noopener noreferrer"
              target="_blank"
            >
              DiLoCo: Distributed Low-Communication Training of Language Models
            </a>
            . Training over slow links, in `lib/training/slowlinks.ts`.
          </li>
          <li>
            3blue1brown —{" "}
            <a
              href="https://www.3blue1brown.com/topics/neural-networks"
              className="focus-ring rounded text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent dark:text-indigo-300"
              rel="noopener noreferrer"
              target="_blank"
            >
              Neural Networks series
            </a>
            . Pedagogical inspiration for the visual intuition layer.
          </li>
        </ul>
      </section>

      <section className="space-y-2">
        <h2 className="text-lg font-medium">Licence</h2>
        <p className="text-sm text-neutral-600 dark:text-neutral-400">
          MIT. Use the code as a learning resource, fork it, ship it. See{" "}
          <a
            href="https://github.com/BrendanJamesLynskey/transformer-explainer/blob/main/LICENSE"
            className="focus-ring rounded text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent dark:text-indigo-300"
            rel="noopener noreferrer"
            target="_blank"
          >
            LICENSE
          </a>{" "}
          for the text.
        </p>
      </section>

      <p className="text-sm">
        <Link
          href="/learn"
          className="focus-ring rounded text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent dark:text-indigo-300"
        >
          → Decoder
        </Link>{" "}
        ·{" "}
        <a
          href={siteHref("/training", ctx)}
          className="focus-ring rounded text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent dark:text-indigo-300"
        >
          Training
        </a>
      </p>

      <p
        className="text-sm text-neutral-600 dark:text-neutral-400"
        data-author-credit
      >
        Built by{" "}
        <a
          href={AUTHOR.url}
          className="focus-ring rounded text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent dark:text-indigo-300"
        >
          {AUTHOR.name}
        </a>
      </p>
    </main>
  );
}
