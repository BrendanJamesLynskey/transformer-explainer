/**
 * /about — what this project is and where it came from.
 */
import Link from "next/link";

export const metadata = {
  title: "About",
  description:
    "How a decoder-only transformer runs, and how it is trained: why this project exists, the references it draws on, and the licence.",
};

export default function AboutPage(): JSX.Element {
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

      <section className="prose prose-neutral dark:prose-invert max-w-none">
        <p>
          This is a graphical, interactive explainer of how a decoder-only
          transformer runs, and how it is trained. Part 1 (Inference): type your
          own tokens, watch every operation execute, and sample the next token
          end-to-end. Part 2 (Training): train that same model, from the same
          initial weights, on a public-domain book in your browser, and follow
          the loss, the gradients and the optimiser step by step. Every matrix
          you see is the actual value the model computed — the visualisations
          are not a separate model.
        </p>
        <p>
          Part 2&rsquo;s trainer has a hand-written backward pass for every
          operation. A NumPy reference of it matches PyTorch autograd and finite
          differences, its optimisers match <code>torch.optim</code>, and the
          TypeScript trainer matches the reference step for step (the checks and
          their tolerances are in{" "}
          <a
            href="https://github.com/BrendanJamesLynskey/transformer-explainer/blob/main/results.md"
            className="focus-ring rounded text-accent hover:underline dark:text-indigo-300"
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
              className="focus-ring rounded text-accent hover:underline dark:text-indigo-300"
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
              className="focus-ring rounded text-accent hover:underline dark:text-indigo-300"
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
              className="focus-ring rounded text-accent hover:underline dark:text-indigo-300"
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
              className="focus-ring rounded text-accent hover:underline dark:text-indigo-300"
              rel="noopener noreferrer"
              target="_blank"
            >
              nanoGPT
            </a>{" "}
            and{" "}
            <a
              href="https://www.youtube.com/watch?v=kCc8FmEb1nY"
              className="focus-ring rounded text-accent hover:underline dark:text-indigo-300"
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
              className="focus-ring rounded text-accent hover:underline dark:text-indigo-300"
              rel="noopener noreferrer"
              target="_blank"
            >
              Adam: A Method for Stochastic Optimization
            </a>
            , and Loshchilov &amp; Hutter, 2017 —{" "}
            <a
              href="https://arxiv.org/abs/1711.05101"
              className="focus-ring rounded text-accent hover:underline dark:text-indigo-300"
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
              className="focus-ring rounded text-accent hover:underline dark:text-indigo-300"
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
              className="focus-ring rounded text-accent hover:underline dark:text-indigo-300"
              rel="noopener noreferrer"
              target="_blank"
            >
              Alice&rsquo;s Adventures in Wonderland
            </a>{" "}
            (Project Gutenberg eBook #11, public domain). Part 2&rsquo;s
            training text, lower-cased to the 64-character alphabet.
          </li>
          <li>
            3blue1brown —{" "}
            <a
              href="https://www.3blue1brown.com/topics/neural-networks"
              className="focus-ring rounded text-accent hover:underline dark:text-indigo-300"
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
            className="focus-ring rounded text-accent hover:underline dark:text-indigo-300"
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
          className="focus-ring rounded text-accent hover:underline dark:text-indigo-300"
        >
          → Part 1: Inference
        </Link>{" "}
        ·{" "}
        <Link
          href="/training"
          className="focus-ring rounded text-accent hover:underline dark:text-indigo-300"
        >
          Part 2: Training
        </Link>
      </p>
    </main>
  );
}
