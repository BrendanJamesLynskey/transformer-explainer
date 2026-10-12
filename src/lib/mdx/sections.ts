/**
 * Section catalogue + filesystem loader for the two parts' chapters.
 *
 * Part 1 (Inference, `/learn`): MDX under `/content/decoder/`.
 * Part 2 (Training, `/training`): MDX under `/content/training/`.
 * Slugs and order are defined here (single source of truth) — the `[slug]`
 * routes validate incoming params against these lists before reading from
 * disk. A chapter can be listed before its MDX ships ("coming soon").
 */
import { readFile } from "node:fs/promises";
import { join } from "node:path";

/** Pedagogical order — see SPEC §5 ("Explainer sections"). */
export const SECTIONS = [
  {
    slug: "01-overview",
    title: "Overview",
    summary: "What a decoder does. Tokens in, logits out.",
  },
  {
    slug: "02-embeddings",
    title: "Embeddings",
    summary: "Token lookup + sinusoidal positional encoding.",
  },
  {
    slug: "03-attention",
    title: "Attention",
    summary: "Q/K/V, scores, mask, softmax, output. The headline act.",
  },
  {
    slug: "04-ffn",
    title: "Feed-forward",
    summary: "Position-wise W₂·GELU(W₁·x + b) + b.",
  },
  {
    slug: "05-layernorm-residuals",
    title: "LayerNorm & residuals",
    summary: "Why residuals; why LayerNorm; pre-norm vs post-norm.",
  },
  {
    slug: "06-stacking",
    title: "Stacking blocks",
    summary: "What changes layer-to-layer.",
  },
  {
    slug: "07-sampling",
    title: "Sampling",
    summary: "Logits → next token. Greedy, temperature, top-k, top-p.",
  },
] as const;

export type SectionSlug = (typeof SECTIONS)[number]["slug"];

/**
 * Part 2, Training (brief 29): 16 chapters. 29A ships 01–05; the rest are
 * listed so the index shows the whole part.
 */
export const TRAINING_SECTIONS = [
  {
    slug: "01-training-loop",
    title: "The training loop",
    summary:
      "Batch, forward, loss, backward, update: Part 1's model trains in your browser.",
  },
  {
    slug: "02-data-and-tokens",
    title: "Data and tokenisation",
    summary:
      "A public-domain corpus, BPE merges learnt live, windows and the validation split.",
  },
  {
    slug: "03-objective",
    title: "The objective",
    summary:
      "Next-token cross-entropy, teacher forcing, perplexity, and the gradient p − y.",
  },
  {
    slug: "04-backprop",
    title: "Backprop through the model",
    summary:
      "The gradient's way back, op by op, and what the forward pass must save.",
  },
  {
    slug: "05-optimisers",
    title: "Optimisers",
    summary:
      "SGD, momentum, Adam and AdamW, on a valley and on the real model.",
  },
  {
    slug: "06-learning-rate-and-schedules",
    title: "Learning rate, batch size and schedules",
    summary:
      "Warmup, cosine and WSD; the learning-rate sweep; batch size and gradient noise.",
  },
  {
    slug: "07-init-norm-mup",
    title: "Initialisation, normalisation and µP",
    summary:
      "Pre-LN vs post-LN, RMSNorm, scale through depth, hyperparameter transfer.",
  },
  {
    slug: "08-stability",
    title: "Stability",
    summary: "A loss spike reproduced and cured: clipping, z-loss, QK-norm.",
  },
  {
    slug: "09-mixed-precision",
    title: "Mixed precision",
    summary:
      "FP32 master weights, bf16 and fp16 with loss scaling, FP8, quantised gradients.",
  },
  {
    slug: "10-memory",
    title: "Memory",
    summary:
      "Parameters, gradients, optimiser states and activations; checkpointing.",
  },
  {
    slug: "11-data-parallelism",
    title: "Parallelism I: data",
    summary:
      "Data parallelism, all-reduce, reduce-scatter and all-gather, ZeRO and FSDP.",
  },
  {
    slug: "12-model-parallelism",
    title: "Parallelism II: model",
    summary:
      "Tensor, pipeline, sequence and expert parallelism; the pipeline bubble.",
  },
  {
    slug: "13-slow-links",
    title: "Training over slow links",
    summary:
      "Top-k with error feedback, DeMo and DiLoCo when workers share a WAN.",
  },
  {
    slug: "14-scaling-laws",
    title: "Scaling laws",
    summary: "Kaplan and Chinchilla, IsoFLOP curves, compute-optimal training.",
  },
  {
    slug: "15-running-a-real-job",
    title: "Running a real job",
    summary: "Throughput, MFU, monitoring, failures, checkpoints and goodput.",
  },
  {
    slug: "16-after-pretraining",
    title: "After pretraining",
    summary: "Mid-training, SFT, preference optimisation and RL, in outline.",
  },
] as const;

export type TrainingSlug = (typeof TRAINING_SECTIONS)[number]["slug"];

const TRAINING_SLUG_SET = new Set<string>(TRAINING_SECTIONS.map((s) => s.slug));

export function isTrainingSlug(slug: string): slug is TrainingSlug {
  return TRAINING_SLUG_SET.has(slug);
}

export function getTrainingMeta(
  slug: TrainingSlug,
): (typeof TRAINING_SECTIONS)[number] {
  return TRAINING_SECTIONS.find((s) => s.slug === slug) ?? TRAINING_SECTIONS[0];
}

/**
 * The key a Training chapter's comments, progress and analytics are stored
 * under (`training-01-training-loop`): the same text columns as Part 1's
 * chapters, so no schema change, and never equal to a Part 1 slug.
 */
export function trainingSectionKey(slug: TrainingSlug): string {
  return `training-${slug}`;
}

/** Slugs whose comment threads exist: Part 1's slugs and Part 2's keys. */
export function isCommentableSlug(slug: string): boolean {
  if (isValidSlug(slug)) return true;
  return (
    slug.startsWith("training-") &&
    isTrainingSlug(slug.slice("training-".length))
  );
}

/** Read a Training chapter's MDX, or null if it has not shipped yet. */
export async function readTrainingMdx(
  slug: TrainingSlug,
): Promise<string | null> {
  const path = join(process.cwd(), "content", "training", `${slug}.mdx`);
  try {
    return await readFile(path, "utf-8");
  } catch {
    return null;
  }
}

const SLUG_SET = new Set<string>(SECTIONS.map((s) => s.slug));

export function isValidSlug(slug: string): slug is SectionSlug {
  return SLUG_SET.has(slug);
}

export function getSectionMeta(slug: SectionSlug): (typeof SECTIONS)[number] {
  return SECTIONS.find((s) => s.slug === slug) ?? SECTIONS[0];
}

/**
 * Read the raw MDX source for a section. Returns `null` if the file
 * doesn't exist yet (a section can be listed in `SECTIONS` before its
 * MDX has shipped, in which case the page renders a "coming soon"
 * placeholder).
 */
export async function readSectionMdx(
  slug: SectionSlug,
): Promise<string | null> {
  const path = join(process.cwd(), "content", "decoder", `${slug}.mdx`);
  try {
    return await readFile(path, "utf-8");
  } catch {
    return null;
  }
}
