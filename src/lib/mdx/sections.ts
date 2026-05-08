/**
 * Section catalogue + filesystem loader for /learn content.
 *
 * MDX sources live under `/content/decoder/`. Their slugs and order are
 * defined here (single source of truth) — the `[slug]` route validates
 * incoming params against this list before reading from disk.
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
