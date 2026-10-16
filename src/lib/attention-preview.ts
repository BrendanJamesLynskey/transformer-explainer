/**
 * The attention weights behind the landing page's hero picture.
 *
 * Runs the full forward pass the lessons run (tokenise, embed, add the
 * positional encoding, then both pre-norm blocks) and returns every head's
 * softmax weights. By default the weights are Training's trained ones, so
 * the picture shows what the model learned; `kind: "random"` gives the
 * seeded initialiser it started from. The landing page renders the result
 * as a static SVG on the server, so the picture is real output of
 * `lib/transformer`, not a drawing, and it costs the browser no JavaScript.
 */
import { configFor, DEFAULT_PARAMS, tracedForward } from "@/lib/compute/traces";
import { initModelWeights } from "@/lib/transformer/init";
import { trainedWeights, type WeightKind } from "@/lib/transformer/trained";
import { ALPHABET, encode } from "@/lib/transformer/tokenizer";
import type { Matrix } from "@/lib/transformer/types";

/** One head's softmax weights, `[S, S]`; rows sum to 1. */
export type HeadPattern = { block: number; head: number; weights: Matrix };

export type AttentionPreview = {
  /** One character per position (the toy tokenizer is character-level). */
  tokens: string[];
  /** Block 0's weights for the chosen head, shape `[S, S]`. */
  weights: Matrix;
  /** Every head of every block, block-major. */
  heads: HeadPattern[];
};

/**
 * Compute the attention weights of every head for `text`.
 *
 * @param text  Input string; its length sets the sequence length (the
 *              trained model saw 8-character windows; positions are
 *              sinusoidal, so longer inputs run, but extrapolate).
 * @param kind  "trained" (default) or "random" (the seeded initialiser).
 * @param seed  The initialiser's seed when `kind` is "random".
 * @param head  Which block-0 head `weights` returns.
 */
export function attentionPreview({
  text,
  kind = "trained",
  seed = 42,
  head = 0,
}: {
  text: string;
  kind?: WeightKind;
  seed?: number;
  head?: number;
}): AttentionPreview {
  const config = configFor({ ...DEFAULT_PARAMS, seqLen: text.length, seed });
  const w = kind === "trained" ? trainedWeights() : initModelWeights(config);
  const tokenIds = encode(text, text.length);
  const trace = tracedForward(tokenIds, config, w);
  const heads: HeadPattern[] = [];
  trace.blocks.forEach((bt, b) =>
    bt.attn.weights.forEach((m, h) =>
      heads.push({ block: b, head: h, weights: m }),
    ),
  );
  return {
    tokens: tokenIds.map((id) => ALPHABET[id] ?? "?"),
    weights: trace.blocks[0]!.attn.weights[head] ?? [],
    heads,
  };
}
