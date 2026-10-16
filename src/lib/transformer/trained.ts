/*
 * src/lib/transformer/trained.ts
 *
 * Operation: the trained weights the Decoder's widgets use by default, and
 *            the switch between them and the seeded random initialiser.
 * Shapes:    trainedWeights() → ModelWeights for TRAINED_CONFIG
 *            (seq_len 8, d_model 16, 2 heads, d_ff 32, 2 blocks, vocab 64).
 * Intuition: the same model, after Training's default run (2,000 AdamW
 *            steps on Alice, validation loss 2.30). Training writes the
 *            file (scripts/training/make_fixtures.py), so the Decoder reads
 *            exactly the weights chapter 1 of Training ends with. Random
 *            init stays one click away: the contrast is the lesson.
 * MDX:       every /learn/<decoder chapter> widget ("Weights" toggle).
 */
import raw from "../../../public/training/default-run-weights.json";

import { initModelWeights } from "./init";
import { unpackWeights, type ModelConfig, type ModelWeights } from "./model";
import type { Matrix, Vector } from "./types";

/** Which weights a widget runs. */
export type WeightKind = "trained" | "random";

/** The trained model's hyperparameters (Training's default run). */
export const TRAINED_CONFIG: ModelConfig = {
  seq_len: raw.config.seq_len,
  d_model: raw.config.d_model,
  n_heads: raw.config.n_heads,
  d_ff: raw.config.d_ff,
  n_blocks: raw.config.n_blocks,
  vocab_size: raw.config.vocab_size,
  seed: raw.config.seed,
};

/** Optimiser steps behind the trained weights. */
export const TRAINED_STEPS: number = raw.steps;

let cache: ModelWeights | null = null;

/** The trained weights, unpacked once and shared (treat as read-only). */
export function trainedWeights(): ModelWeights {
  cache ??= unpackWeights(
    raw.params as unknown as Record<string, Matrix | Vector>,
    TRAINED_CONFIG,
  );
  return cache;
}

/**
 * Can `config` run on the trained weights? Every shape must match except
 * the sequence length (positions are sinusoidal, not learned) and the seed
 * (it only seeds the random initialiser). Pass `need` to check only the
 * fields a partial computation uses (an attention-only trace ignores
 * `d_ff` and `n_blocks`, for example).
 */
export function fitsTrained(
  config: Pick<ModelConfig, "d_model"> & Partial<ModelConfig>,
  need: ReadonlyArray<"n_heads" | "d_ff" | "n_blocks" | "vocab_size"> = [
    "n_heads",
    "d_ff",
    "n_blocks",
    "vocab_size",
  ],
): boolean {
  if (config.d_model !== TRAINED_CONFIG.d_model) return false;
  return need.every((k) => config[k] === TRAINED_CONFIG[k]);
}

/**
 * The weights a widget runs: the trained ones when asked for and the
 * config fits them, otherwise the seeded initialiser for `config`.
 */
export function weightsFor(
  config: ModelConfig,
  kind: WeightKind,
): ModelWeights {
  return kind === "trained" && fitsTrained(config)
    ? trainedWeights()
    : initModelWeights(config);
}
