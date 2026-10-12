/*
 * src/lib/training/trainer.ts
 *
 * Operation: the training loop: draw a batch, compute the loss and its
 *            gradients, clip, step the optimiser at the scheduled learning
 *            rate; now and then, score the validation text and sample.
 * Shapes:    createTrainer(config, corpusIds) → Trainer;
 *            trainStep(tr) → StepLog; evaluate(tr) → validation loss.
 * Intuition: this is all training is, at any scale: the same four lines
 *            (batch, loss.backward(), clip, step) repeated. What changes at
 *            scale is how big each part is and how many machines share it.
 *            The model starts as Part 1's model, from Part 1's initialiser
 *            (init.ts: mulberry32 + Box-Muller), so the lessons' random
 *            weights are literally the ones being trained here.
 *            Matches scripts/training/make_fixtures.py:train_run step for
 *            step (tests/unit/training/trainer.test.ts).
 * MDX:       /training/01-training-loop.
 */
import { initModelWeights } from "@/lib/transformer/init";
import type { ModelConfig } from "@/lib/transformer/model";
import { mulberry32 } from "@/lib/transformer/random";
import { sample } from "@/lib/transformer/sampling";
import { ALPHABET } from "@/lib/transformer/tokenizer";

import { sampleBatch, splitIds, valBatch, type Batch } from "./data";
import { forwardCached, lossAndGrads, lossOnly } from "./model-grad";
import {
  clipByGlobalNorm,
  createOptimizer,
  optimizerStep,
  type OptimizerOptions,
  type OptimizerState,
} from "./optimizers";
import { cloneParams, fromWeights, type ParamMap } from "./params";
import { lrAt, type Schedule } from "./schedules";

/** Everything that defines a training run. */
export type TrainConfig = {
  model: ModelConfig;
  batch: number;
  steps: number;
  optimizer: OptimizerOptions;
  /** The schedule's `total` is the run's `steps`. */
  schedule: Omit<Schedule, "total">;
  /** Max global gradient norm (null: no clipping). */
  clip: number | null;
  /** Seed for the batch draws (the model's seed is `model.seed`). */
  dataSeed: number;
};

/** Part 1's model (SPEC §4), seed 42. */
export const PART1_MODEL: ModelConfig = {
  seq_len: 8,
  d_model: 16,
  n_heads: 2,
  d_ff: 32,
  n_blocks: 2,
  vocab_size: 64,
  seed: 42,
};

/** The default run: AdamW, warmup 50 steps + cosine, 2,000 steps of 16 windows. */
export const DEFAULT_TRAIN: TrainConfig = {
  model: PART1_MODEL,
  batch: 16,
  steps: 2000,
  optimizer: { kind: "adamw", beta1: 0.9, beta2: 0.99, eps: 1e-8, weightDecay: 0.01 },
  schedule: { kind: "cosine", lr: 0.01, warmup: 50, minRatio: 0.1 },
  clip: 1.0,
  dataSeed: 1,
};

export type Trainer = {
  cfg: TrainConfig;
  params: ParamMap;
  opt: OptimizerState;
  rng: () => number;
  train: number[];
  val: number[];
  valSet: Batch;
  /** steps taken so far */
  t: number;
};

export type StepLog = {
  step: number;
  lr: number;
  loss: number;
  gradNorm: number;
  clipScale: number;
};

/** A trainer at step 0: Part 1's initial weights, a fresh optimiser. */
export function createTrainer(cfg: TrainConfig, corpusIds: number[]): Trainer {
  const params = fromWeights(initModelWeights(cfg.model));
  const { train, val } = splitIds(corpusIds);
  return {
    cfg,
    params,
    opt: createOptimizer(params, cfg.optimizer),
    rng: mulberry32(cfg.dataSeed),
    train,
    val,
    valSet: valBatch(val, cfg.model.seq_len),
    t: 0,
  };
}

/** One training step; returns what it measured. */
export function trainStep(tr: Trainer, batch?: Batch): StepLog {
  const { cfg } = tr;
  const b = batch ?? sampleBatch(tr.train, cfg.batch, cfg.model.seq_len, tr.rng);
  const { loss, grads } = lossAndGrads(tr.params, cfg.model, b.inputs, b.targets);
  const { norm, scale } =
    cfg.clip === null
      ? { norm: NaN, scale: 1 }
      : clipByGlobalNorm(grads, cfg.clip);
  const lr = lrAt(tr.t, { ...cfg.schedule, total: cfg.steps });
  optimizerStep(tr.opt, tr.params, grads, lr);
  const log = { step: tr.t, lr, loss, gradNorm: norm, clipScale: scale };
  tr.t += 1;
  return log;
}

/** Mean loss on the fixed validation windows. */
export function evaluate(tr: Trainer): number {
  return lossOnly(tr.params, tr.cfg.model, tr.valSet.inputs, tr.valSet.targets);
}

/** A snapshot of the weights (e.g. to hand to an animation). */
export function snapshot(tr: Trainer): ParamMap {
  return cloneParams(tr.params);
}

/**
 * Generate `n` characters after `prompt` at temperature τ with a seeded
 * sampler (sampling.ts), reading at most the last seq_len characters.
 */
export function sampleText(
  params: ParamMap,
  model: ModelConfig,
  promptIds: number[],
  n: number,
  temperature: number,
  seed: number,
): string {
  const rng = mulberry32(seed);
  const ids = promptIds.slice();
  let out = "";
  for (let k = 0; k < n; k++) {
    const ctx = ids.slice(-model.seq_len);
    const { logits } = forwardCached(params, model, ctx);
    const next = sample(logits[logits.length - 1]!, { kind: "temperature", temperature }, rng);
    ids.push(next);
    out += ALPHABET[next]!;
  }
  return out;
}
