/**
 * The first 200 steps of the default run = scripts/training (losses,
 * gradient norms, learning rates, validation, weights, a sample). Float
 * order differs a little between NumPy and this code, and training
 * amplifies it; the tolerance is 1e-9 relative.
 */
import { describe, expect, it } from "vitest";

import tx from "../fixtures/training/train_run.json";
import { mismatch } from "../helpers/close";

import { CORPUS_IDS } from "./corpus";

import {
  DEFAULT_TRAIN,
  createTrainer,
  evaluate,
  sampleText,
  snapshot,
  trainStep,
} from "@/lib/training/trainer";
import { encodeChars } from "@/lib/training/data";

describe("the default run, first 200 steps", () => {
  it("matches the Python run step for step", () => {
    const tr = createTrainer({ ...DEFAULT_TRAIN, steps: tx.config.steps }, CORPUS_IDS);
    const loss: number[] = [];
    const norms: number[] = [];
    const lrs: number[] = [];
    const evals: { step: number; val: number }[] = [];
    for (let t = 0; t < tx.config.steps; t++) {
      if (t % 100 === 0) evals.push({ step: t, val: evaluate(tr) });
      const s = trainStep(tr);
      loss.push(s.loss);
      norms.push(s.gradNorm);
      lrs.push(s.lr);
    }
    evals.push({ step: tx.config.steps, val: evaluate(tr) });
    expect(mismatch(loss, tx.loss, 1e-9)).toBeNull();
    expect(mismatch(norms, tx.gradNorm, 1e-9)).toBeNull();
    expect(mismatch(lrs, tx.lr, 1e-15)).toBeNull();
    expect(mismatch(evals, tx.evals, 1e-9)).toBeNull();
    const p = snapshot(tr);
    expect(mismatch((p["tok_emb"] as number[][])[0], tx.tok_emb_row0, 1e-9)).toBeNull();
    expect(
      mismatch((p["block_1.ffn.W2"] as number[][]).at(-1), tx.ffn_W2_last, 1e-9),
    ).toBeNull();
    expect(sampleText(p, DEFAULT_TRAIN.model, encodeChars("alice "), 40, 0.8, 7)).toBe(tx.sample);
  }, 60_000);

  it("can run without clipping", () => {
    const tr = createTrainer({ ...DEFAULT_TRAIN, clip: null, steps: 2, batch: 2 }, CORPUS_IDS);
    const s = trainStep(tr);
    expect(Number.isNaN(s.gradNorm)).toBe(true);
    expect(s.clipScale).toBe(1);
  });
});
