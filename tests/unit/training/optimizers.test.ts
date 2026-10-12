/**
 * Optimisers, clipping and schedules against scripts/training (whose
 * optimisers equal torch.optim to ≤ 2.2e-16 over 25 steps; results.md §3).
 */
import { describe, expect, it } from "vitest";

import fx from "../fixtures/training/optimizers.json";
import sx from "../fixtures/training/schedules.json";
import { mismatch } from "../helpers/close";

import { lossAndGrads } from "@/lib/training/model-grad";
import {
  clipByGlobalNorm,
  createOptimizer,
  globalNorm,
  optimizerStep,
  type OptimizerOptions,
} from "@/lib/training/optimizers";
import { fromWeights } from "@/lib/training/params";
import { lrAt, type Schedule } from "@/lib/training/schedules";
import { PART1_MODEL } from "@/lib/training/trainer";
import { initModelWeights } from "@/lib/transformer/init";

function options(spec: Record<string, unknown>): OptimizerOptions {
  const kind = spec["kind"] === "adam_l2" ? "adam" : (spec["kind"] as OptimizerOptions["kind"]);
  return {
    kind,
    momentum: (spec["momentum"] as number | undefined) ?? 0.9,
    weightDecay: (spec["weight_decay"] as number | undefined) ?? 0,
  };
}

function sched(s: Record<string, unknown>): Schedule {
  return {
    kind: s["kind"] as Schedule["kind"],
    lr: s["lr"] as number,
    total: s["total"] as number,
    warmup: (s["warmup"] as number | undefined) ?? 0,
    minRatio: (s["min_ratio"] as number | undefined) ?? 0.1,
    decayFrac: (s["decay_frac"] as number | undefined) ?? 0.2,
  };
}

describe("10 steps of each optimiser = the Python reference", () => {
  for (const run of fx.runs) {
    it(`${run.spec.kind}: losses, gradient norms, learning rates and final weights`, () => {
      const params = fromWeights(initModelWeights(PART1_MODEL));
      const opt = createOptimizer(params, options(run.spec));
      const losses: number[] = [];
      const norms: number[] = [];
      const lrs: number[] = [];
      fx.batches.forEach((b, t) => {
        const { loss, grads } = lossAndGrads(params, PART1_MODEL, b.inputs, b.targets);
        norms.push(clipByGlobalNorm(grads, fx.clip).norm);
        const lr = lrAt(t, sched(fx.schedule));
        optimizerStep(opt, params, grads, lr);
        losses.push(loss);
        lrs.push(lr);
      });
      expect(mismatch(losses, run.loss, 1e-11)).toBeNull();
      expect(mismatch(norms, run.gradNorm, 1e-11)).toBeNull();
      expect(mismatch(lrs, run.lr, 1e-15)).toBeNull();
      expect(mismatch(params, run.params, 1e-11)).toBeNull();
    });
  }
});

describe("clipping", () => {
  it("scales to the max norm (as clip_grad_norm_) and leaves small gradients alone", () => {
    const g = { a: [3, 4] };
    const r = clipByGlobalNorm(g, 1);
    expect(r.norm).toBe(5);
    expect(globalNorm(g)).toBeCloseTo(5 / (5 + 1e-6), 12);
    const h = { a: [[0.3, 0.4]] };
    expect(clipByGlobalNorm(h, 1).scale).toBe(1);
    expect(h.a).toEqual([[0.3, 0.4]]);
  });
});

describe("schedules = the Python reference", () => {
  for (const s of sx.schedules) {
    it(`${s.spec.kind} (lr ${s.spec.lr}, warmup ${"warmup" in s.spec ? s.spec.warmup : 0})`, () => {
      const got = Array.from({ length: s.spec.total }, (_, t) => lrAt(t, sched(s.spec)));
      expect(mismatch(got, s.lr, 1e-15)).toBeNull();
    });
  }
  it("warms up linearly, then decays to the floor", () => {
    const s: Schedule = { kind: "cosine", lr: 1, total: 100, warmup: 10 };
    expect(lrAt(0, s)).toBeCloseTo(0.1, 15);
    expect(lrAt(10, s)).toBe(1);
    expect(lrAt(99, s)).toBeGreaterThan(0.1);
    expect(lrAt(5, { kind: "constant", lr: 0.3, total: 10 })).toBe(0.3);
  });
});
