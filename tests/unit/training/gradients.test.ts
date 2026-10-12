/**
 * The TypeScript backward pass against scripts/training/model.py (whose
 * gradients equal PyTorch autograd to ~1e-16 and central differences to
 * ~1e-9; see results.md §3). Tolerance here: 1e-12, relative to max(1, |x|).
 */
import { describe, expect, it } from "vitest";

import fx from "../fixtures/training/gradients.json";
import { mismatch } from "../helpers/close";

import {
  forwardCached,
  lossAndGrads,
  lossOnly,
} from "@/lib/training/model-grad";
import { fromWeights, paramNames, type ParamMap } from "@/lib/training/params";
import { initModelWeights } from "@/lib/transformer/init";
import { forwardTyped } from "@/lib/transformer/model";

import { PART1_MODEL } from "@/lib/training/trainer";

const init = () => fromWeights(initModelWeights(PART1_MODEL));
const ordered = (p: ParamMap): ParamMap =>
  Object.fromEntries(
    paramNames(PART1_MODEL).map((k) => [k, structuredClone(p[k]!)]),
  );

describe("forwardCached is Part 1's forward pass", () => {
  it("gives exactly forwardTyped's logits", () => {
    const ids = fx.inputs[0]!;
    const w = initModelWeights(PART1_MODEL);
    const want = forwardTyped(ids, PART1_MODEL, w).logits;
    expect(forwardCached(fromWeights(w), PART1_MODEL, ids).logits).toEqual(
      want,
    );
  });
});

describe("loss and gradients = the Python reference", () => {
  for (const c of fx.cases) {
    it(`${c.name}: loss, per-position losses and every gradient`, () => {
      // the fixture's keys are sorted; the trainer walks paramNames order
      const params: ParamMap =
        "params" in c ? ordered(c.params as ParamMap) : init();
      const r = lossAndGrads(params, PART1_MODEL, fx.inputs, fx.targets);
      expect(mismatch(r.loss, c.loss)).toBeNull();
      expect(mismatch(r.perPosition, c.perPosition)).toBeNull();
      expect(Object.keys(r.grads)).toEqual(paramNames(PART1_MODEL));
      expect(mismatch(r.grads, c.grads)).toBeNull();
      expect(
        Math.abs(lossOnly(params, PART1_MODEL, fx.inputs, fx.targets) - r.loss),
      ).toBe(0);
    });
  }

  it("records the gradient at every stop on the way back, as Python does", () => {
    const rec: { name: string; value: number[][] }[] = [];
    lossAndGrads(init(), PART1_MODEL, fx.inputs, fx.targets, rec);
    expect(rec.map((r) => r.name)).toEqual(fx.record.map((r) => r.name));
    expect(mismatch(rec, fx.record)).toBeNull();
  });
});
