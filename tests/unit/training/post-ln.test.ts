/**
 * Post-LN blocks (chapter 7) in the TypeScript backward pass against
 * scripts/training/model.py (whose post-LN gradients equal PyTorch autograd,
 * checked when the fixture is written): a 4-block post-LN model at
 * initialisation and with perturbed weights, every gradient to 1e-12.
 */
import { describe, expect, it } from "vitest";

import fx from "../fixtures/training/gradients_post.json";
import { mismatch } from "../helpers/close";

import { lossAndGrads } from "@/lib/training/model-grad";
import { fromWeights, paramNames, type ParamMap } from "@/lib/training/params";
import { initModelWeights } from "@/lib/transformer/init";
import type { ModelConfig } from "@/lib/transformer/model";

const config = fx.config as ModelConfig & { norm: "post" };
const ordered = (p: ParamMap): ParamMap =>
  Object.fromEntries(
    paramNames(config).map((k) => [k, structuredClone(p[k]!)]),
  );

describe("post-LN loss and gradients = the Python reference", () => {
  for (const c of fx.cases) {
    it(c.name, () => {
      const params =
        "params" in c
          ? ordered(c.params as ParamMap)
          : fromWeights(initModelWeights(config));
      const rec: { name: string; value: number[][] }[] = [];
      const r = lossAndGrads(params, config, fx.inputs, fx.targets, rec);
      expect(mismatch(r.loss, c.loss)).toBeNull();
      expect(mismatch(r.grads, c.grads)).toBeNull();
      expect(rec.map((x) => x.name).slice(0, 6)).toEqual([
        "dlogits",
        "dxf",
        "dx_top",
        "block_3.dact",
        "block_3.dpre",
        "block_3.dh",
      ]);
    });
  }
});
