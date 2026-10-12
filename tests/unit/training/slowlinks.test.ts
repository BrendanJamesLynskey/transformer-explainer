/**
 * Training over slow links (slowlinks.ts) against scripts/training/slowlinks.py
 * (tests/unit/fixtures/training/slowlinks.json and anim_demo.json, written by
 * make_fixtures.py --groups ch13): 20 steps of each method from Part 1's
 * initial weights (validation every 10), the bytes each sends, the DCT,
 * the chunking, and two DeMo syncs from the trained model. The DCT is
 * summed in a different order from NumPy's BLAS, so 1e-9.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { fromWeights, type ParamMap } from "@/lib/training/params";
import {
  bytesFor,
  chunkLen,
  createDemo,
  dctMatrix,
  demoSync,
  keepCount,
  topkIdx,
  trainWorkers,
  DEFAULT_WORKERS,
  type Method,
} from "@/lib/training/slowlinks";
import { PART1_MODEL } from "@/lib/training/trainer";
import { initModelWeights } from "@/lib/transformer/init";
import { lossAndGrads } from "@/lib/training/model-grad";
import { sampleBatch, splitIds } from "@/lib/training/data";
import { mulberry32 } from "@/lib/transformer/random";

import dx from "../fixtures/training/anim_demo.json";
import fx from "../fixtures/training/slowlinks.json";
import { mismatch } from "../helpers/close";
import { CORPUS_IDS } from "./corpus";

const ids = CORPUS_IDS;

describe("pieces", () => {
  it("the DCT matrix = NumPy's, and is orthonormal", () => {
    const d = dctMatrix(8);
    expect(mismatch(d, fx.dct8, 1e-15)).toBeNull();
    for (let i = 0; i < 8; i++)
      for (let j = 0; j < 8; j++) {
        let s = 0;
        for (let k = 0; k < 8; k++) s += d[i]![k]! * d[j]![k]!;
        expect(s).toBeCloseTo(i === j ? 1 : 0, 12);
      }
  });
  it("chunks and kept counts = Python's", () => {
    for (const [n, c] of Object.entries(fx.chunks)) {
      expect(chunkLen(Number(n))).toBe(c.m);
      expect(keepCount(c.m)).toBe(c.k);
    }
    expect(keepCount(40)).toBe(2); // Python rounds 2.5 to even
    expect(keepCount(8)).toBe(1);
  });
  it("top-k: largest magnitudes, ties to the lower index, ascending", () => {
    expect(topkIdx([0.1, -3, 2, -2, 0.5], 3)).toEqual([1, 2, 3]);
    expect(topkIdx([1, -1, 1], 2)).toEqual([0, 1]);
  });
  it("bytes per step = Python's", () => {
    for (const m of ["dense", "topk", "demo", "diloco"] as Method[]) {
      const b = bytesFor(m, fx.sizes, 4, 10);
      expect(mismatch(b, (fx.bytes as Record<string, unknown>)[m])).toBeNull();
    }
  });
});

describe("20 steps of each method = Python's", () => {
  for (const r of fx.runs) {
    const m = r.cfg.method as Method;
    it(`${m}`, () => {
      const run = trainWorkers(
        {
          ...DEFAULT_WORKERS,
          method: m,
          lr: r.cfg.lr,
          H: r.cfg.H,
          steps: r.cfg.steps,
          dataSeed: r.cfg.data_seed,
          evalEvery: 10,
        },
        PART1_MODEL,
        ids,
      );
      expect(mismatch(run.evals, r.evals, 1e-9)).toBeNull();
      const p = run.params as ParamMap;
      expect(mismatch(p, r.params as unknown as ParamMap, 1e-9)).toBeNull();
    });
  }
});

describe("DeMo sync from the trained model (chapter 13's animation)", () => {
  const ck = JSON.parse(
    readFileSync(
      join(__dirname, "../../../public/training/default-run-weights.json"),
      "utf8",
    ),
  ) as { params: ParamMap };
  it("both syncs, every worker, = Python's", () => {
    const params = ck.params;
    const st = createDemo(params, 4);
    const { train } = splitIds(ids);
    const rng = mulberry32(dx.seed);
    for (let step = 0; step < 2; step++) {
      const b = sampleBatch(train, 16, 8, rng);
      const grads = [0, 1, 2, 3].map(
        (w) =>
          lossAndGrads(
            params,
            PART1_MODEL,
            b.inputs.slice(w * 4, w * 4 + 4),
            b.targets.slice(w * 4, w * 4 + 4),
          ).grads,
      );
      const { rec } = demoSync(st, grads, dx.tensor);
      expect(mismatch({ ...rec, step }, dx.states[step], 1e-9)).toBeNull();
    }
  });
  it("starts from Part 1's weights in the live run", () => {
    expect(fromWeights(initModelWeights(PART1_MODEL))["tok_emb"]).toBeDefined();
  });
});
