/**
 * Frame tests for chapters 11-16's animations (visual standard §4): the
 * states the site builds equal the states scripts/training/anim_c.py (and
 * slowlinks.py) builds from the reference, and every caption built from a
 * TS state equals the caption built from Python's.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import results from "@/data/training/results.json";
import {
  DEMO_STAGES,
  demoCaption,
  demoStates,
} from "@/lib/anim/training/demo-steps";
import {
  goodputCaption,
  goodputStates,
} from "@/lib/anim/training/goodput-steps";
import {
  isoflopCaption,
  isoflopStates,
  type IsoRow,
  type IsoState,
} from "@/lib/anim/training/isoflop-steps";
import {
  pipelineCaption,
  pipelineStates,
} from "@/lib/anim/training/pipeline-steps";
import {
  completeChunks,
  ringCaption,
  ringStates,
  type RingState,
} from "@/lib/anim/training/ring-steps";
import { sftCaption, sftData, sftStates } from "@/lib/anim/training/sft-steps";
import { lossAndGrads } from "@/lib/training/model-grad";
import { paramNames, type ParamMap } from "@/lib/training/params";
import type { DemoRecord } from "@/lib/training/slowlinks";
import {
  pipeline,
  simulateGoodput,
  type PipeKind,
  type PipeResult,
  type Segment,
} from "@/lib/training/systems";
import { PART1_MODEL } from "@/lib/training/trainer";
import { sampleBatch, splitIds } from "@/lib/training/data";
import { mulberry32 } from "@/lib/transformer/random";

import dx from "../fixtures/training/anim_demo.json";
import gx from "../fixtures/training/anim_goodput.json";
import ix from "../fixtures/training/anim_isoflop.json";
import px from "../fixtures/training/anim_pipeline.json";
import rx from "../fixtures/training/anim_ring.json";
import { mismatch } from "../helpers/close";
import { CORPUS_IDS } from "./corpus";

const trained = (() => {
  const ck = JSON.parse(
    readFileSync(
      join(__dirname, "../../../public/training/default-run-weights.json"),
      "utf8",
    ),
  ) as { params: ParamMap };
  return Object.fromEntries(
    paramNames(PART1_MODEL).map((k) => [k, ck.params[k]!]),
  ) as ParamMap;
})();

describe("chapter 11: ring all-reduce of real gradients", () => {
  // the four workers' gradients, computed here by the TS trainer
  const { train } = splitIds(CORPUS_IDS);
  const b = sampleBatch(train, 16, 8, mulberry32(rx.seed));
  const grads = [0, 1, 2, 3].map((w) => {
    const g = lossAndGrads(
      trained,
      PART1_MODEL,
      b.inputs.slice(w * 4, w * 4 + 4),
      b.targets.slice(w * 4, w * 4 + 4),
    ).grads;
    return g[rx.tensor] as number[];
  });
  const states = ringStates(grads);
  it("the workers' gradients = Python's", () => {
    expect(mismatch(grads, rx.grads, 1e-12)).toBeNull();
  });
  it("every state = Python's; captions match", () => {
    expect(mismatch(states, rx.states, 1e-12)).toBeNull();
    states.forEach((s, i) =>
      expect(ringCaption(s, 4, 16)).toBe(
        ringCaption(rx.states[i] as RingState, 4, 16),
      ),
    );
  });
  it("the finished chunks: one per worker after reduce-scatter, all at the end", () => {
    const rs = completeChunks(states[3]!, 4);
    rs.forEach((row) => expect(row.filter(Boolean)).toHaveLength(1));
    completeChunks(states.at(-1)!, 4).forEach((row) =>
      expect(row.every(Boolean)).toBe(true),
    );
    // every rank's chunk it marks finished equals the sum there
    states.forEach((s) => {
      const done = completeChunks(s, 4);
      done.forEach((row, r) =>
        row.forEach((ok, c) => {
          if (!ok) return;
          for (let i = c * 4; i < c * 4 + 4; i++)
            expect(s.bufs[r]![i]).toBeCloseTo(rx.sum[i]!, 12);
        }),
      );
    });
  });
});

describe("chapter 12: pipeline schedules", () => {
  for (const kind of ["gpipe", "1f1b", "interleaved"] as const) {
    const py = (px as Record<string, unknown>)[kind] as PipeResult;
    const ts = pipeline(kind as PipeKind, py.p, py.m, py.v);
    it(`${kind}: states and captions from the TS simulation = from Python's`, () => {
      const a = pipelineStates(ts);
      const b = pipelineStates(py);
      expect(mismatch(a, b, 1e-12)).toBeNull();
      a.forEach((s, i) =>
        expect(pipelineCaption(s, ts)).toBe(pipelineCaption(b[i]!, py)),
      );
      expect(a.at(-1)!.done).toBe(ts.items.length);
    });
  }
});

describe("chapter 13: DeMo syncs", () => {
  const recs = dx.states as unknown as DemoRecord[];
  const states = demoStates(recs);
  it("two syncs × every stage, captions built from Python's records", () => {
    expect(states).toHaveLength(2 * DEMO_STAGES.length);
    for (const s of states)
      expect(demoCaption(s, recs).length).toBeGreaterThan(20);
    // error feedback: the second sync's momentum is not just its gradient
    const w = recs[1]!.workers[0]!;
    const differs = w.momentum.some((m, i) => Math.abs(m - w.grad[i]!) > 1e-9);
    expect(differs).toBe(true);
    // what a worker keeps plus what it sends is its momentum (orthonormal DCT)
    const w0 = recs[0]!.workers[0]!;
    let a = 0;
    let b = 0;
    let c = 0;
    w0.momentum.forEach((x) => (a += x * x));
    w0.residual.forEach((x) => (b += x * x));
    w0.sent.forEach((x) => (c += x * x));
    expect(b + c).toBeCloseTo(a, 12);
  });
});

describe("chapter 14: IsoFLOP sweep", () => {
  const iso = results.isoflop;
  const rows = iso.rows as unknown as IsoRow[];
  const states = isoflopStates(rows, {
    a: iso.a,
    intercept: iso.frontier.intercept,
  });
  it("every state = Python's (the fits recomputed here)", () => {
    expect(mismatch(states, ix.states, 1e-9)).toBeNull();
  });
  it("captions match", () => {
    states.forEach((s, i) =>
      expect(isoflopCaption(s, rows)).toBe(
        isoflopCaption(ix.states[i] as IsoState, rows),
      ),
    );
  });
});

describe("chapter 15: a week of failures and checkpoints", () => {
  const g = gx.params;
  const sim = simulateGoodput(
    g.tau,
    g.delta,
    g.mtbf,
    g.restart,
    g.horizon,
    g.seed,
  );
  it("the simulation = Python's", () => {
    expect(mismatch(sim, gx.sim, 1e-9)).toBeNull();
  });
  it("states and captions from it = from Python's segments", () => {
    const a = goodputStates(sim.segments, g.horizon);
    const b = goodputStates(gx.sim.segments as Segment[], g.horizon);
    expect(mismatch(a, b, 1e-9)).toBeNull();
    a.forEach((s, i) => expect(goodputCaption(s)).toBe(goodputCaption(b[i]!)));
    expect(a.at(-1)!.failures).toBe(gx.sim.failures);
  });
});

describe("chapter 16: SFT and preferences on the trained model", () => {
  const d = sftData(trained);
  const py = results.sft;
  it("every log probability = Python's", () => {
    expect(mismatch(d.promptTokens, py.promptTokens, 1e-12)).toBeNull();
    expect(mismatch(d.responseTokens, py.responseTokens, 1e-12)).toBeNull();
    expect(mismatch(d.chosenTokens, py.pref.chosenTokens, 1e-12)).toBeNull();
    expect(
      mismatch(d.rejectedTokens, py.pref.rejectedTokens, 1e-12),
    ).toBeNull();
  });
  it("the states end at the SFT loss and the two sums", () => {
    const st = sftStates(d);
    const lastResp = st.filter((s) => s.kind === "response").at(-1)!;
    expect(lastResp.kind === "response" && lastResp.sftLoss).toBeCloseTo(
      py.sftLoss,
      12,
    );
    const cmp = st.at(-1)!;
    expect(cmp.kind).toBe("compare");
    if (cmp.kind === "compare") {
      expect(cmp.chosen).toBeCloseTo(py.pref.chosenLogp, 12);
      expect(cmp.rejected).toBeCloseTo(py.pref.rejectedLogp, 12);
    }
    st.forEach((s) => expect(sftCaption(s, d).length).toBeGreaterThan(20));
  });
});
