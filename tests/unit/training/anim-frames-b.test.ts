/**
 * Frame tests for chapters 6-10's animations (visual standard §4): every
 * state the site builds equals the state scripts/training/anim_b.py builds
 * from the reference (1e-12, relative for values above 1), and the caption
 * of each TS state equals the caption of Python's.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import results from "@/data/training/results.json";

import ax from "../fixtures/training/anim_attention.json";
import dx from "../fixtures/training/anim_depth.json";
import rx from "../fixtures/training/anim_rounding.json";
import sx from "../fixtures/training/anim_schedules.json";
import { mismatch } from "../helpers/close";

import {
  attentionGrowthCaption,
  attentionGrowthStates,
  type AttentionGrowthState,
  type Snapshots,
} from "@/lib/anim/training/attention-growth-steps";
import {
  DEPTH_STDS,
  depthCaption,
  depthStates,
  type DepthState,
} from "@/lib/anim/training/depth-steps";
import { memoryCaption, memoryStates } from "@/lib/anim/training/memory-steps";
import {
  roundingCaption,
  roundingStates,
  type RoundingState,
} from "@/lib/anim/training/rounding-steps";
import {
  scheduleCaption,
  scheduleStates,
  type ScheduleRow,
  type ScheduleState,
} from "@/lib/anim/training/schedule-steps";
import { LLAMA_7B, MIXED_ADAM, PART1_MEM } from "@/lib/training/memory";
import { paramNames, type ParamMap } from "@/lib/training/params";
import { PART1_MODEL } from "@/lib/training/trainer";

const pub = (f: string) =>
  JSON.parse(
    readFileSync(join(__dirname, "../../../public/training", f), "utf8"),
  ) as unknown;

const ordered = (p: ParamMap): ParamMap =>
  Object.fromEntries(paramNames(PART1_MODEL).map((k) => [k, p[k]!]));

describe("chapter 6: schedules", () => {
  const rows = results.schedules.rows as unknown as ScheduleRow[];
  const states = scheduleStates(rows);
  const names = rows.map((r) => r.name);
  it("every state = Python's (lr from schedules.ts, the measured curves)", () => {
    expect(mismatch(states, sx.states, 1e-15)).toBeNull();
  });
  it("captions match", () => {
    states.forEach((s, i) =>
      expect(scheduleCaption(s, names)).toBe(
        scheduleCaption(sx.states[i] as ScheduleState, names),
      ),
    );
  });
  it("WSD holds the peak until its decay; cosine has decayed by the end", () => {
    const wsd = names.indexOf("WSD");
    const cos = names.indexOf("warmup + cosine");
    const mid = states.find((s) => s.step === 1000)!;
    expect(mid.lrs[wsd]).toBe(0.01);
    expect(states.at(-1)!.lrs[cos]).toBeCloseTo(0.001, 6);
  });
});

describe("chapter 7: scale through depth", () => {
  for (const std of DEPTH_STDS) {
    it(`std ${std}: every state = Python's, captions match`, () => {
      const states = depthStates(std);
      const want = (dx.stds as Record<string, unknown[]>)[String(std)]!;
      expect(mismatch(states, want, 1e-12)).toBeNull();
      states.forEach((s, i) =>
        expect(depthCaption(s)).toBe(depthCaption(want[i] as DepthState)),
      );
    });
  }
  it("post-LN holds the stream at RMS 1; at std 0.25 pre-LN's grows with depth", () => {
    const s = depthStates(0.25).at(-1)!;
    for (const v of s.rms[1].slice(1)) expect(v).toBeCloseTo(1, 3);
    expect(s.rms[0].at(-1)!).toBeGreaterThan(3 * s.rms[0][0]!);
  });
});

describe("chapter 8: attention during the high-learning-rate run", () => {
  const raw = pub("attention-snapshots.json") as Snapshots;
  const snap: Snapshots = {
    steps: raw.steps,
    params: Object.fromEntries(
      Object.entries(raw.params).map(([k, p]) => [k, ordered(p)]),
    ),
  };
  const states = attentionGrowthStates(
    snap,
    results.attnRun.evals as { step: number; val: number }[],
  );
  it("every state = Python's, captions match", () => {
    expect(mismatch(states, ax.states, 1e-12)).toBeNull();
    states.forEach((s, i) =>
      expect(attentionGrowthCaption(s)).toBe(
        attentionGrowthCaption(ax.states[i] as AttentionGrowthState),
      ),
    );
  });
  it("the largest score grows by orders of magnitude and the entropy falls", () => {
    expect(states.at(-1)!.maxScore).toBeGreaterThan(20 * states[1]!.maxScore);
    expect(states.at(-1)!.entropy).toBeLessThan(states[1]!.entropy);
  });
});

describe("chapter 9: rounding the model's gradients", () => {
  const ck = pub("default-run-weights.json") as { params: ParamMap };
  const states = roundingStates(ordered(ck.params));
  it("every state = Python's, captions match", () => {
    expect(mismatch(states, rx.states, 1e-12)).toBeNull();
    states.forEach((s, i) =>
      expect(roundingCaption(s)).toBe(
        roundingCaption(rx.states[i] as RoundingState),
      ),
    );
  });
  it("FP32 and BF16 lose nothing to underflow; loss scaling rescues FP16", () => {
    const at = (f: string, k: number) =>
      states.find((s) => s.fmt === f && s.scaleExp === k)!;
    expect(at("fp32", 0).zero).toBe(0);
    expect(at("bf16", 0).zero).toBe(0);
    expect(at("fp16", 0).zero).toBeGreaterThan(0);
    expect(at("fp16", 16).zero).toBe(0);
    expect(at("fp16", 24).overflow).toBeGreaterThan(0);
  });
});

describe("chapter 10: memory through one step", () => {
  it("captions are defined for every event", () => {
    for (const arch of [LLAMA_7B, PART1_MEM])
      for (const r of ["none", "selective", "full"] as const)
        for (const e of memoryStates(arch, MIXED_ADAM, r))
          expect(memoryCaption(e, arch).length).toBeGreaterThan(20);
  });
});
