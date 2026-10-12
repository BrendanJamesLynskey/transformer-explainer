/**
 * The memory calculator (src/lib/training/memory.ts) against
 * scripts/training/memory.py: the breakdowns results.md §10 quotes and the
 * timeline chapter 10's animation draws, exactly; and the parameter counts
 * against the published LLaMA 7B (6.7B) and Part 1's model (5,376).
 */
import { describe, expect, it } from "vitest";

import results from "@/data/training/results.json";

import fx from "../fixtures/training/anim_memory.json";
import { mismatch } from "../helpers/close";

import {
  FP32_ADAM,
  LLAMA_7B,
  MIXED_ADAM,
  PART1_MEM,
  breakdown,
  fmtBytes,
  paramCount,
  timeline,
  type Arch,
  type Recompute,
} from "@/lib/training/memory";

describe("parameter counts", () => {
  it("LLaMA 7B: 6,738,415,616 (Touvron et al. 2023: 6.7B)", () => {
    expect(paramCount(LLAMA_7B)).toBe(6_738_415_616);
  });
  it("Part 1's model: 5,376, as the trainer's parameters", () => {
    expect(paramCount(PART1_MEM)).toBe(results.model.params);
  });
});

describe("breakdowns = results.json (memory.py)", () => {
  const m = results.memory;
  it("LLaMA 7B, mixed precision, each recomputation", () => {
    for (const r of ["none", "selective", "full"] as Recompute[])
      expect(
        mismatch(breakdown(LLAMA_7B, MIXED_ADAM, r), m.llama.mixed[r], 0),
      ).toBeNull();
  });
  it("LLaMA 7B, FP32 Adam; Part 1's model", () => {
    expect(
      mismatch(breakdown(LLAMA_7B, FP32_ADAM), m.llama.fp32, 0),
    ).toBeNull();
    expect(
      mismatch(breakdown(PART1_MEM, MIXED_ADAM), m.part1.mixed, 0),
    ).toBeNull();
  });
  it("mixed-precision Adam is 16 bytes per parameter; FP32 Adam too", () => {
    expect(breakdown(LLAMA_7B, MIXED_ADAM).bytesPerParam).toBe(16);
    expect(breakdown(LLAMA_7B, FP32_ADAM).bytesPerParam).toBe(16);
    expect(
      breakdown(LLAMA_7B, { ...MIXED_ADAM, optimizer: "sgd" }).bytesPerParam,
    ).toBe(8);
  });
});

describe("timelines = Python's, exactly", () => {
  for (const c of fx.cases) {
    it(`${c.arch.name}, recompute ${c.recompute}`, () => {
      const got = timeline(
        c.arch as Arch,
        MIXED_ADAM,
        c.recompute as Recompute,
      );
      expect(mismatch(got, c.states, 0)).toBeNull();
    });
  }
  it("recomputation lowers the peak", () => {
    const peak = (r: Recompute) =>
      Math.max(...timeline(LLAMA_7B, MIXED_ADAM, r).map((e) => e.total));
    expect(peak("full")).toBeLessThan(peak("selective"));
    expect(peak("selective")).toBeLessThan(peak("none"));
  });
});

describe("fmtBytes", () => {
  it("uses decimal units", () => {
    expect(fmtBytes(512)).toBe("512 B");
    expect(fmtBytes(1500)).toBe("1.5 kB");
    expect(fmtBytes(212.75e9)).toBe("213 GB");
    expect(fmtBytes(3.2e12)).toBe("3.2 TB");
  });
});
