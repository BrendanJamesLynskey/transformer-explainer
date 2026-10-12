/**
 * The rounding code (src/lib/training/rounding.ts, vendored from the
 * Numerics site) against scripts/training/precision.py on random values over
 * 50 binades, every FP8 value and the midpoint between each pair of
 * neighbours (the ties), and the overflow and underflow edges: identical
 * doubles, with and without saturation, in every format.
 */
import { describe, expect, it } from "vitest";

import fx from "../fixtures/training/rounding.json";

import {
  FORMAT_ORDER,
  INFO,
  bitString,
  decode,
  encode,
  roundTo,
} from "@/lib/training/rounding";

const dec = (v: number | string): number =>
  v === "inf"
    ? Infinity
    : v === "-inf"
      ? -Infinity
      : v === "nan"
        ? NaN
        : Number(v);

describe("roundTo = the Python reference, bit for bit", () => {
  for (const fid of FORMAT_ORDER) {
    for (const sat of [false, true]) {
      it(`${fid}${sat ? " (saturating)" : ""}`, () => {
        const want = (fx.rounded as Record<string, (number | string)[]>)[
          sat ? `${fid}_sat` : fid
        ]!;
        fx.xs.forEach((x, i) => {
          const w = dec(want[i]!);
          const g = roundTo(x, fid, sat);
          if (Number.isNaN(w)) expect(Number.isNaN(g)).toBe(true);
          else expect(g).toBe(w);
        });
      });
    }
  }
});

describe("format constants", () => {
  it("match the specifications", () => {
    expect(INFO.fp16.max).toBe(65504);
    expect(INFO.bf16.max).toBeCloseTo(3.3895313892515355e38, -30);
    expect(INFO.e4m3.max).toBe(448);
    expect(INFO.e5m2.max).toBe(57344);
    expect(INFO.fp16.minSub).toBe(2 ** -24);
    expect(INFO.e4m3.minSub).toBe(2 ** -9);
    expect(INFO.e5m2.minSub).toBe(2 ** -16);
  });
  it("encodes and decodes round trip on every FP8 code", () => {
    for (const fid of ["e4m3", "e5m2"] as const) {
      for (let c = 0; c <= INFO[fid].maxCode; c++) {
        expect(encode(decode(c, fid), fid)).toBe(c);
      }
    }
  });
  it("prints bit patterns", () => {
    expect(bitString(1, "bf16")).toBe("0011111110000000");
    expect(bitString(-2, "e4m3")).toBe("11000000");
    expect(bitString(1e9, "fp16")).toBe("0111110000000000"); // overflow: +inf
    expect(roundTo(1e9, "e4m3")).toBeNaN();
    expect(roundTo(1e9, "e4m3", true)).toBe(448);
    expect(roundTo(Infinity, "fp16")).toBe(Infinity);
    expect(roundTo(-Infinity, "fp16", true)).toBe(-65504);
    expect(roundTo(-Infinity, "e4m3")).toBeNaN();
    expect(roundTo(NaN, "bf16")).toBeNaN();
    expect(Object.is(roundTo(-1e-30, "fp16"), -0)).toBe(true);
  });
});
