/*
 * src/lib/training/rounding.ts
 *
 * Operation: round a double to a low-precision floating-point format
 *            (FP32, BF16, FP16, FP8 E5M2, FP8 E4M3), round to nearest,
 *            ties to even; decode a format's code; the format's constants.
 * Shapes:    roundTo(x, fid) → number; encode(x, fid) → code; decode(code, fid).
 * Intuition: a format keeps m bits of mantissa: near |x| its values are
 *            2^(e − m) apart (e = floor(log2 |x|), at least the format's
 *            smallest exponent), so x lands on the nearest multiple of that
 *            spacing. Below the smallest value it rounds to zero (underflow);
 *            above the largest it becomes ∞ (or NaN, or saturates).
 * Source:    vendored from the Numerics site, numerics-explained
 *            src/lib/num/model.ts (commit ca9bbd9, MIT licence, same author),
 *            with the rounding modes and formats this site does not use
 *            removed; bit-identical to scripts/training/precision.py (the
 *            same code in Python), checked by tests/unit/training/precision.test.ts.
 * MDX:       /training/09-mixed-precision.
 */

export type FormatId = "fp32" | "bf16" | "fp16" | "e5m2" | "e4m3";

type Kind = "ieee" | "fn";

/** Sign, exponent and mantissa widths and the exponent bias of each format. */
export const FORMATS: Record<
  FormatId,
  { name: string; e: number; m: number; bias: number; kind: Kind }
> = {
  fp32: { name: "FP32", e: 8, m: 23, bias: 127, kind: "ieee" },
  bf16: { name: "BF16", e: 8, m: 7, bias: 127, kind: "ieee" },
  fp16: { name: "FP16", e: 5, m: 10, bias: 15, kind: "ieee" },
  e5m2: { name: "FP8 E5M2", e: 5, m: 2, bias: 15, kind: "ieee" },
  e4m3: { name: "FP8 E4M3", e: 4, m: 3, bias: 7, kind: "fn" },
};

export const FORMAT_ORDER: FormatId[] = [
  "fp32",
  "bf16",
  "fp16",
  "e5m2",
  "e4m3",
];

const DV = new DataView(new ArrayBuffer(8));

/** 2**k as a double, exactly. */
export function pow2(k: number): number {
  if (k > 1023) return Infinity;
  if (k < -1074) return 0;
  if (k < -1022) return pow2(k + 52) * pow2(-52);
  DV.setUint32(0, (k + 1023) * 0x100000);
  DV.setUint32(4, 0);
  return DV.getFloat64(0);
}

/** floor(log2(a)) for a finite a > 0, exactly. */
export function exponentOf(a: number): number {
  DV.setFloat64(0, a);
  const biased = (DV.getUint32(0) >>> 20) & 0x7ff;
  if (biased === 0) return exponentOf(a * pow2(64)) - 64;
  return biased - 1023;
}

function signBit(x: number): boolean {
  DV.setFloat64(0, x);
  return DV.getUint32(0) >= 0x80000000;
}

export type Info = {
  bits: number;
  e: number;
  m: number;
  bias: number;
  kind: Kind;
  emin: number;
  emax: number;
  /** largest finite value */
  max: number;
  /** smallest normal value */
  minNormal: number;
  /** smallest subnormal value */
  minSub: number;
  maxCode: number;
  infCode: number;
  nanCode: number;
  signBit: number;
};

/** A format's derived constants (all exact). */
export function info(fid: FormatId): Info {
  const { e, m, bias, kind } = FORMATS[fid];
  const top = pow2(e) - 1;
  const emin = 1 - bias;
  const ieee = kind === "ieee";
  const emax = ieee ? top - 1 - bias : top - bias;
  const maxMant = ieee ? pow2(m) - 1 : pow2(m) - 2;
  return {
    bits: 1 + e + m,
    e,
    m,
    bias,
    kind,
    emin,
    emax,
    max: pow2(emax) * (1 + maxMant / pow2(m)),
    minNormal: pow2(emin),
    minSub: pow2(emin - m),
    maxCode: (emax + bias) * pow2(m) + maxMant,
    infCode: ieee ? top * pow2(m) : -1,
    nanCode: ieee ? top * pow2(m) + pow2(m - 1) : top * pow2(m) + (pow2(m) - 1),
    signBit: pow2(e + m),
  };
}

export const INFO: Record<FormatId, Info> = {
  fp32: info("fp32"),
  bf16: info("bf16"),
  fp16: info("fp16"),
  e5m2: info("e5m2"),
  e4m3: info("e4m3"),
};

/** Sign, biased exponent and mantissa fields of a code. */
export function fields(
  code: number,
  fid: FormatId,
): { s: number; E: number; M: number } {
  const f = INFO[fid];
  const s = code >= f.signBit ? 1 : 0;
  const rest = code - s * f.signBit;
  const mm = pow2(f.m);
  return { s, E: Math.floor(rest / mm), M: rest % mm };
}

/** The value of a code (NaN for NaN codes; signed zeros and infinities). */
export function decode(code: number, fid: FormatId): number {
  const f = INFO[fid];
  const { s, E, M } = fields(code, fid);
  const top = pow2(f.e) - 1;
  if (f.kind === "ieee" && E === top) {
    if (M === 0) return s ? -Infinity : Infinity;
    return NaN;
  }
  if (f.kind === "fn" && E === top && M === pow2(f.m) - 1) return NaN;
  const v =
    E === 0 ? pow2(f.emin - f.m) * M : pow2(E - f.bias - f.m) * (pow2(f.m) + M);
  return s ? -v : v;
}

function magnitudeCode(v: number, f: Info): number {
  if (v < f.minNormal) return v / f.minSub;
  const E = exponentOf(v);
  return (E + f.bias) * pow2(f.m) + (v / pow2(E - f.m) - pow2(f.m));
}

/**
 * Round x to the format (to nearest, ties to even) and return its code.
 * `sat` saturates on overflow (OCP's SAT mode) instead of giving ∞ / NaN.
 */
export function encode(x: number, fid: FormatId, sat = false): number {
  const f = INFO[fid];
  const sign = signBit(x) ? f.signBit : 0;
  if (Number.isNaN(x)) return sign + f.nanCode;
  const a = Math.abs(x);
  if (a === Infinity) {
    if (sat) return sign + f.maxCode;
    return sign + (f.kind === "ieee" ? f.infCode : f.nanCode);
  }
  if (a === 0) return sign;
  // the spacing of the format at |x|: 2^(max(e, emin) − m)
  const q = pow2(Math.max(exponentOf(a), f.emin) - f.m);
  const t = a / q;
  let n = Math.floor(t);
  const frac = t - n;
  if (frac > 0.5 || (frac === 0.5 && n % 2 === 1)) n += 1;
  if (n === 0) return sign; // underflow: rounds to (signed) zero
  const v = n * q;
  if (v > f.max) {
    if (sat) return sign + f.maxCode;
    return sign + (f.kind === "ieee" ? f.infCode : f.nanCode);
  }
  return sign + magnitudeCode(v, f);
}

/** x rounded to the format, as a double. */
export function roundTo(x: number, fid: FormatId, sat = false): number {
  return decode(encode(x, fid, sat), fid);
}

/** The format's bit pattern of x as a string of 0s and 1s (sign, exponent, mantissa). */
export function bitString(x: number, fid: FormatId): string {
  const f = INFO[fid];
  let code = encode(x, fid);
  let out = "";
  for (let i = 0; i < f.bits; i++) {
    out = String(code % 2) + out;
    code = Math.floor(code / 2);
  }
  return out;
}
