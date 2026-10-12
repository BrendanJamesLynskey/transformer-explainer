/**
 * Chapter 9's hero: the model's real gradients, rounded. The trainer's
 * backward pass (model-grad.ts) computes all 5,376 gradients of the
 * trained model (the default run's weights) on "alice was"; each state
 * rounds every one of them to a format (rounding.ts, the Numerics site's
 * bit-exact code) after multiplying by a loss scale 2^k, and counts how many
 * vanish (round to zero), how many overflow, and the mean relative error of
 * the rest, with the histogram of |g · 2^k| by power of two.
 * First the five formats at scale 1; then FP16 with growing loss scales.
 * scripts/training/anim_b.py:rounding_states builds the same list.
 */
import { encodeChars } from "@/lib/training/data";
import { lossAndGrads } from "@/lib/training/model-grad";
import { paramNames, type ParamMap } from "@/lib/training/params";
import {
  INFO,
  exponentOf,
  pow2,
  roundTo,
  type FormatId,
} from "@/lib/training/rounding";
import { PART1_MODEL } from "@/lib/training/trainer";

export const GRAD_TEXT = "alice was";
export const HIST_LO = -48;
export const HIST_HI = 16;
export const FP16_SCALES = [0, 4, 8, 12, 16, 20, 24];
export const FORMATS_SHOWN: FormatId[] = [
  "fp32",
  "bf16",
  "fp16",
  "e5m2",
  "e4m3",
];

export type RoundingState = {
  fmt: FormatId;
  scaleExp: number;
  /** count of nonzero |g 2^k| in [2^b, 2^(b+1)), b = HIST_LO … HIST_HI − 1 (ends clamp) */
  bins: number[];
  nonzero: number;
  /** nonzero gradients that round to 0 */
  zero: number;
  /** gradients that round to ±∞ (or NaN in E4M3) */
  overflow: number;
  meanRelErr: number;
  minSub: number;
  minNormal: number;
  max: number;
};

/** Every gradient of the model on the window, flattened in parameter order. */
export function flatGrads(params: ParamMap): number[] {
  const ids = encodeChars(GRAD_TEXT);
  const { grads } = lossAndGrads(
    params,
    PART1_MODEL,
    [ids.slice(0, -1)],
    [ids.slice(1)],
  );
  const out: number[] = [];
  for (const k of paramNames(PART1_MODEL)) {
    const g = grads[k]!;
    for (const r of g) {
      if (Array.isArray(r)) for (const x of r) out.push(x);
      else out.push(r);
    }
  }
  return out;
}

export function roundingState(
  g: number[],
  fmt: FormatId,
  k: number,
): RoundingState {
  const scale = pow2(k);
  const bins = new Array<number>(HIST_HI - HIST_LO).fill(0);
  let zero = 0;
  let overflow = 0;
  let err = 0;
  let nOk = 0;
  let nonzero = 0;
  for (const v of g) {
    if (v === 0) continue;
    nonzero++;
    const s = v * scale;
    const b = exponentOf(Math.abs(s));
    bins[Math.min(Math.max(b, HIST_LO), HIST_HI - 1) - HIST_LO]! += 1;
    const r = roundTo(s, fmt);
    if (r === 0) zero++;
    else if (!Number.isFinite(r)) overflow++;
    else {
      err += Math.abs(r - s) / Math.abs(s);
      nOk++;
    }
  }
  const f = INFO[fmt];
  return {
    fmt,
    scaleExp: k,
    bins,
    nonzero,
    zero,
    overflow,
    meanRelErr: nOk ? err / nOk : 0,
    minSub: f.minSub,
    minNormal: f.minNormal,
    max: f.max,
  };
}

export function roundingStates(params: ParamMap): RoundingState[] {
  const g = flatGrads(params);
  return [
    ...FORMATS_SHOWN.map((f) => roundingState(g, f, 0)),
    ...FP16_SCALES.slice(1).map((k) => roundingState(g, "fp16", k)),
  ];
}

const NAME: Record<FormatId, string> = {
  fp32: "FP32",
  bf16: "BF16",
  fp16: "FP16",
  e5m2: "FP8 E5M2",
  e4m3: "FP8 E4M3",
};

export function roundingCaption(s: RoundingState): string {
  const pct = (x: number) => `${((100 * x) / s.nonzero).toFixed(1)}%`;
  const scale = s.scaleExp ? ` × loss scale 2^${s.scaleExp}` : "";
  const over = s.overflow ? `, ${s.overflow} overflow` : "";
  return `${NAME[s.fmt]}${scale}: ${s.zero} of ${s.nonzero} gradients (${pct(s.zero)}) round to zero${over}; the rest are off by ${(100 * s.meanRelErr).toPrecision(2)}% on average.`;
}
