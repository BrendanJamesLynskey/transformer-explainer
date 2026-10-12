"use client";

/**
 * Chapter 9's hero: the model's own gradients, rounded. The trainer's
 * backward pass computes all 5,376 gradients of the trained model on
 * "alice was" in the browser; each step rounds them to a format (the
 * Numerics site's bit-exact rounding, vendored) after a loss scale 2^k. The
 * histogram is |g · 2^k| by power of two; the shaded bands are where the
 * format cannot follow: below its smallest value everything becomes 0,
 * above its largest it overflows.
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import {
  PHONE_BELOW,
  useContainerWidth,
} from "@/components/viz/useContainerWidth";
import { useSvgFont } from "@/components/viz/useSvgFont";
import {
  HIST_HI,
  HIST_LO,
  flatGrads,
  roundingCaption,
  roundingState,
  FORMATS_SHOWN,
  FP16_SCALES,
} from "@/lib/anim/training/rounding-steps";
import {
  FORMATS,
  bitString,
  exponentOf,
  pow2,
  roundTo,
} from "@/lib/training/rounding";
import { OKABE_ITO } from "@/lib/viz/palette";

import { useTrainedWeights } from "./useTrainingData";

export default function RoundingAnimation({
  children,
  testId = "rounding-animation",
}: {
  children?: ReactNode;
  testId?: string;
}): JSX.Element {
  const trained = useTrainedWeights(true);
  const grads = useMemo(
    () => (trained && !("error" in trained) ? flatGrads(trained) : null),
    [trained],
  );
  const states = useMemo(
    () =>
      grads
        ? [
            ...FORMATS_SHOWN.map((f) => roundingState(grads, f, 0)),
            ...FP16_SCALES.slice(1).map((k) => roundingState(grads, "fp16", k)),
          ]
        : [],
    [grads],
  );
  const st = useStepper(Math.max(1, states.length), {
    stepMs: 1100,
    resetKey: states.length ? "ready" : "",
  });
  const [, setHover] = useState<string | null>(null);
  const box = useContainerWidth();
  const phone = box.width > 0 && box.width < PHONE_BELOW;
  const W = phone ? 360 : 720;
  const font = useSvgFont(W);
  const fs = font.fs;

  // a typical gradient: the median magnitude (for the bit pattern line)
  const typical = useMemo(() => {
    if (!grads) return 0;
    const nz = grads
      .filter((v) => v !== 0)
      .sort((a, b) => Math.abs(a) - Math.abs(b));
    return nz[Math.floor(nz.length / 2)] ?? 0;
  }, [grads]);

  if (!grads || states.length === 0) {
    return (
      <div
        data-pending-widget
        className="my-8 min-h-[40rem] text-sm text-neutral-600 dark:text-neutral-400"
      >
        {trained && "error" in trained
          ? `Could not load the trained weights: ${trained.error}`
          : "Loading the trained weights…"}
      </div>
    );
  }
  const s = states[st.step]!;
  const nb = HIST_HI - HIST_LO;
  const left = fs(12) * 2.6;
  const right = 8;
  const top = fs(12) * 2;
  const plotH = phone ? 170 : 200;
  const H = top + plotH + fs(11) * 2 + 8;
  const bw = (W - left - right) / nb;
  const bx = (b: number) => left + (b - HIST_LO) * bw;
  const maxCount = Math.max(...s.bins, 1);
  const lgMinSub = exponentOf(s.minSub);
  const lgMax = exponentOf(s.max);
  const hl = s.scaleExp ? "lscale" : "round";
  const scaled = typical * pow2(s.scaleExp);
  const rounded = roundTo(scaled, s.fmt);

  const visual = (
    <div ref={box.ref} className="min-w-0">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="mx-auto w-full text-neutral-800 dark:text-neutral-200"
        role="img"
        aria-label={`Histogram of the model's ${s.nonzero} nonzero gradients by power of two, times 2^${s.scaleExp}, against the range of ${FORMATS[s.fmt].name}: ${s.zero} round to zero, ${s.overflow} overflow`}
      >
        <defs>
          <pattern
            id="flush-hatch"
            width={6}
            height={6}
            patternUnits="userSpaceOnUse"
            patternTransform="rotate(45)"
          >
            <line
              x1={0}
              y1={0}
              x2={0}
              y2={6}
              stroke={OKABE_ITO.vermillion}
              strokeWidth={2}
              strokeOpacity={0.5}
            />
          </pattern>
        </defs>
        <text
          x={left}
          y={fs(12)}
          fontSize={fs(12)}
          fontWeight={600}
          fill="currentColor"
        >
          {FORMATS[s.fmt].name}
          {s.scaleExp ? ` · loss scale 2^${s.scaleExp}` : ""}
        </text>
        {/* rounds to zero: below half the smallest subnormal */}
        <rect
          x={left}
          y={top}
          width={Math.max(0, bx(Math.min(HIST_HI, lgMinSub - 1)) - left)}
          height={plotH}
          fill="url(#flush-hatch)"
          data-testid="flush-band"
        />
        {/* overflow: above the largest value */}
        {lgMax < HIST_HI && (
          <rect
            x={bx(lgMax + 1)}
            y={top}
            width={W - right - bx(lgMax + 1)}
            height={plotH}
            fill="url(#flush-hatch)"
          />
        )}
        {s.bins.map((c, i) => {
          const b = HIST_LO + i;
          const h = (plotH * c) / maxCount;
          // below half the smallest subnormal rounds to 0; above the largest value overflows
          const lost = b < lgMinSub - 1 || b > lgMax;
          return (
            <rect
              key={i}
              x={bx(b) + 0.5}
              y={top + plotH - h}
              width={Math.max(0.5, bw - 1)}
              height={h}
              fill={lost ? OKABE_ITO.vermillion : OKABE_ITO.blue}
            />
          );
        })}
        <line
          x1={left}
          x2={W - right}
          y1={top + plotH}
          y2={top + plotH}
          stroke="currentColor"
          strokeOpacity={0.4}
        />
        {[-48, -40, -32, -24, -16, -8, 0, 8].map((b) => (
          <text
            key={b}
            x={bx(b)}
            y={top + plotH + fs(11) + 2}
            fontSize={fs(11)}
            textAnchor="middle"
            fill="currentColor"
          >
            {`2^${b}`}
          </text>
        ))}
        <text
          x={W - right}
          y={top + plotH + fs(11) * 2 + 4}
          fontSize={fs(11)}
          textAnchor="end"
          fill="currentColor"
          fillOpacity={0.8}
        >
          |gradient × scale|
        </text>
      </svg>
    </div>
  );

  const stats = (
    <div
      className="space-y-1 font-mono text-xs text-neutral-700 dark:text-neutral-300"
      data-testid="rounding-stats"
    >
      <p>
        rounds to 0: {s.zero} · overflows: {s.overflow} · mean relative error of
        the rest: {(100 * s.meanRelErr).toPrecision(2)}%
      </p>
      <p className="break-all">
        a typical gradient {typical.toExponential(3)}
        {s.scaleExp ? ` × 2^${s.scaleExp}` : ""} → {FORMATS[s.fmt].name} bits{" "}
        {bitString(scaled, s.fmt)} ={" "}
        {rounded === 0 ? "0" : rounded.toExponential(3)}
      </p>
    </div>
  );

  return (
    <AnimationPanel
      testId={testId}
      title="Rounding the model's real gradients"
      summary="All 5,376 gradients of the trained model on one window, computed in your browser, then rounded to each format. First the five formats as they are; then FP16 with the loss multiplied by 2^k before the backward pass (loss scaling), which shifts every gradient up by k powers of two."
      stepper={st}
      stepLabel="state"
      caption={roundingCaption(s)}
      visual={visual}
      stats={stats}
      equation={children}
      hl={hl}
      onEquationHover={setHover}
    />
  );
}
