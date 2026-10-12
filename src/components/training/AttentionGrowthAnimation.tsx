"use client";

/**
 * Chapter 8's hero: attention collapsing during the run at peak learning
 * rate 0.1. Left, block 2 head 1's attention on "alice was" at each saved
 * snapshot, computed in the browser by the site's forward pass on the
 * snapshot's weights (`attentionGrowthStates`). Right, the largest
 * attention score over the run (log scale) for the baseline and with
 * QK-norm, the mean of three seeds as recorded (results.md §8b).
 */
import { useEffect, useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import {
  PHONE_BELOW,
  useContainerWidth,
} from "@/components/viz/useContainerWidth";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { showChar } from "@/lib/anim/format";
import {
  ATTN_TEXT,
  attentionGrowthCaption,
  attentionGrowthStates,
  type Snapshots,
} from "@/lib/anim/training/attention-growth-steps";
import { paramNames, type ParamMap } from "@/lib/training/params";
import { PART1_MODEL } from "@/lib/training/trainer";
import { OKABE_ITO } from "@/lib/viz/palette";

export type GrowthCurve = {
  name: string;
  stats: { step: number; maxScore: number; attnEntropy: number }[];
};

let snapPromise: Promise<Snapshots> | null = null;
function loadSnapshots(): Promise<Snapshots> {
  snapPromise ??= fetch("/training/attention-snapshots.json")
    .then((r) => {
      if (!r.ok) throw new Error(`snapshots: HTTP ${r.status}`);
      return r.json() as Promise<Snapshots>;
    })
    .then((j) => ({
      steps: j.steps,
      params: Object.fromEntries(
        Object.entries(j.params).map(([t, p]) => [
          t,
          Object.fromEntries(
            paramNames(PART1_MODEL).map((k) => [k, (p as ParamMap)[k]!]),
          ),
        ]),
      ),
    }));
  return snapPromise;
}

const DASH = ["", "6 3"];
const COLOURS = [OKABE_ITO.vermillion, OKABE_ITO.blue];

export default function AttentionGrowthAnimation({
  evals,
  curves,
  children,
  testId = "attention-growth-animation",
}: {
  evals: { step: number; val: number }[];
  curves: GrowthCurve[];
  children?: ReactNode;
  testId?: string;
}): JSX.Element {
  const [snap, setSnap] = useState<Snapshots | { error: string } | null>(null);
  useEffect(() => {
    let live = true;
    loadSnapshots()
      .then((s) => live && setSnap(s))
      .catch((e: Error) => live && setSnap({ error: e.message }));
    return () => {
      live = false;
    };
  }, []);
  const states = useMemo(
    () =>
      snap && !("error" in snap) ? attentionGrowthStates(snap, evals) : [],
    [snap, evals],
  );
  const st = useStepper(Math.max(1, states.length), {
    stepMs: 1200,
    resetKey: states.length ? "ready" : "",
  });
  const [, setHover] = useState<string | null>(null);
  const box = useContainerWidth();
  const phone = box.width > 0 && box.width < PHONE_BELOW;
  const W = phone ? 360 : 720;
  const font = useSvgFont(W);
  const fs = font.fs;

  if (!snap || "error" in snap || states.length === 0) {
    return (
      <div
        data-pending-widget
        className="my-8 min-h-[44rem] text-sm text-neutral-600 dark:text-neutral-400"
      >
        {snap && "error" in snap
          ? `Could not load the snapshots: ${snap.error}`
          : "Loading the run's weight snapshots…"}
      </div>
    );
  }
  const s = states[st.step]!;
  const chars = Array.from(ATTN_TEXT.slice(0, -1));
  const S = chars.length;

  // ─── layout: heat map left (top on phones), curve right ─────
  const hmSize = phone ? W - 2 * fs(12) * 1.4 : 300;
  const cell = hmSize / S;
  const hmX = fs(12) * 1.4;
  const hmY = fs(12) * 1.6;
  const cx0 = phone ? fs(12) * 3.4 : hmX + hmSize + 60;
  const cy0 = phone ? hmY + hmSize + fs(12) * 3.4 : hmY;
  const cw = W - cx0 - 10;
  const ch = phone ? 170 : hmSize - fs(11) * 2;
  const H = cy0 + ch + fs(11) * 2 + (phone ? 2 : 1) * (fs(12) + 10) + 6;
  const tMax = curves[0]!.stats[curves[0]!.stats.length - 1]!.step;
  const lo = 0;
  const hi = 6; // log10 of the score axis: 1 .. 1,000,000
  const cx = (t: number) => cx0 + (cw * t) / tMax;
  const cy = (v: number) =>
    cy0 + ch * (1 - (Math.log10(Math.max(1, v)) - lo) / (hi - lo));
  const hl = st.step === 0 ? "score" : "ent";

  const visual = (
    <div ref={box.ref} className="min-w-0">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="mx-auto w-full text-neutral-800 dark:text-neutral-200"
        role="img"
        aria-label={`Attention weights of block 2, head 1 on "${ATTN_TEXT}" at step ${s.step}, and the largest attention score over training for ${curves.map((c) => c.name).join(" and ")}`}
      >
        <text x={hmX} y={fs(12)} fontSize={fs(12)} fill="currentColor">
          block 2 · head 1 · step {s.step.toLocaleString("en-GB")}
        </text>
        {s.weights.map((row, i) =>
          row.map((p, j) => (
            <rect
              key={`${i}-${j}`}
              x={hmX + j * cell}
              y={hmY + i * cell}
              width={cell - 1}
              height={cell - 1}
              fill={j > i ? "url(#mask-hatch)" : OKABE_ITO.vermillion}
              fillOpacity={j > i ? 0.4 : Math.max(0.04, p)}
            />
          )),
        )}
        <defs>
          <pattern
            id="mask-hatch"
            width={6}
            height={6}
            patternUnits="userSpaceOnUse"
            patternTransform="rotate(45)"
          >
            <line x1={0} y1={0} x2={0} y2={6} stroke="#888" strokeWidth={2} />
          </pattern>
        </defs>
        {chars.map((c, i) => (
          <g key={i}>
            <text
              x={hmX - 4}
              y={hmY + i * cell + cell / 2 + fs(11) / 3}
              fontSize={fs(11)}
              textAnchor="end"
              fill="currentColor"
            >
              {showChar(c)}
            </text>
            <text
              x={hmX + i * cell + cell / 2}
              y={hmY + hmSize + fs(11) + 2}
              fontSize={fs(11)}
              textAnchor="middle"
              fill="currentColor"
            >
              {showChar(c)}
            </text>
          </g>
        ))}
        <text x={cx0} y={cy0 - 6} fontSize={fs(12)} fill="currentColor">
          largest attention score (log scale)
        </text>
        {[1, 10, 100, 1000, 10000, 100000, 1000000].map((v) => (
          <g key={v}>
            <line
              x1={cx0}
              x2={cx0 + cw}
              y1={cy(v)}
              y2={cy(v)}
              stroke="currentColor"
              strokeOpacity={0.12}
            />
            <text
              x={cx0 - 4}
              y={cy(v) + fs(10) / 3}
              fontSize={fs(10)}
              textAnchor="end"
              fill="currentColor"
            >
              {v >= 1000 ? `1e${Math.log10(v)}` : v}
            </text>
          </g>
        ))}
        {curves.map((c, k) => (
          <path
            key={c.name}
            d={c.stats
              .map(
                (p, i) =>
                  `${i ? "L" : "M"}${cx(p.step).toFixed(1)},${cy(p.maxScore).toFixed(1)}`,
              )
              .join("")}
            fill="none"
            stroke={COLOURS[k]}
            strokeWidth={2.5}
            strokeDasharray={DASH[k]}
          />
        ))}
        <line
          x1={cx(s.step)}
          x2={cx(s.step)}
          y1={cy0}
          y2={cy0 + ch}
          stroke="currentColor"
          strokeOpacity={0.5}
        />
        <circle
          cx={cx(s.step)}
          cy={cy(s.maxScore)}
          r={5}
          fill={COLOURS[0]}
          stroke="currentColor"
        />
        {[0, 500, 1000, 1500]
          .filter((t) => t <= tMax)
          .map((t) => (
            <text
              key={t}
              x={cx(t)}
              y={cy0 + ch + fs(11) + 2}
              fontSize={fs(11)}
              textAnchor={t === 0 ? "start" : "middle"}
              fill="currentColor"
            >
              {t}
            </text>
          ))}
        {curves.map((c, k) => {
          const ly =
            cy0 + ch + fs(11) * 2 + 6 + (phone ? k : 0) * (fs(12) + 10);
          const lx = cx0 + (phone ? 0 : k * (cw / 2));
          return (
            <g key={c.name}>
              <line
                x1={lx}
                x2={lx + 22}
                y1={ly + fs(12) / 2}
                y2={ly + fs(12) / 2}
                stroke={COLOURS[k]}
                strokeWidth={3}
                strokeDasharray={DASH[k]}
              />
              <text
                x={lx + 26}
                y={ly + fs(12) * 0.85}
                fontSize={fs(12)}
                fill="currentColor"
              >
                {c.name}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );

  return (
    <AnimationPanel
      testId={testId}
      title="Attention collapses at a high learning rate"
      summary="Part 1's model trained with AdamW at peak learning rate 0.1 (ten times the default). At each saved step the site runs its own forward pass on the saved weights: the heat map is one head's attention on the window (darker = more weight; hatched = masked future). The curve is the largest attention score on the validation windows over the run, mean of three seeds, with and without QK-norm; the dot is this window's largest score at the saved step."
      stepper={st}
      stepLabel="snapshot"
      caption={attentionGrowthCaption(s)}
      visual={visual}
      equation={children}
      hl={hl}
      onEquationHover={setHover}
    />
  );
}
