"use client";

/**
 * Chapter 11's hero: a ring all-reduce of four data-parallel workers' real
 * gradients. Each worker computes the gradient of the trained model's final
 * LayerNorm gain on its own four windows (the TS trainer, in the browser);
 * the ring sums them chunk by chunk (systems.ts `ringStates`): three
 * reduce-scatter steps, three all-gather steps. Bars are the values in each
 * worker's buffer; a chunk with a check mark holds the finished sum.
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
  completeChunks,
  ringCaption,
  ringStates,
  valuesSent,
} from "@/lib/anim/training/ring-steps";
import { sampleBatch, splitIds } from "@/lib/training/data";
import { lossAndGrads } from "@/lib/training/model-grad";
import { PART1_MODEL } from "@/lib/training/trainer";
import { mulberry32 } from "@/lib/transformer/random";
import { OKABE_ITO } from "@/lib/viz/palette";

import { useCorpus, useTrainedWeights } from "./useTrainingData";

const SELECT =
  "focus-ring h-11 w-full rounded border border-neutral-300 bg-white px-2 text-sm dark:border-neutral-700 dark:bg-neutral-950";
const CHUNK_COLOUR = [
  OKABE_ITO.blue,
  OKABE_ITO.orange,
  OKABE_ITO.green,
  OKABE_ITO.vermillion,
];
const TENSOR = "ln_final.gamma";
const N = 4;

export default function RingAllReduceAnimation({
  children,
  testId = "ring-animation",
}: {
  children?: ReactNode;
  testId?: string;
}): JSX.Element {
  const [seed, setSeed] = useState(11);
  const corpus = useCorpus();
  const trained = useTrainedWeights(true);
  const grads = useMemo(() => {
    if (!corpus || "error" in corpus || !trained || "error" in trained)
      return null;
    const { train } = splitIds(corpus.ids);
    const b = sampleBatch(train, 16, 8, mulberry32(seed));
    return Array.from({ length: N }, (_, w) => {
      const g = lossAndGrads(
        trained,
        PART1_MODEL,
        b.inputs.slice(w * 4, w * 4 + 4),
        b.targets.slice(w * 4, w * 4 + 4),
      ).grads;
      return (g[TENSOR] as number[]).slice();
    });
  }, [corpus, trained, seed]);
  const states = useMemo(() => (grads ? ringStates(grads) : []), [grads]);
  const st = useStepper(Math.max(1, states.length), {
    stepMs: 1100,
    resetKey: grads ? String(seed) : "",
  });
  const [, setHover] = useState<string | null>(null);
  const box = useContainerWidth();
  const phone = box.width > 0 && box.width < PHONE_BELOW;
  const W = phone ? 360 : 720;
  const font = useSvgFont(W);
  const fs = font.fs;

  if (!grads || states.length === 0) {
    const err =
      (corpus && "error" in corpus && corpus.error) ||
      (trained && "error" in trained && trained.error);
    return (
      <p
        data-pending-widget
        className="my-8 min-h-[44rem] text-sm text-neutral-600 dark:text-neutral-400"
      >
        {err ? `Could not load the data: ${err}` : "Loading the model…"}
      </p>
    );
  }
  const s = states[st.step]!;
  const size = grads[0]!.length;
  const done = completeChunks(s, N);
  const vmax = Math.max(
    1e-12,
    ...states.flatMap((x) => x.bufs.flat().map((v) => Math.abs(v))),
  );
  const sending = new Map(s.sends.map((x) => [x.from, x.chunk]));
  const hl =
    s.phase === "reduce-scatter" ? "rs" : s.phase === "all-gather" ? "ag" : "";

  // ─── layout: four boxes on a ring (2 × 2), arrows clockwise ──────────
  const gap = phone ? 64 : 90;
  const bw = (W - gap) / 2;
  const top = fs(12) + 8;
  const bh = phone ? 120 : 130;
  const H = top + 2 * bh + gap + fs(11) + 8;
  const pos = [
    { x: 0, y: top },
    { x: bw + gap, y: top },
    { x: bw + gap, y: top + bh + gap },
    { x: 0, y: top + bh + gap },
  ];
  const pad = 8;
  const barW = (bw - 2 * pad) / size;
  const mid = (y: number) => y + fs(12) + 6 + (bh - fs(12) - 24) / 2;
  const half = (bh - fs(12) - 24) / 2;

  const visual = (
    <div ref={box.ref} className="min-w-0">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="mx-auto w-full text-neutral-800 dark:text-neutral-200"
        role="img"
        aria-label={`Ring all-reduce of four workers' gradients, ${s.phase} step ${s.step}`}
      >
        <defs>
          <marker
            id="ring-arrow"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="7"
            markerHeight="7"
            orient="auto-start-reverse"
          >
            <path d="M0,0L10,5L0,10Z" fill="currentColor" />
          </marker>
        </defs>
        <text x={0} y={fs(12)} fontSize={fs(12)} fill="currentColor">
          {s.phase === "start"
            ? "each worker's own gradient"
            : s.phase === "reduce-scatter"
              ? "reduce-scatter: send one chunk, add the one received"
              : "all-gather: pass the finished chunks round"}
        </text>
        {pos.map((p, r) => (
          <g key={r} data-testid={`ring-worker-${r}`}>
            <rect
              x={p.x}
              y={p.y}
              width={bw}
              height={bh}
              rx={6}
              fill="none"
              stroke="currentColor"
              strokeOpacity={0.35}
            />
            <text
              x={p.x + pad}
              y={p.y + fs(12) + 2}
              fontSize={fs(12)}
              fill="currentColor"
            >
              worker {r + 1}
            </text>
            <line
              x1={p.x + pad}
              x2={p.x + bw - pad}
              y1={mid(p.y)}
              y2={mid(p.y)}
              stroke="currentColor"
              strokeOpacity={0.3}
            />
            {s.bufs[r]!.map((v, i) => {
              const c = Math.floor((i * N) / size);
              const h = (Math.abs(v) / vmax) * half;
              const fin = done[r]![c];
              const out = sending.get(r) === c;
              return (
                <rect
                  key={i}
                  x={p.x + pad + i * barW + 1}
                  y={v >= 0 ? mid(p.y) - h : mid(p.y)}
                  width={Math.max(1, barW - 2)}
                  height={Math.max(0.5, h)}
                  fill={CHUNK_COLOUR[c]}
                  fillOpacity={fin ? 1 : out ? 0.85 : 0.45}
                  stroke={out ? "currentColor" : "none"}
                  strokeWidth={out ? 1.5 : 0}
                />
              );
            })}
            {Array.from({ length: N }, (_, c) => (
              <text
                key={c}
                x={p.x + pad + (c + 0.5) * (size / N) * barW}
                y={p.y + bh - 6}
                fontSize={fs(11)}
                textAnchor="middle"
                fill="currentColor"
                data-done={done[r]![c] ? "true" : "false"}
              >
                {done[r]![c] ? `✓${c + 1}` : `${c + 1}`}
              </text>
            ))}
          </g>
        ))}
        {/* clockwise arrows 1→2→3→4→1 */}
        {[0, 1, 2, 3].map((r) => {
          const a = pos[r]!;
          const b = pos[(r + 1) % N]!;
          const ax = a.x + bw / 2;
          const ay = a.y + bh / 2;
          const bx = b.x + bw / 2;
          const by = b.y + bh / 2;
          let x1: number, y1: number, x2: number, y2: number;
          if (ay === by) {
            const dir = bx > ax ? 1 : -1;
            x1 = ax + (dir * bw) / 2 + dir * 4;
            x2 = bx - (dir * bw) / 2 - dir * 4;
            y1 = y2 = ay + (dir > 0 ? -bh / 4 : bh / 4);
          } else {
            const dir = by > ay ? 1 : -1;
            y1 = ay + (dir * bh) / 2 + dir * 4;
            y2 = by - (dir * bh) / 2 - dir * 4;
            x1 = x2 = ax + (dir > 0 ? bw / 4 : -bw / 4);
          }
          const c = sending.get(r);
          return (
            <g key={`a${r}`}>
              <line
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke={c === undefined ? "currentColor" : CHUNK_COLOUR[c]}
                strokeOpacity={c === undefined ? 0.25 : 1}
                strokeWidth={c === undefined ? 1.5 : 3}
                markerEnd="url(#ring-arrow)"
              />
              {c !== undefined && (
                <text
                  x={x1 === x2 ? x1 + (x1 > W / 2 ? -8 : 8) : (x1 + x2) / 2}
                  y={(y1 + y2) / 2 + (y1 === y2 ? -6 : 4)}
                  fontSize={fs(11)}
                  textAnchor={
                    x1 === x2 ? (x1 > W / 2 ? "end" : "start") : "middle"
                  }
                  fill="currentColor"
                >
                  {phone ? `#${c + 1}` : `chunk ${c + 1}`}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );

  const sent = valuesSent(s, size, N);
  const stats = (
    <p
      className="font-mono text-xs text-neutral-700 dark:text-neutral-300"
      data-testid="ring-stats"
    >
      sent by each worker so far: {sent} of {size} values · when done: 2(n−1)/n
      × {size} = {(2 * (N - 1) * size) / N} values, the same for any n
    </p>
  );

  const params = (
    <label className="flex min-w-0 flex-col gap-1 text-sm">
      <span className="font-medium text-neutral-700 dark:text-neutral-300">
        The workers&apos; windows
      </span>
      <select
        value={seed}
        onChange={(e) => setSeed(Number(e.target.value))}
        className={SELECT}
        aria-label="Batch seed for the ring animation"
      >
        {[11, 12, 13].map((v) => (
          <option key={v} value={v}>
            batch drawn with seed {v}
          </option>
        ))}
      </select>
    </label>
  );

  return (
    <AnimationPanel
      testId={testId}
      title="A ring all-reduce of four workers' gradients"
      summary="Four data-parallel workers, each with the gradient of the trained model's final LayerNorm gain (16 values) on its own four windows, computed in your browser. The ring sums them without any worker sending more than 2(n − 1)/n of the vector."
      stepper={st}
      stepLabel="step"
      countFrom={0}
      caption={ringCaption(s, N, size)}
      visual={visual}
      stats={stats}
      equation={children}
      hl={hl}
      onEquationHover={setHover}
      params={params}
    />
  );
}
