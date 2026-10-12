"use client";

/**
 * Chapter 13's hero: two DeMo syncs of four workers, computed in the
 * browser from the trained model (slowlinks.ts `demoSync`), shown for the
 * first 64-value chunk of one attention matrix: gradients, momenta, their
 * DCT, the top 4 coefficients each worker sends, the residual it keeps,
 * the gathered coefficients, their average, the inverse DCT and its sign.
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
  DEMO_HL,
  demoCaption,
  demoStates,
  type DemoStage,
} from "@/lib/anim/training/demo-steps";
import { sampleBatch, splitIds } from "@/lib/training/data";
import { lossAndGrads } from "@/lib/training/model-grad";
import {
  createDemo,
  demoSync,
  type DemoRecord,
} from "@/lib/training/slowlinks";
import { PART1_MODEL } from "@/lib/training/trainer";
import { mulberry32 } from "@/lib/transformer/random";
import { OKABE_ITO } from "@/lib/viz/palette";

import { useCorpus, useTrainedWeights } from "./useTrainingData";

const TENSOR = "block_0.attn.W_q";
const SEED = 5;
const WORKER_COLOUR = [
  OKABE_ITO.blue,
  OKABE_ITO.orange,
  OKABE_ITO.green,
  OKABE_ITO.vermillion,
];

const ROW_LABEL: Record<DemoStage, string> = {
  grad: "gradient g",
  momentum: "momentum M = βM + g",
  dct: "DCT of the chunk",
  topk: "top 4 of 64 sent",
  residual: "kept: M − IDCT(sent)",
  gather: "gathered from all",
  mean: "mean per position",
  inverse: "IDCT → M*",
  sign: "sign(M*)",
};

export default function DemoSyncAnimation({
  children,
  testId = "demo-animation",
}: {
  children?: ReactNode;
  testId?: string;
}): JSX.Element {
  const corpus = useCorpus();
  const trained = useTrainedWeights(true);
  const records = useMemo(() => {
    if (!corpus || "error" in corpus || !trained || "error" in trained)
      return null;
    const { train } = splitIds(corpus.ids);
    const rng = mulberry32(SEED);
    const st = createDemo(trained, 4);
    const out: DemoRecord[] = [];
    for (let step = 0; step < 2; step++) {
      const b = sampleBatch(train, 16, 8, rng);
      const grads = [0, 1, 2, 3].map(
        (w) =>
          lossAndGrads(
            trained,
            PART1_MODEL,
            b.inputs.slice(w * 4, w * 4 + 4),
            b.targets.slice(w * 4, w * 4 + 4),
          ).grads,
      );
      out.push(demoSync(st, grads, TENSOR).rec!);
    }
    return out;
  }, [corpus, trained]);
  const states = useMemo(() => (records ? demoStates(records) : []), [records]);
  const st = useStepper(Math.max(1, states.length), {
    stepMs: 1400,
    resetKey: records ? "ready" : "",
  });
  const [, setHover] = useState<string | null>(null);
  const box = useContainerWidth();
  const phone = box.width > 0 && box.width < PHONE_BELOW;
  const W = phone ? 360 : 720;
  const font = useSvgFont(W);
  const fs = font.fs;

  if (!records || states.length === 0) {
    return (
      <p
        data-pending-widget
        className="my-8 min-h-[46rem] text-sm text-neutral-600 dark:text-neutral-400"
      >
        Loading the model…
      </p>
    );
  }
  const s = states[st.step]!;
  const r = records[s.sync]!;
  const n = r.chunk;
  const perWorker = ["grad", "momentum", "dct", "topk", "residual"].includes(
    s.stage,
  );
  const series = (w: number): number[] => {
    const x = r.workers[w]!;
    if (s.stage === "grad") return x.grad;
    if (s.stage === "momentum") return x.momentum;
    if (s.stage === "residual") return x.residual;
    return x.coef;
  };

  // ─── layout ─────────────────────────────────────────────────────────
  const left = phone ? 6 : 128;
  const top = fs(12) + 10;
  const rowH = phone ? 58 : 62;
  const labelH = phone ? fs(11) + 4 : 0;
  const plotW = W - left - 6;
  const bw = plotW / n;
  const rows = perWorker ? 4 : s.stage === "gather" ? 4 : 1;
  const H = top + rows * (rowH + labelH) + 12;
  const rowVals = perWorker
    ? [0, 1, 2, 3].map(series)
    : s.stage === "gather"
      ? r.workers.map((w) => {
          const out = new Array<number>(n).fill(0);
          w.idx.forEach((i, j) => (out[i] = w.sent[j]!));
          return out;
        })
      : [
          s.stage === "mean"
            ? r.meanCoef
            : s.stage === "inverse"
              ? r.aggregate
              : r.aggregate.map((v) => Math.sign(v)),
        ];
  const vmax = Math.max(1e-12, ...rowVals.flat().map((v) => Math.abs(v)));

  const visual = (
    <div ref={box.ref} className="min-w-0">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="mx-auto w-full text-neutral-800 dark:text-neutral-200"
        role="img"
        aria-label={`DeMo sync ${s.sync + 1}, stage ${s.stage}`}
      >
        <text x={0} y={fs(12)} fontSize={fs(12)} fill="currentColor">
          {ROW_LABEL[s.stage]} · first {n} values of {TENSOR}
        </text>
        {rowVals.map((vals, k) => {
          const y0 = top + k * (rowH + labelH) + labelH;
          const mid = y0 + rowH / 2;
          const sentIdx =
            perWorker && (s.stage === "topk" || s.stage === "residual")
              ? new Set(r.workers[k]!.idx)
              : null;
          const colour = rows === 1 ? OKABE_ITO.purple : WORKER_COLOUR[k]!;
          return (
            <g key={k} data-testid={`demo-row-${k}`}>
              {rows > 1 ? (
                <text
                  x={phone ? 0 : left - 6}
                  y={phone ? y0 - 3 : mid + fs(11) / 3}
                  fontSize={fs(11)}
                  textAnchor={phone ? "start" : "end"}
                  fill="currentColor"
                >
                  worker {k + 1}
                </text>
              ) : null}
              <line
                x1={left}
                x2={left + plotW}
                y1={mid}
                y2={mid}
                stroke="currentColor"
                strokeOpacity={0.25}
              />
              {vals.map((v, i) => {
                const h = (Math.abs(v) / vmax) * (rowH / 2 - 4);
                const picked = sentIdx?.has(i);
                const dim =
                  s.stage === "topk"
                    ? !picked
                    : s.stage === "gather" && v === 0;
                return (
                  <rect
                    key={i}
                    x={left + i * bw + 0.5}
                    y={v >= 0 ? mid - h : mid}
                    width={Math.max(1, bw - 1)}
                    height={Math.max(v === 0 ? 0 : 0.8, h)}
                    fill={colour}
                    fillOpacity={dim ? 0.2 : 0.9}
                    stroke={
                      picked && s.stage === "topk" ? "currentColor" : "none"
                    }
                    strokeWidth={1}
                    data-sent={picked ? "true" : undefined}
                  />
                );
              })}
            </g>
          );
        })}
      </svg>
    </div>
  );

  const w0 = r.workers[0]!;
  const stats = (
    <p
      className="font-mono text-xs text-neutral-700 dark:text-neutral-300"
      data-testid="demo-stats"
    >
      per chunk of {n}: each worker sends {r.k} values + {r.k} one-byte
      positions = {r.k * 3} bytes instead of {n * 2} · worker 1 sends positions{" "}
      {w0.idx.map((i) => i + 1).join(", ")}
    </p>
  );

  return (
    <AnimationPanel
      testId={testId}
      title="One DeMo sync, step by step"
      summary="Four workers' momenta for one 64-value chunk of the trained model's first attention query matrix, computed in your browser. Each worker sends 4 coefficients of 64 (and where they sit), keeps the rest for next time, and every worker ends with the same averaged step."
      stepper={st}
      stepLabel="stage"
      caption={demoCaption(s, records)}
      visual={visual}
      stats={stats}
      equation={children}
      hl={DEMO_HL[s.stage]}
      onEquationHover={setHover}
    />
  );
}
