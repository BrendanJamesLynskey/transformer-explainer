"use client";

/**
 * Chapter 12's hero: one training step of a pipeline as a timeline, from
 * the discrete-event simulation (systems.ts `pipeline`): a row per device,
 * a box per forward (light) or backward (dark) of a micro-batch, the bubble
 * the empty space. The schedule, the micro-batches and the chunks re-run
 * the simulation.
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
  pipelineCaption,
  pipelineStates,
} from "@/lib/anim/training/pipeline-steps";
import { pipeline, type PipeKind } from "@/lib/training/systems";
import { OKABE_ITO } from "@/lib/viz/palette";

const SELECT =
  "focus-ring h-11 w-full rounded border border-neutral-300 bg-white px-2 text-sm dark:border-neutral-700 dark:bg-neutral-950";
const MB_COLOUR = [
  OKABE_ITO.blue,
  OKABE_ITO.orange,
  OKABE_ITO.green,
  OKABE_ITO.vermillion,
  OKABE_ITO.sky,
  OKABE_ITO.purple,
  OKABE_ITO.yellow,
  "#7f7f7f",
];

export default function PipelineAnimation({
  children,
  testId = "pipeline-animation",
}: {
  children?: ReactNode;
  testId?: string;
}): JSX.Element {
  const [kind, setKind] = useState<PipeKind>("1f1b");
  const [m, setM] = useState(8);
  const [v, setV] = useState(2);
  const p = 4;
  const result = useMemo(() => pipeline(kind, p, m, v), [kind, m, v]);
  const states = useMemo(() => pipelineStates(result), [result]);
  const st = useStepper(states.length, {
    stepMs: 450,
    resetKey: `${kind}-${m}-${v}`,
  });
  const [, setHover] = useState<string | null>(null);
  const box = useContainerWidth();
  const phone = box.width > 0 && box.width < PHONE_BELOW;
  const W = phone ? 360 : 720;
  const font = useSvgFont(W);
  const fs = font.fs;
  const s = states[st.step]!;

  const left = phone ? 34 : 70;
  const top = fs(12) + 10;
  const rowH = phone ? 34 : 40;
  const plotW = W - left - 6;
  const T = result.makespan;
  const x = (t: number) => left + (plotW * t) / T;
  const H = top + p * rowH + fs(11) * 2 + 14;

  const visual = (
    <div ref={box.ref} className="min-w-0">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="mx-auto w-full text-neutral-800 dark:text-neutral-200"
        role="img"
        aria-label={`Pipeline schedule ${kind}, ${p} devices, ${m} micro-batches: bubble ${(100 * result.bubble).toFixed(1)}%`}
      >
        <defs>
          <pattern
            id="pipe-idle"
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
              stroke="currentColor"
              strokeOpacity={0.25}
              strokeWidth={2}
            />
          </pattern>
        </defs>
        <text x={left} y={fs(12)} fontSize={fs(12)} fill="currentColor">
          {phone
            ? "time → (hatched = idle)"
            : "time → (forward light, backward dark; hatched = idle, the bubble)"}
        </text>
        {Array.from({ length: p }, (_, d) => (
          <g key={d}>
            <text
              x={left - 4}
              y={top + d * rowH + rowH / 2 + fs(11) / 3}
              fontSize={fs(11)}
              textAnchor="end"
              fill="currentColor"
            >
              {phone ? `d${d + 1}` : `device ${d + 1}`}
            </text>
            <rect
              x={left}
              y={top + d * rowH + 3}
              width={Math.max(0, x(s.t) - left)}
              height={rowH - 6}
              fill="url(#pipe-idle)"
            />
          </g>
        ))}
        {result.items
          .filter((it) => it.start <= s.t)
          .map((it, i) => {
            const active = s.t < it.end;
            const w = Math.max(0, x(it.end) - x(it.start));
            return (
              <g key={i}>
                {/* an opaque underlay, so the idle hatching shows only where a device is idle */}
                <rect
                  x={x(it.start) + 0.5}
                  y={top + it.device * rowH + 3}
                  width={Math.max(0, w - 1)}
                  height={rowH - 6}
                  className="fill-neutral-50 dark:fill-neutral-900"
                />
                <rect
                  x={x(it.start) + 0.5}
                  y={top + it.device * rowH + 3}
                  width={Math.max(0, w - 1)}
                  height={rowH - 6}
                  fill={MB_COLOUR[it.mb % MB_COLOUR.length]}
                  fillOpacity={it.op === "F" ? 0.45 : 0.95}
                  stroke={active ? "currentColor" : "none"}
                  strokeWidth={active ? 2 : 0}
                  data-active={active ? "true" : "false"}
                />
                {w > fs(11) * 2.3 && (
                  <text
                    x={x(it.start) + w / 2}
                    y={top + it.device * rowH + rowH / 2 + fs(11) / 3}
                    fontSize={fs(11)}
                    textAnchor="middle"
                    fill="currentColor"
                  >
                    {it.op}
                    {it.mb + 1}
                  </text>
                )}
              </g>
            );
          })}
        <line
          x1={x(s.t)}
          x2={x(s.t)}
          y1={top}
          y2={top + p * rowH}
          stroke="currentColor"
          strokeWidth={2}
        />
        <text
          x={left}
          y={top + p * rowH + fs(11) + 6}
          fontSize={fs(11)}
          fill="currentColor"
        >
          0
        </text>
        <text
          x={W - 6}
          y={top + p * rowH + fs(11) + 6}
          fontSize={fs(11)}
          textAnchor="end"
          fill="currentColor"
        >
          {phone
            ? `${Number(T.toFixed(2))} time units`
            : `${Number(T.toFixed(2))} (forward = 1, backward = 2 per micro-batch)`}
        </text>
      </svg>
    </div>
  );

  const stats = (
    <p
      className="font-mono text-xs text-neutral-700 dark:text-neutral-300"
      data-testid="pipeline-stats"
    >
      measured bubble {(100 * result.bubble).toFixed(1)}% · formula (p − 1)/
      {result.v > 1 ? "(v·m)" : "m"} = {(100 * result.closedForm).toFixed(1)}% ·
      step {Number(T.toFixed(3))} vs {result.ideal} of work per device
    </p>
  );

  const params = (
    <>
      <label className="flex min-w-0 flex-col gap-1 text-sm">
        <span className="font-medium text-neutral-700 dark:text-neutral-300">
          Schedule
        </span>
        <select
          value={kind}
          onChange={(e) => setKind(e.target.value as PipeKind)}
          className={SELECT}
          aria-label="Pipeline schedule"
        >
          <option value="gpipe">
            GPipe (all forwards, then all backwards)
          </option>
          <option value="1f1b">1F1B (one forward, one backward)</option>
          <option value="interleaved">
            Interleaved 1F1B (v chunks per device)
          </option>
        </select>
      </label>
      <label className="flex min-w-0 flex-col gap-1 text-sm">
        <span className="font-medium text-neutral-700 dark:text-neutral-300">
          Micro-batches m
        </span>
        <select
          value={m}
          onChange={(e) => setM(Number(e.target.value))}
          className={SELECT}
          aria-label="Micro-batches"
        >
          {[4, 8, 16].map((k) => (
            <option key={k} value={k}>
              {k}
            </option>
          ))}
        </select>
      </label>
      {kind === "interleaved" && (
        <label className="flex min-w-0 flex-col gap-1 text-sm">
          <span className="font-medium text-neutral-700 dark:text-neutral-300">
            Chunks per device v
          </span>
          <select
            value={v}
            onChange={(e) => setV(Number(e.target.value))}
            className={SELECT}
            aria-label="Chunks per device"
          >
            {[2, 4].map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
        </label>
      )}
    </>
  );

  return (
    <AnimationPanel
      testId={testId}
      title="A pipeline's step, and its bubble"
      summary={`${p} devices hold a quarter of the layers each; a batch is cut into micro-batches that flow through them. Every box is a simulated event (a backward takes twice a forward); the idle time is measured, not drawn by hand.`}
      stepper={st}
      stepLabel="event"
      caption={pipelineCaption(s, result)}
      visual={visual}
      stats={stats}
      equation={children}
      hl={s.done === result.items.length ? "bubble" : "pp"}
      onEquationHover={setHover}
      params={params}
    />
  );
}
