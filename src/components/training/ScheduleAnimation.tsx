"use client";

/**
 * Chapter 6's hero: four learning-rate schedules (top) and the validation
 * loss each one measured (bottom), drawn up to the current step. The
 * learning rates come from the trainer's schedules.ts, the losses are the
 * mean over three seeds of the recorded runs (results.md §6b).
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
  scheduleCaption,
  scheduleStates,
  toSchedule,
  type ScheduleRow,
} from "@/lib/anim/training/schedule-steps";
import { lrAt } from "@/lib/training/schedules";
import { OKABE_ITO } from "@/lib/viz/palette";

const COLOURS = [
  OKABE_ITO.orange,
  OKABE_ITO.sky,
  OKABE_ITO.blue,
  OKABE_ITO.vermillion,
];
const DASH = ["", "6 3", "2 3", "10 3 2 3"];
const TOTAL = 2000;

export default function ScheduleAnimation({
  rows,
  children,
  testId = "schedule-animation",
}: {
  rows: ScheduleRow[];
  children?: ReactNode;
  testId?: string;
}): JSX.Element {
  const states = useMemo(() => scheduleStates(rows, TOTAL), [rows]);
  const names = rows.map((r) => r.name);
  const st = useStepper(states.length, { stepMs: 250 });
  const [, setHover] = useState<string | null>(null);
  const box = useContainerWidth();
  const phone = box.width > 0 && box.width < PHONE_BELOW;
  const W = phone ? 360 : 720;
  const font = useSvgFont(W);
  const fs = font.fs;
  const s = states[st.step]!;
  const hl = s.step < 50 ? "warm" : s.step >= TOTAL * 0.8 ? "decay" : "cosine";

  // ─── layout ─────────────────────────────────────
  const left = fs(12) * 3.2;
  const right = 8;
  const top1 = fs(12) + 6;
  const h1 = phone ? 110 : 120;
  const gap = fs(12) * 2.6;
  const top2 = top1 + h1 + gap;
  const h2 = phone ? 150 : 170;
  const legendY = top2 + h2 + fs(12) * 2.4;
  const perRow = phone ? 2 : 4;
  const H = legendY + Math.ceil(rows.length / perRow) * (fs(12) + 10) + 4;
  const x = (t: number) => left + ((W - left - right) * t) / TOTAL;
  const lrMax = Math.max(...rows.map((r) => r.spec.lr));
  const y1 = (lr: number) => top1 + h1 * (1 - lr / lrMax);
  const yLo = 2.0;
  const yHi = 4.3;
  const y2 = (v: number) =>
    top2 + (h2 * (yHi - Math.min(yHi, Math.max(yLo, v)))) / (yHi - yLo);
  const lrPath = (r: ScheduleRow) => {
    const sch = toSchedule(r.spec, TOTAL);
    const pts: string[] = [];
    for (let t = 0; t <= TOTAL; t += 10)
      pts.push(
        `${pts.length ? "L" : "M"}${x(t).toFixed(1)},${y1(lrAt(Math.min(t, TOTAL - 1), sch)).toFixed(1)}`,
      );
    return pts.join("");
  };
  const lossPath = (r: ScheduleRow) =>
    r.evalSteps
      .slice(0, st.step + 1)
      .map((t, i) => [t, r.meanEvals[i]] as const)
      .filter(([, v]) => v !== null)
      .map(
        ([t, v], i) =>
          `${i ? "L" : "M"}${x(t).toFixed(1)},${y2(v!).toFixed(1)}`,
      )
      .join("");

  const visual = (
    <div ref={box.ref} className="min-w-0">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="mx-auto w-full text-neutral-800 dark:text-neutral-200"
        role="img"
        aria-label={`Top: learning rate against step for ${names.join(", ")}. Bottom: their validation loss, drawn up to step ${s.step}.`}
      >
        <text x={left} y={top1 - 6} fontSize={fs(12)} fill="currentColor">
          learning rate (peak {lrMax})
        </text>
        <rect
          x={left}
          y={top1}
          width={W - left - right}
          height={h1}
          fill="none"
          stroke="currentColor"
          strokeOpacity={0.2}
        />
        {rows.map((r, k) => (
          <path
            key={r.name}
            d={lrPath(r)}
            fill="none"
            stroke={COLOURS[k]}
            strokeWidth={2}
            strokeDasharray={DASH[k]}
          />
        ))}
        <text x={left} y={top2 - 6} fontSize={fs(12)} fill="currentColor">
          validation loss (mean of 3 seeds)
        </text>
        {[2.0, 2.5, 3.0, 3.5, 4.0].map((v) => (
          <g key={v}>
            <line
              x1={left}
              x2={W - right}
              y1={y2(v)}
              y2={y2(v)}
              stroke="currentColor"
              strokeOpacity={0.12}
            />
            <text
              x={left - 4}
              y={y2(v) + fs(11) / 3}
              fontSize={fs(11)}
              textAnchor="end"
              fill="currentColor"
            >
              {v.toFixed(1)}
            </text>
          </g>
        ))}
        {rows.map((r, k) => (
          <path
            key={r.name}
            d={lossPath(r)}
            fill="none"
            stroke={COLOURS[k]}
            strokeWidth={2}
            strokeDasharray={DASH[k]}
          />
        ))}
        <line
          x1={x(s.step)}
          x2={x(s.step)}
          y1={top1}
          y2={top2 + h2}
          stroke="currentColor"
          strokeOpacity={0.5}
          data-testid="cursor"
        />
        {rows.map((r, k) => (
          <circle
            key={r.name}
            cx={x(Math.min(s.step, TOTAL - 1))}
            cy={y1(s.lrs[k]!)}
            r={4}
            fill={COLOURS[k]}
            stroke="currentColor"
          />
        ))}
        {[0, 500, 1000, 1500, 2000].map((t) => (
          <text
            key={t}
            x={x(t)}
            y={top2 + h2 + fs(11) + 4}
            fontSize={fs(11)}
            textAnchor={t === 0 ? "start" : t === TOTAL ? "end" : "middle"}
            fill="currentColor"
          >
            {t}
          </text>
        ))}
        {rows.map((r, k) => {
          const col = k % perRow;
          const row = Math.floor(k / perRow);
          const lx = left + col * ((W - left) / perRow);
          const ly = legendY + row * (fs(12) + 10);
          return (
            <g key={r.name}>
              <line
                x1={lx}
                x2={lx + 22}
                y1={ly - fs(12) / 3}
                y2={ly - fs(12) / 3}
                stroke={COLOURS[k]}
                strokeWidth={3}
                strokeDasharray={DASH[k]}
              />
              <text x={lx + 26} y={ly} fontSize={fs(12)} fill="currentColor">
                {r.name}
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
      title="Four schedules, one peak learning rate"
      summary="AdamW at peak learning rate 0.01, 2,000 steps, three seeds each. Top: the learning rate the schedule sets at each step (the trainer's schedules.ts). Bottom: the validation loss the runs measured, every 50 steps."
      stepper={st}
      stepLabel="point"
      countFrom={0}
      caption={scheduleCaption(s, names)}
      visual={visual}
      equation={children}
      hl={hl}
      onEquationHover={setHover}
    />
  );
}
