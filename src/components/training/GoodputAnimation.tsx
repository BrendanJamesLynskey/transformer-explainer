"use client";

/**
 * Chapter 15's hero: a week of a job the size of Llama 3 405B's, segment by
 * segment (systems.ts `simulateGoodput`, seeded): a row per day; training
 * green, checkpoints blue, work lost to a failure vermillion (hatched),
 * restarts grey (hatched). The checkpoint interval re-runs the simulation.
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
  GOODPUT_HL,
  goodputCaption,
  goodputStates,
} from "@/lib/anim/training/goodput-steps";
import {
  simulateGoodput,
  youngInterval,
  type Segment,
} from "@/lib/training/systems";
import { OKABE_ITO } from "@/lib/viz/palette";

const SELECT =
  "focus-ring h-11 w-full rounded border border-neutral-300 bg-white px-2 text-sm dark:border-neutral-700 dark:bg-neutral-950";

export type GoodputParams = {
  mtbf: number;
  delta: number;
  restart: number;
  horizon: number;
  tau: number;
  seed: number;
};

const FILL: Record<Segment["kind"], string> = {
  work: OKABE_ITO.green,
  checkpoint: OKABE_ITO.blue,
  lost: "url(#gp-lost)",
  restart: "url(#gp-restart)",
};

export default function GoodputAnimation({
  params: g,
  children,
  testId = "goodput-animation",
}: {
  params: GoodputParams;
  children?: ReactNode;
  testId?: string;
}): JSX.Element {
  const [tau, setTau] = useState(g.tau);
  const sim = useMemo(
    () => simulateGoodput(tau, g.delta, g.mtbf, g.restart, g.horizon, g.seed),
    [tau, g],
  );
  const states = useMemo(
    () => goodputStates(sim.segments, g.horizon),
    [sim, g.horizon],
  );
  const st = useStepper(states.length, {
    stepMs: 140,
    resetKey: String(tau),
  });
  const [, setHover] = useState<string | null>(null);
  const box = useContainerWidth();
  const phone = box.width > 0 && box.width < PHONE_BELOW;
  const W = phone ? 360 : 720;
  const font = useSvgFont(W);
  const fs = font.fs;
  const s = states[st.step]!;
  const days = Math.round(g.horizon / 86400);

  const left = phone ? 40 : 56;
  const top = fs(12) + 10;
  const rowH = phone ? 20 : 24;
  const plotW = W - left - 6;
  const H = top + days * (rowH + 6) + fs(11) + 14;
  const X = (sec: number) => left + (plotW * sec) / 86400;

  // split each shown segment at midnight into per-day pieces
  const pieces: {
    kind: Segment["kind"];
    day: number;
    a: number;
    b: number;
    cur: boolean;
  }[] = [];
  sim.segments.slice(0, st.step + 1).forEach((seg, i) => {
    let a = seg.start;
    const end = Math.min(seg.end, g.horizon);
    while (a < end) {
      const day = Math.floor(a / 86400);
      const b = Math.min(end, (day + 1) * 86400);
      pieces.push({
        kind: seg.kind,
        day,
        a: a - day * 86400,
        b: b - day * 86400,
        cur: i === st.step,
      });
      a = b;
    }
  });

  const visual = (
    <div ref={box.ref} className="min-w-0">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="mx-auto w-full text-neutral-800 dark:text-neutral-200"
        role="img"
        aria-label={`A week of training with failures and checkpoints every ${Math.round(tau / 60)} minutes: goodput ${(100 * sim.goodput).toFixed(1)}%`}
      >
        <defs>
          <pattern
            id="gp-lost"
            width={6}
            height={6}
            patternUnits="userSpaceOnUse"
            patternTransform="rotate(45)"
          >
            <rect
              width={6}
              height={6}
              fill={OKABE_ITO.vermillion}
              fillOpacity={0.35}
            />
            <line
              x1={0}
              y1={0}
              x2={0}
              y2={6}
              stroke={OKABE_ITO.vermillion}
              strokeWidth={2.5}
            />
          </pattern>
          <pattern
            id="gp-restart"
            width={6}
            height={6}
            patternUnits="userSpaceOnUse"
            patternTransform="rotate(-45)"
          >
            <rect width={6} height={6} fill="#7f7f7f" fillOpacity={0.3} />
            <line
              x1={0}
              y1={0}
              x2={0}
              y2={6}
              stroke="#7f7f7f"
              strokeWidth={2}
            />
          </pattern>
        </defs>
        <text x={left} y={fs(12)} fontSize={fs(12)} fill="currentColor">
          one row per day, midnight to midnight
        </text>
        {Array.from({ length: days }, (_, d) => (
          <g key={d}>
            <text
              x={left - 4}
              y={top + d * (rowH + 6) + rowH / 2 + fs(11) / 3}
              fontSize={fs(11)}
              textAnchor="end"
              fill="currentColor"
            >
              {phone ? `d${d + 1}` : `day ${d + 1}`}
            </text>
            <rect
              x={left}
              y={top + d * (rowH + 6)}
              width={plotW}
              height={rowH}
              fill="none"
              stroke="currentColor"
              strokeOpacity={0.2}
            />
          </g>
        ))}
        {pieces.map((p, i) => (
          <rect
            key={i}
            x={X(p.a)}
            y={top + p.day * (rowH + 6)}
            width={Math.max(
              p.kind === "checkpoint" ? 1.5 : 0.5,
              X(p.b) - X(p.a),
            )}
            height={rowH}
            fill={FILL[p.kind]}
            stroke={p.cur ? "currentColor" : "none"}
            strokeWidth={p.cur ? 1.5 : 0}
            data-kind={p.kind}
          />
        ))}
        {[0, 6, 12, 18, 24].map((h) => (
          <text
            key={h}
            x={X(h * 3600)}
            y={top + days * (rowH + 6) + fs(11)}
            fontSize={fs(11)}
            textAnchor={h === 0 ? "start" : h === 24 ? "end" : "middle"}
            fill="currentColor"
          >
            {h}:00
          </text>
        ))}
      </svg>
    </div>
  );

  const young = youngInterval(g.delta, g.mtbf);
  const stats = (
    <p
      className="font-mono text-xs text-neutral-700 dark:text-neutral-300"
      data-testid="goodput-stats"
    >
      failures so far {s.failures} · lost {(s.lost / 3600).toFixed(1)} h ·
      restarting {(s.restarting / 3600).toFixed(1)} h · checkpointing{" "}
      {(s.checkpointing / 3600).toFixed(1)} h · whole week: goodput{" "}
      {(100 * sim.goodput).toFixed(1)}%, {sim.failures} failures · Young&apos;s
      interval √(2δM) = {Math.round(young / 60)} min
    </p>
  );

  const params = (
    <label className="flex min-w-0 flex-col gap-1 text-sm">
      <span className="font-medium text-neutral-700 dark:text-neutral-300">
        Checkpoint every
      </span>
      <select
        value={tau}
        onChange={(e) => setTau(Number(e.target.value))}
        className={SELECT}
        aria-label="Checkpoint interval"
      >
        <option value={900}>15 minutes</option>
        <option value={3600}>1 hour</option>
        <option value={14400}>4 hours</option>
      </select>
    </label>
  );

  return (
    <AnimationPanel
      testId={testId}
      title="A week of a large job: failures, checkpoints, goodput"
      summary={`Failures arrive at random, on average every ${(g.mtbf / 3600).toFixed(1)} hours (Llama 3 405B's rate on 16,384 GPUs); a checkpoint takes ${Math.round(g.delta)} s, a restart ${Math.round(g.restart / 60)} min (illustrative). A failure throws away everything since the last checkpoint. Seeded, so the same week every time.`}
      stepper={st}
      stepLabel="segment"
      caption={goodputCaption(s)}
      visual={visual}
      stats={stats}
      equation={children}
      hl={GOODPUT_HL[s.segment.kind]}
      onEquationHover={setHover}
      params={params}
    />
  );
}
