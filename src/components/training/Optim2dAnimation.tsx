"use client";

/**
 * Chapter 5: SGD, momentum, Nesterov momentum, Adam and AdamW on a 2-D
 * valley 50 times steeper across than along. Nesterov (brief 44) is drawn
 * dashed as well as in its own colour, so it reads apart from momentum
 * without colour. Each dot is one optimiser's (x, y) after t
 * steps, computed by the trainer's own optimizers.ts (`optim2dStates`);
 * Adam's two moment estimates are printed under the picture.
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import {
  PHONE_BELOW,
  useContainerWidth,
} from "@/components/viz/useContainerWidth";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { sig } from "@/lib/anim/format";
import {
  OPTIM2D,
  VALLEY,
  optim2dCaption,
  optim2dStates,
} from "@/lib/anim/training/optim2d-steps";
import { OKABE_ITO } from "@/lib/viz/palette";

const COLOURS = [
  OKABE_ITO.orange,
  OKABE_ITO.green,
  OKABE_ITO.blue,
  OKABE_ITO.purple,
  OKABE_ITO.vermillion,
];
const MARKERS = ["circle", "square", "triangle", "diamond", "down"] as const;
/** Nesterov's path is dashed (never colour alone). */
const DASH: Record<string, string | undefined> = { nesterov: "6 4" };

function Marker({
  kind,
  x,
  y,
  r,
  fill,
}: {
  kind: (typeof MARKERS)[number];
  x: number;
  y: number;
  r: number;
  fill: string;
}): JSX.Element {
  if (kind === "circle")
    return <circle cx={x} cy={y} r={r} fill={fill} stroke="currentColor" />;
  if (kind === "square")
    return (
      <rect
        x={x - r}
        y={y - r}
        width={2 * r}
        height={2 * r}
        fill={fill}
        stroke="currentColor"
      />
    );
  if (kind === "down")
    return (
      <path
        d={`M${x},${y + r * 1.2}L${x + r * 1.1},${y - r * 0.8}L${x - r * 1.1},${y - r * 0.8}Z`}
        fill={fill}
        stroke="currentColor"
      />
    );
  if (kind === "triangle")
    return (
      <path
        d={`M${x},${y - r * 1.2}L${x + r * 1.1},${y + r * 0.8}L${x - r * 1.1},${y + r * 0.8}Z`}
        fill={fill}
        stroke="currentColor"
      />
    );
  return (
    <path
      d={`M${x},${y - r * 1.3}L${x + r * 1.1},${y}L${x},${y + r * 1.3}L${x - r * 1.1},${y}Z`}
      fill={fill}
      stroke="currentColor"
    />
  );
}

export default function Optim2dAnimation({
  children,
  testId = "optim2d-animation",
}: {
  children?: ReactNode;
  testId?: string;
}): JSX.Element {
  const states = useMemo(() => optim2dStates(), []);
  const st = useStepper(states.length, { stepMs: 300 });
  const [hover, setHover] = useState<string | null>(null);
  const box = useContainerWidth();
  const phone = box.width > 0 && box.width < PHONE_BELOW;
  const W = phone ? 360 : 720;
  const font = useSvgFont(W);
  const fs = font.fs;
  const s = states[st.step]!;
  const hl = st.step === 0 ? "grad" : "update";

  // ─── layout: equal scales in x and y ─────────────────────
  const xLo = -3;
  const xHi = 3;
  const yLo = -2;
  const yHi = 3.2;
  const pad = 6;
  const k = (W - 2 * pad) / (xHi - xLo);
  const plotH = (yHi - yLo) * k;
  const legendY = plotH + 2 * pad + fs(12) + 6;
  const perRow = phone ? 2 : 3;
  const legendRows = Math.ceil(OPTIM2D.length / perRow);
  const H = legendY + (legendRows - 1) * (fs(12) + 10) + fs(12) + 10;
  const px = (x: number) => pad + (x - xLo) * k;
  const py = (y: number) => pad + (yHi - y) * k;
  const levels = [0.05, 0.3, 1.5, 6, 20, 60];

  const visual = (
    <div ref={box.ref} className="min-w-0">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="mx-auto w-full text-neutral-800 dark:text-neutral-200"
        role="img"
        aria-label="Contours of a long, narrow valley with its minimum at (2, 1) and the origin marked; five paths, one per optimiser, from (−2.5, −1.5) towards the minimum; Nesterov's is dashed"
      >
        <defs>
          <clipPath id="valley-clip">
            <rect x={pad} y={pad} width={W - 2 * pad} height={plotH} />
          </clipPath>
        </defs>
        <rect
          x={pad}
          y={pad}
          width={W - 2 * pad}
          height={plotH}
          fill="none"
          stroke="currentColor"
          strokeOpacity={0.3}
        />
        <g clipPath="url(#valley-clip)">
          {levels.map((c) => (
            <ellipse
              key={c}
              cx={px(VALLEY.xStar)}
              cy={py(VALLEY.yStar)}
              rx={Math.sqrt((2 * c) / VALLEY.a) * k}
              ry={Math.sqrt((2 * c) / VALLEY.b) * k}
              fill="none"
              stroke="currentColor"
              strokeOpacity={0.22}
            />
          ))}
          <line
            x1={px(0)}
            x2={px(0)}
            y1={pad}
            y2={pad + plotH}
            stroke="currentColor"
            strokeOpacity={0.12}
          />
          <line
            x1={pad}
            x2={W - pad}
            y1={py(0)}
            y2={py(0)}
            stroke="currentColor"
            strokeOpacity={0.12}
          />
          {OPTIM2D.map((o, i) => {
            const pts = states.slice(0, st.step + 1).map((q) => q.points[i]!);
            const d = pts
              .map(
                ([x, y], j) =>
                  `${j ? "L" : "M"}${px(x).toFixed(1)},${py(y).toFixed(1)}`,
              )
              .join("");
            return (
              <path
                key={o.kind}
                d={d}
                fill="none"
                stroke={COLOURS[i]}
                strokeWidth={2}
                strokeOpacity={0.85}
                strokeDasharray={DASH[o.kind]}
              />
            );
          })}
          {OPTIM2D.map((o, i) => {
            const [x, y] = s.points[i]!;
            return (
              <g key={o.kind} data-testid={`opt-${o.kind}`}>
                <Marker
                  kind={MARKERS[i]!}
                  x={px(x)}
                  y={py(y)}
                  r={6}
                  fill={COLOURS[i]!}
                />
              </g>
            );
          })}
        </g>
        <circle
          cx={px(VALLEY.xStar)}
          cy={py(VALLEY.yStar)}
          r={3}
          fill="currentColor"
        />
        <text
          x={px(VALLEY.xStar)}
          y={py(VALLEY.yStar) - 34}
          fontSize={fs(11)}
          textAnchor="middle"
          fill="currentColor"
        >
          minimum (2, 1)
        </text>
        <text
          x={px(0) + 6}
          y={py(0) + fs(11) + 4}
          fontSize={fs(11)}
          fill="currentColor"
          fillOpacity={0.8}
        >
          origin
        </text>
        {OPTIM2D.map((o, i) => {
          const col = i % perRow;
          const row = Math.floor(i / perRow);
          const lx = pad + col * ((W - 2 * pad) / perRow);
          const ly = legendY + row * (fs(12) + 10);
          return (
            <g key={o.kind} data-legend={o.kind}>
              {DASH[o.kind] && (
                <line
                  x1={lx}
                  x2={lx + 12}
                  y1={ly + 4}
                  y2={ly + 4}
                  stroke={COLOURS[i]}
                  strokeWidth={2}
                  strokeDasharray="3 2"
                />
              )}
              <Marker
                kind={MARKERS[i]!}
                x={lx + 6}
                y={ly - fs(12) / 3}
                r={5}
                fill={COLOURS[i]!}
              />
              <text x={lx + 16} y={ly} fontSize={fs(12)} fill="currentColor">
                {`${o.label} η ${o.lr}${o.weightDecay ? `, λ ${o.weightDecay}` : ""}${o.momentum ? `, μ ${o.momentum}` : ""}`}
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
      title="Five optimisers in a narrow valley"
      summary="f(x, y) = ½((x − 2)² + 50 (y − 1)²), from (−2.5, −1.5), 60 steps each. Plain SGD's step must stay small for the steep direction, so it crawls along the shallow one; momentum builds speed along it; Nesterov, at the same settings, takes the gradient where momentum is about to carry it and stops swinging sooner; Adam rescales each direction; AdamW's decay pulls it towards the origin."
      stepper={st}
      stepLabel="step"
      countFrom={0}
      caption={optim2dCaption(s)}
      visual={visual}
      equation={children}
      hl={hl}
      onEquationHover={setHover}
      stats={
        <p
          className="font-mono text-xs text-neutral-700 dark:text-neutral-300"
          data-testid="adam-moments"
        >
          Adam after {s.t} steps: m = ({sig(s.adamM[0]!)}, {sig(s.adamM[1]!)}) ·
          v = ({sig(s.adamV[0]!)}, {sig(s.adamV[1]!)}){hover ? "" : ""}
        </p>
      }
    />
  );
}
