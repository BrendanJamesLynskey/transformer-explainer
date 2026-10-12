"use client";

/**
 * Chapter 7's hero: scale through depth at initialisation. An 8-block
 * version of Part 1's model, pre-LN beside post-LN, on "alice was": the
 * forward pass fills in the RMS of the residual stream entering each block;
 * the backward pass then fills in each block's gradient norm, from the top.
 * Computed in the browser by the trainer's forward and backward passes
 * (model-grad.ts, which also does post-LN; `depthStates`).
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
  DEPTH,
  DEPTH_STDS,
  depthCaption,
  depthStates,
} from "@/lib/anim/training/depth-steps";
import { OKABE_ITO } from "@/lib/viz/palette";

const SELECT =
  "focus-ring h-11 rounded border border-neutral-300 bg-white px-2 text-sm dark:border-neutral-700 dark:bg-neutral-950";

export default function DepthAnimation({
  children,
  testId = "depth-animation",
}: {
  children?: ReactNode;
  testId?: string;
}): JSX.Element {
  const [std, setStd] = useState<number>(DEPTH_STDS[1]);
  const states = useMemo(() => depthStates(std), [std]);
  const st = useStepper(states.length, { stepMs: 600, resetKey: String(std) });
  const [, setHover] = useState<string | null>(null);
  const box = useContainerWidth();
  const phone = box.width > 0 && box.width < PHONE_BELOW;
  const W = phone ? 360 : 720;
  const font = useSvgFont(W);
  const fs = font.fs;
  const s = states[st.step]!;
  const last = states[states.length - 1]!;
  const rmsMax = Math.max(...last.rms[0], ...last.rms[1]);
  const gMax = Math.max(...last.grad[0], ...last.grad[1]);
  const hl = s.phase === "forward" ? "resid" : s.layer >= 0 ? "grad" : "resid";

  // ─── layout: two panels side by side (stacked on phones) ─────
  const panelW = phone ? W : W / 2;
  const rowH = 90;
  const labelH = fs(12) + 8;
  const panelH = labelH + 2 * rowH + 2 * (fs(11) + 10) + fs(11) + 6;
  const H = phone ? 2 * panelH + 8 : panelH;
  const panels = [
    { name: "pre-LN", k: 0 as const },
    { name: "post-LN", k: 1 as const },
  ];

  const visual = (
    <div ref={box.ref} className="min-w-0">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="mx-auto w-full text-neutral-800 dark:text-neutral-200"
        role="img"
        aria-label={`Residual-stream RMS into each of ${DEPTH} blocks and each block's gradient norm, pre-LN and post-LN, matrices initialised at standard deviation ${std}`}
      >
        {panels.map(({ name, k }) => {
          const ox = phone ? 0 : k * panelW;
          const oy = phone ? k * (panelH + 8) : 0;
          const pad = 10;
          const n = DEPTH + 1;
          const bw = (panelW - 2 * pad) / n;
          const r1 = oy + labelH + fs(11) + 6;
          const r2 = r1 + rowH + fs(11) + 10;
          return (
            <g key={name} data-testid={`depth-${k ? "post" : "pre"}`}>
              <text
                x={ox + pad}
                y={oy + fs(12)}
                fontSize={fs(12)}
                fontWeight={600}
                fill="currentColor"
              >
                {name}
              </text>
              <text
                x={ox + pad}
                y={r1 - 4}
                fontSize={fs(11)}
                fill="currentColor"
                fillOpacity={0.8}
              >
                residual RMS (max {rmsMax.toFixed(2)})
              </text>
              <text
                x={ox + pad}
                y={r2 - 4}
                fontSize={fs(11)}
                fill="currentColor"
                fillOpacity={0.8}
              >
                ‖gradient‖ per block (max {gMax.toPrecision(2)})
              </text>
              {Array.from({ length: n }, (_, i) => {
                const v = s.rms[k][i];
                const h = v === undefined ? 0 : (rowH * v) / rmsMax;
                const active = s.phase === "forward" && i === s.layer;
                return (
                  <rect
                    key={`r${i}`}
                    x={ox + pad + i * bw + 2}
                    y={r1 + rowH - h}
                    width={bw - 4}
                    height={h}
                    fill={OKABE_ITO.sky}
                    fillOpacity={active ? 1 : 0.55}
                    stroke={active ? "currentColor" : "none"}
                  />
                );
              })}
              {Array.from({ length: DEPTH }, (_, i) => {
                const j = i - s.layer;
                const v =
                  s.phase === "backward" && j >= 0 ? s.grad[k][j] : undefined;
                const h = v === undefined ? 0 : (rowH * v) / gMax;
                const active = s.phase === "backward" && i === s.layer;
                return (
                  <rect
                    key={`g${i}`}
                    x={ox + pad + i * bw + 2}
                    y={r2 + rowH - h}
                    width={bw - 4}
                    height={h}
                    fill={OKABE_ITO.orange}
                    fillOpacity={active ? 1 : 0.55}
                    stroke={active ? "currentColor" : "none"}
                  />
                );
              })}
              <line
                x1={ox + pad}
                x2={ox + panelW - pad}
                y1={r1 + rowH}
                y2={r1 + rowH}
                stroke="currentColor"
                strokeOpacity={0.3}
              />
              <line
                x1={ox + pad}
                x2={ox + panelW - pad}
                y1={r2 + rowH}
                y2={r2 + rowH}
                stroke="currentColor"
                strokeOpacity={0.3}
              />
              {Array.from({ length: n }, (_, i) => (
                <text
                  key={`l${i}`}
                  x={ox + pad + i * bw + bw / 2}
                  y={r2 + rowH + fs(11) + 2}
                  fontSize={fs(11)}
                  textAnchor="middle"
                  fill="currentColor"
                >
                  {i < DEPTH ? i + 1 : "out"}
                </text>
              ))}
            </g>
          );
        })}
      </svg>
    </div>
  );

  const params = (
    <label className="flex min-w-0 flex-col gap-1 text-sm">
      <span className="font-medium text-neutral-700 dark:text-neutral-300">
        Initial weight scale
      </span>
      <select
        value={std}
        onChange={(e) => setStd(Number(e.target.value))}
        aria-label="Initial standard deviation of the weight matrices"
        className={SELECT}
      >
        <option value={0.25}>std 0.25 (= 1/√16, Xavier-like)</option>
        <option value={0.02}>std 0.02 (the site&apos;s, GPT-2&apos;s)</option>
      </select>
    </label>
  );

  return (
    <AnimationPanel
      testId={testId}
      title="Scale through depth: pre-LN vs post-LN"
      summary={`Part 1's model with ${DEPTH} blocks instead of 2, at initialisation, on "alice was". Blue: how large the residual stream is entering each block. Orange: how large each block's weight gradients are, filled in by the backward pass from the top.`}
      stepper={st}
      stepLabel="stage"
      caption={depthCaption(s)}
      visual={visual}
      equation={children}
      hl={hl}
      onEquationHover={setHover}
      params={params}
    />
  );
}
