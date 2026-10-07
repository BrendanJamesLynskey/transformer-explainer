"use client";

/**
 * Chapter 04's animation: the feed-forward network at one position. The
 * d_model inputs expand to d_ff pre-activations (bars growing), GELU is
 * applied neuron by neuron (the current input marked on the curve), the
 * result contracts back to d_model, and the neurons that fired are
 * picked out. Every state comes from `ffnStates` over the
 * `/api/compute/ffn` trace, computed in the browser (`computeFfn`).
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { VectorStrip } from "@/components/viz/VectorStrip";
import {
  PHONE_BELOW,
  useContainerWidth,
} from "@/components/viz/useContainerWidth";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { FFN_HL, ffnCaption, ffnStates, geluCurve } from "@/lib/anim/ffn-steps";
import { maxAbs, showChar, sig } from "@/lib/anim/format";
import { DEFAULT_PARAMS, computeFfn } from "@/lib/compute/traces";
import { MUTED, PART_COLOUR } from "@/lib/viz/palette";

export default function FFNAnimation({
  testId = "ffn-animation",
  children,
}: {
  testId?: string;
  children?: ReactNode;
}): JSX.Element {
  const [text, setText] = useState(DEFAULT_PARAMS.text);
  const [pos, setPos] = useState(5);
  const [hover, setHover] = useState<string | null>(null);
  const tr = useMemo(() => computeFfn({ ...DEFAULT_PARAMS, text }), [text]);
  const states = useMemo(() => ffnStates(tr, pos), [tr, pos]);
  const st = useStepper(states.length, {
    stepMs: 450,
    smooth: true,
    resetKey: `${text}|${pos}`,
  });
  const box = useContainerWidth();
  const phone = box.width > 0 && box.width < PHONE_BELOW;
  const W = phone ? 360 : 720;
  const font = useSvgFont(W);
  const fs = font.fs;
  const s = states[st.step]!;
  const D = s.x.length;
  const F = tr.pre[0]!.length;
  const hl = FFN_HL[s.phase];
  const on = (k: string) => hl === k || hover === k;
  // the bars grow during the expand step (smooth), then stay
  const grow =
    s.phase === "expand"
      ? Math.min(1, st.frac * 1.5 + (st.playing ? 0 : 1))
      : 1;

  // ─── layout ──────────────────────────────────────────────
  const cell = phone ? 20 : 26;
  const xY = fs(11) + 8;
  const barsX = 0;
  const curveW = phone ? W : 220;
  const barsW = phone ? W : W - curveW - 30;
  const bw = barsW / F;
  const barsY = xY + cell + fs(11) * 2 + 20;
  const barsH = phone ? 150 : 170;
  const mid = barsY + barsH / 2;
  const preMax = Math.max(1e-9, maxAbs(tr.pre[s.pos]!));
  const yOf = (v: number) => mid - (v / preMax) * (barsH / 2 - 4);
  const cx = phone ? 0 : barsW + 30;
  const cy = phone ? barsY + barsH + fs(11) + 24 : barsY;
  const ch = phone ? 140 : barsH;
  const yOut = (phone ? cy + ch + 20 : barsY + barsH + 18) + fs(11) + 8;
  const H = yOut + cell + 12;

  // the GELU chart spans the pre-activations seen at this position
  const lo = Math.min(-2, -preMax * 1.1);
  const hi = Math.max(2, preMax * 1.1);
  const curve = geluCurve(lo, hi);
  const gyLo = -0.2;
  const gyHi = hi;
  const px = (x: number) => cx + 30 + ((x - lo) / (hi - lo)) * (curveW - 40);
  const py = (y: number) =>
    cy + ch - 16 - ((y - gyLo) / (gyHi - gyLo)) * (ch - 26);
  const path = curve
    .map(
      ([x, y], k) => `${k ? "L" : "M"}${px(x).toFixed(1)},${py(y).toFixed(1)}`,
    )
    .join("");
  const n = s.neuron;
  const fired = new Set(s.fired ?? []);

  const visual = (
    <div ref={box.ref} className="min-w-0">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="mx-auto w-full text-neutral-800 dark:text-neutral-200"
        role="img"
        aria-label="The input vector as coloured cells, the hidden layer as bars around a zero line with GELU outputs filled in, the GELU curve with the current neuron marked, and the output vector"
      >
        <VectorStrip
          x={phone ? (W - D * cell) / 2 : 0}
          y={xY}
          values={s.x}
          cell={cell}
          height={cell}
          max={maxAbs(s.x)}
          label={`x = LN₂(h) at position ${s.pos} ('${showChar(tr.tokens[s.pos] ?? "?")}'): ${D} numbers`}
          fontSize={fs(11)}
          activeColour={PART_COLOUR.ln}
        />

        {/* the hidden layer */}
        <text x={barsX} y={barsY - 8} fontSize={fs(11)} fill="currentColor">
          {s.pre
            ? `hidden layer: ${F} neurons (outline u, filled GELU(u))`
            : `hidden layer: ${F} neurons`}
        </text>
        <line
          x1={barsX}
          x2={barsX + barsW}
          y1={mid}
          y2={mid}
          stroke="currentColor"
          strokeOpacity={0.4}
        />
        {Array.from({ length: F }, (_, k) => {
          const u = s.pre?.[k];
          const g = s.act?.[k];
          const x = barsX + k * bw + 1;
          const w = Math.max(1, bw - 2);
          const isN = k === n;
          const muted = s.phase === "fired" && !fired.has(k);
          return (
            <g
              key={k}
              data-testid={`neuron-${k}`}
              data-fired={
                s.phase === "fired" && fired.has(k) ? "true" : undefined
              }
            >
              {u !== undefined && u !== null && (
                <rect
                  x={x}
                  y={Math.min(mid, yOf(u * grow))}
                  width={w}
                  height={Math.abs(yOf(u * grow) - mid)}
                  fill="none"
                  stroke={on("up") ? PART_COLOUR.ffn : "currentColor"}
                  strokeOpacity={muted ? 0.25 : 0.6}
                />
              )}
              {g !== undefined && g !== null && (
                <rect
                  x={x}
                  y={Math.min(mid, yOf(g))}
                  width={w}
                  height={Math.max(0.5, Math.abs(yOf(g) - mid))}
                  fill={muted ? MUTED.light : PART_COLOUR.ffn}
                  fillOpacity={muted ? 0.5 : 0.85}
                />
              )}
              {isN && (
                <rect
                  x={x - 1}
                  y={barsY}
                  width={w + 2}
                  height={barsH}
                  fill="none"
                  stroke="currentColor"
                  strokeWidth={2}
                />
              )}
            </g>
          );
        })}

        {/* the GELU curve */}
        <g data-testid="gelu-chart">
          <text
            x={cx}
            y={cy - 10 - fs(11)}
            fontSize={fs(11)}
            fill="currentColor"
          >
            GELU(u)
          </text>
          <text x={cx} y={cy - 6} fontSize={fs(11)} fill="currentColor">
            shaded: this position&apos;s u
          </text>
          <rect
            x={px(-preMax)}
            y={py(gyHi)}
            width={px(preMax) - px(-preMax)}
            height={py(gyLo) - py(gyHi)}
            fill={PART_COLOUR.ffn}
            fillOpacity={0.12}
          />
          <line
            x1={px(lo)}
            x2={px(hi)}
            y1={py(0)}
            y2={py(0)}
            stroke="currentColor"
            strokeOpacity={0.3}
          />
          <line
            x1={px(0)}
            x2={px(0)}
            y1={py(gyLo)}
            y2={py(gyHi)}
            stroke="currentColor"
            strokeOpacity={0.3}
          />
          <path
            d={path}
            fill="none"
            stroke={PART_COLOUR.ffn}
            strokeWidth={on("gelu") ? 3 : 2}
          />
          {n >= 0 && s.pre && (
            <g>
              <line
                x1={px(s.pre[n]!)}
                x2={px(s.pre[n]!)}
                y1={py(0)}
                y2={py(s.act![n]!)}
                stroke="currentColor"
                strokeDasharray="3 2"
              />
              <circle
                cx={px(s.pre[n]!)}
                cy={py(s.act![n]!)}
                r={5}
                fill={PART_COLOUR.ffn}
                stroke="currentColor"
              />
              <text
                x={cx + 30}
                y={cy + ch + 2}
                fontSize={fs(11)}
                fill="currentColor"
              >
                u{n} = {sig(s.pre[n]!)} → {sig(s.act![n]!)}
              </text>
            </g>
          )}
        </g>

        {s.out && (
          <VectorStrip
            x={phone ? (W - D * cell) / 2 : 0}
            y={yOut}
            values={s.out}
            cell={cell}
            height={cell}
            max={maxAbs(s.out)}
            label={`y = GELU(u) W₂ + b₂: back to ${D} numbers`}
            fontSize={fs(11)}
            testId="ffn-out"
          />
        )}
      </svg>
    </div>
  );

  return (
    <AnimationPanel
      testId={testId}
      title="Inside the feed-forward network"
      summary={`${D} numbers expand to ${F} (d_ff = ${F / D}·d here; GPT-2 uses 4·d), GELU bends each, ${D} come back. Block 1, your input.`}
      stepper={st}
      stepLabel="step"
      caption={ffnCaption(s)}
      visual={visual}
      equation={children}
      hl={hl}
      onEquationHover={setHover}
      params={
        <>
          <label className="flex min-w-0 flex-col gap-1 text-sm">
            <span className="font-medium text-neutral-700 dark:text-neutral-300">
              Your input (up to {tr.tokens.length} characters)
            </span>
            <input
              type="text"
              value={text}
              maxLength={tr.tokens.length}
              onChange={(e) => setText(e.target.value.toLowerCase())}
              aria-label="FFN animation input"
              className="focus-ring h-11 rounded border border-neutral-300 bg-white px-2 font-mono text-sm dark:border-neutral-700 dark:bg-neutral-950"
            />
          </label>
          <label className="flex min-w-0 flex-col gap-1 text-sm">
            <span className="font-medium text-neutral-700 dark:text-neutral-300">
              Position
            </span>
            <select
              value={pos}
              onChange={(e) => setPos(Number(e.target.value))}
              aria-label="FFN position"
              className="focus-ring h-11 rounded border border-neutral-300 bg-white px-2 font-mono text-sm dark:border-neutral-700 dark:bg-neutral-950"
            >
              {tr.tokens.map((c, i) => (
                <option key={i} value={i}>
                  {i}: {showChar(c)}
                </option>
              ))}
            </select>
          </label>
        </>
      }
    />
  );
}
