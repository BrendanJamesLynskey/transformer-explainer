"use client";

/**
 * Chapter 05's animation: LayerNorm and the residual stream, for one
 * position of block 0. The block diagram at the top shows where the step
 * is (the stream runs straight along the bottom; each sub-layer branches
 * off through its LayerNorm and comes back in at ⊕). The dot plot shows
 * the d_model numbers as a distribution: the mean is found, the variance,
 * then the dots shift to mean 0 and spread to unit variance, then γ and β
 * apply. The strips show the stream, LN(x), the sub-layer's update Δ and
 * x + Δ. Every state comes from `lnStates` over a real run of block 0
 * (`residualTrace`), computed in the browser.
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
import { maxAbs, norm, showChar, sig } from "@/lib/anim/format";
import {
  LN_PHASES,
  lnCaption,
  lnHl,
  lnStates,
  meanStd,
  residualTrace,
  type LnState,
} from "@/lib/anim/ln-steps";
import { DEFAULT_PARAMS } from "@/lib/compute/traces";
import { OKABE_ITO, PART_COLOUR } from "@/lib/viz/palette";

const GAMMAS = [1, 0.5, 2] as const;
const BETAS = [0, 0.5, -0.5] as const;

/** Where the marker sits in the block diagram for a state (0 … 7). */
function anchor(s: LnState): number {
  const k = s.sub === 1 ? 0 : 4;
  if (s.phase === "input") return k;
  if (s.phase === "sublayer") return k + 2;
  if (s.phase === "residual") return k + 3;
  return k + 1; // the LayerNorm steps
}

export default function LayerNormAnimation({
  testId = "layernorm-animation",
  children,
}: {
  testId?: string;
  children?: ReactNode;
}): JSX.Element {
  const [text, setText] = useState(DEFAULT_PARAMS.text);
  const [pos, setPos] = useState(5);
  const [gamma, setGamma] = useState<number>(1);
  const [beta, setBeta] = useState<number>(0);
  const [hover, setHover] = useState<string | null>(null);
  const tr = useMemo(
    () => residualTrace({ ...DEFAULT_PARAMS, text }, gamma, beta),
    [text, gamma, beta],
  );
  const states = useMemo(() => lnStates(tr, pos), [tr, pos]);
  const st = useStepper(states.length, {
    stepMs: 1000,
    smooth: true,
    resetKey: `${text}|${pos}|${gamma}|${beta}`,
  });
  const box = useContainerWidth();
  const phone = box.width > 0 && box.width < PHONE_BELOW;
  const W = phone ? 360 : 720;
  const font = useSvgFont(W);
  const fs = font.fs;
  const s = states[st.step]!;
  const prev = states[Math.max(0, st.step - 1)]!;
  const D = s.x.length;
  const hl = lnHl(s);
  const on = (...keys: string[]) =>
    keys.includes(hl) || (hover !== null && keys.includes(hover));
  // progress into this step (paused: the step's final frame)
  const t = st.step === 0 ? 1 : Math.min(1, st.frac * 2 + (st.playing ? 0 : 1));
  const phaseIdx = LN_PHASES.indexOf(s.phase);

  // ─── layout: block diagram ───────────────────────────────
  const k = W / 720;
  const sy = fs(11) + 74; // the stream line
  const ly = sy - 46; // the sub-layer lane
  const bw = phone ? 46 : 74;
  const bh = Math.max(28, fs(11) + 14);
  // x of: stream in, LN, sub-layer, ⊕ (sub-layer 1), then the same for 2
  const ax = [40, 150, 250, 335, 400, 490, 590, 675].map((v) => v * k);
  const stage = (i: number) => ({
    x: ax[i]!,
    // on the stream, or just under a box (not over its label)
    y: i % 4 === 0 || i % 4 === 3 ? sy : ly + bh + 7,
  });
  const a = stage(anchor(prev));
  const b = stage(anchor(s));
  const g = st.step === 0 ? 1 : Math.min(1, st.frac * 3 + (st.playing ? 0 : 1));
  const mx = a.x + (b.x - a.x) * g;
  const my = a.y + (b.y - a.y) * g;
  const diagH = sy + 16;

  // ─── layout: dot plot and strips ─────────────────────────
  const cell = phone ? 20 : 20;
  const stripW = D * cell;
  const plotX = phone ? 10 : stripW + 50;
  const plotW = phone ? W - 20 : W - plotX - 10;
  const plotY = diagH + fs(11) * 2 + 18;
  const lanes = 4;
  const laneH = 16;
  const plotH = lanes * laneH + 18;
  const stripX = phone ? (W - stripW) / 2 : 0;
  const stripTop = phone
    ? plotY + plotH + fs(11) * 2 + 24
    : diagH + fs(11) + 22;
  const stripGap = cell + fs(11) + 14;
  const H = Math.max(plotY + plotH + fs(11) + 12, stripTop + 4 * stripGap);

  // the plot's range covers every vector this sub-layer shows
  const R = useMemo(() => {
    let m = 0;
    for (const x of states) m = Math.max(m, maxAbs(x.values));
    return Math.max(1, Math.ceil(m * 2) / 2);
  }, [states]);
  const px = (v: number) => plotX + ((v + R) / (2 * R)) * plotW;
  const glide = prev.sub === s.sub && st.step > 0 && t < 1;
  const shown = s.values.map((v, i) =>
    glide ? prev.values[i]! + (v - prev.values[i]!) * t : v,
  );
  const ms = meanStd(shown);
  const showMean = phaseIdx >= 1 && phaseIdx <= 5;
  const showStd = phaseIdx >= 2 && phaseIdx <= 5;
  const dotColour = phaseIdx >= 3 ? PART_COLOUR.ln : PART_COLOUR.emb;
  const subColour = s.sub === 1 ? PART_COLOUR.attn : PART_COLOUR.ffn;
  const xName = s.sub === 1 ? "x" : "h";
  const lnName = s.sub === 1 ? "LN₁" : "LN₂";

  const boxEl = (
    i: number,
    label: string,
    colour: string,
    lit: boolean,
    id: string,
  ) => (
    <g data-testid={id} data-active={anchor(s) === i || undefined}>
      <rect
        x={ax[i]! - bw / 2}
        y={ly}
        width={bw}
        height={bh}
        rx={6}
        fill={colour}
        fillOpacity={anchor(s) === i ? 0.3 : 0.06}
        stroke={lit || anchor(s) === i ? colour : "currentColor"}
        strokeOpacity={lit || anchor(s) === i ? 1 : 0.4}
        strokeWidth={lit || anchor(s) === i ? 2.5 : 1}
      />
      <text
        x={ax[i]!}
        y={ly + bh / 2 + fs(11) / 3}
        textAnchor="middle"
        fontSize={fs(11)}
        fill="currentColor"
      >
        {label}
      </text>
    </g>
  );
  const plus = (i: number, id: string) => {
    const lit = anchor(s) === i || hover === (i === 3 ? "res" : "res2");
    return (
      <g data-testid={id} data-active={anchor(s) === i || undefined}>
        <circle
          cx={ax[i]}
          cy={sy}
          r={10}
          className="fill-neutral-50 dark:fill-neutral-900"
          stroke={lit ? PART_COLOUR.emb : "currentColor"}
          strokeWidth={lit ? 2.5 : 1.2}
        />
        <text
          x={ax[i]}
          y={sy + fs(12) / 3}
          textAnchor="middle"
          fontSize={fs(12)}
          fill="currentColor"
        >
          +
        </text>
      </g>
    );
  };
  // a branch: up from the stream, through LN and the sub-layer, down into ⊕
  const branch = (i0: number) => {
    const x0 = ax[i0]! + (ax[i0 + 1]! - ax[i0]!) * 0.35;
    const y = ly + bh / 2;
    return `M${x0},${sy} L${x0},${y} L${ax[i0 + 1]! - bw / 2},${y} M${ax[i0 + 1]! + bw / 2},${y} L${ax[i0 + 2]! - bw / 2},${y} M${ax[i0 + 2]! + bw / 2},${y} L${ax[i0 + 3]!},${y} L${ax[i0 + 3]!},${sy - 10}`;
  };

  const visual = (
    <div ref={box.ref} className="min-w-0">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="mx-auto w-full text-neutral-800 dark:text-neutral-200"
        role="img"
        aria-label="A decoder block drawn as a residual stream with two branches, each through a LayerNorm and a sub-layer back into an addition; a dot plot of the vector's numbers with its mean and spread; and the stream, its normalised copy, the sub-layer's update and their sum as rows of coloured cells"
      >
        {/* the block diagram */}
        <text x={0} y={fs(11)} fontSize={fs(11)} fill="currentColor">
          block 1 at position {s.pos} (&apos;
          {showChar(tr.tokens[s.pos] ?? "?")}&apos;): the residual stream
        </text>
        <line
          x1={4}
          x2={W - 4}
          y1={sy}
          y2={sy}
          stroke={PART_COLOUR.emb}
          strokeWidth={on("x", "x2", "res", "res2") ? 5 : 3}
          strokeOpacity={0.7}
        />
        <path
          d={branch(0)}
          fill="none"
          stroke="currentColor"
          strokeOpacity={0.5}
        />
        <path
          d={branch(4)}
          fill="none"
          stroke="currentColor"
          strokeOpacity={0.5}
        />
        {boxEl(
          1,
          "LN₁",
          PART_COLOUR.ln,
          on("mu", "sig", "hat", "gb") && s.sub === 1,
          "ln1-box",
        )}
        {boxEl(2, "Attn", PART_COLOUR.attn, on("sub"), "attn-box")}
        {boxEl(
          5,
          "LN₂",
          PART_COLOUR.ln,
          on("mu", "sig", "hat", "gb") && s.sub === 2,
          "ln2-box",
        )}
        {boxEl(6, "FFN", PART_COLOUR.ffn, on("sub2"), "ffn-box")}
        {plus(3, "add1")}
        {plus(7, "add2")}
        <text x={4} y={sy + fs(11) + 6} fontSize={fs(11)} fill="currentColor">
          x
        </text>
        <text
          x={ax[4]! - 6}
          y={sy + fs(11) + 6}
          fontSize={fs(11)}
          fill="currentColor"
        >
          h
        </text>
        <text
          x={W - 4}
          y={sy + fs(11) + 6}
          textAnchor="end"
          fontSize={fs(11)}
          fill="currentColor"
        >
          y
        </text>
        <circle cx={mx} cy={my} r={6} fill={OKABE_ITO.blue} />

        {/* the distribution */}
        <g data-testid="ln-plot">
          <text
            x={plotX}
            y={plotY - fs(11) - 10}
            fontSize={fs(11)}
            fill="currentColor"
          >
            {phaseIdx <= 2
              ? `the ${D} numbers of ${xName}`
              : phaseIdx === 3
                ? `${xName} − μ`
                : phaseIdx === 4
                  ? `(${xName} − μ) / √(σ² + ε)`
                  : `${lnName}(${xName}) = γ ⊙ x̂ + β`}
          </text>
          <text x={plotX} y={plotY - 6} fontSize={fs(11)} fill="currentColor">
            mean {sig(Math.abs(ms.mean) < 1e-9 ? 0 : ms.mean)}, spread{" "}
            {sig(ms.std)}
          </text>
          {showStd && (
            <rect
              data-testid="ln-std"
              x={px(ms.mean - ms.std)}
              y={plotY}
              width={Math.max(1, px(ms.mean + ms.std) - px(ms.mean - ms.std))}
              height={lanes * laneH}
              fill={PART_COLOUR.ln}
              fillOpacity={on("sig") ? 0.3 : 0.15}
            />
          )}
          <line
            x1={plotX}
            x2={plotX + plotW}
            y1={plotY + lanes * laneH + 2}
            y2={plotY + lanes * laneH + 2}
            stroke="currentColor"
            strokeOpacity={0.5}
          />
          {[-R, 0, R].map((v) => (
            <g key={v}>
              <line
                x1={px(v)}
                x2={px(v)}
                y1={plotY + lanes * laneH + 2}
                y2={plotY + lanes * laneH + 7}
                stroke="currentColor"
                strokeOpacity={0.5}
              />
              <text
                x={px(v)}
                y={plotY + lanes * laneH + 8 + fs(11)}
                textAnchor="middle"
                fontSize={fs(11)}
                fill="currentColor"
              >
                {sig(v)}
              </text>
            </g>
          ))}
          {showMean && (
            <line
              data-testid="ln-mean"
              x1={px(ms.mean)}
              x2={px(ms.mean)}
              y1={plotY - 2}
              y2={plotY + lanes * laneH + 2}
              stroke="currentColor"
              strokeWidth={on("mu") ? 3 : 2}
              strokeDasharray="4 2"
            />
          )}
          {shown.map((v, i) => (
            <circle
              key={i}
              cx={px(v)}
              cy={plotY + (i % lanes) * laneH + laneH / 2}
              r={5}
              fill={dotColour}
              fillOpacity={0.85}
              stroke="currentColor"
              strokeOpacity={0.6}
            />
          ))}
        </g>

        {/* the strips */}
        <VectorStrip
          x={stripX}
          y={stripTop}
          values={s.x}
          cell={cell}
          height={cell}
          max={maxAbs(s.x)}
          label={`${xName}, the stream in  ‖${xName}‖ = ${sig(norm(s.x))}`}
          fontSize={fs(11)}
          testId="ln-x"
        />
        {s.ln && (
          <VectorStrip
            x={stripX}
            y={stripTop + stripGap}
            values={s.ln}
            cell={cell}
            height={cell}
            max={maxAbs(s.ln)}
            label={`${lnName}(${xName}), a copy for the sub-layer`}
            fontSize={fs(11)}
            activeColour={PART_COLOUR.ln}
            testId="ln-out"
          />
        )}
        {s.delta && (
          <VectorStrip
            x={stripX}
            y={stripTop + 2 * stripGap}
            values={s.delta}
            cell={cell}
            height={cell}
            max={maxAbs(s.delta)}
            label={`Δ = ${s.sub === 1 ? "Attn" : "FFN"}(${lnName}(${xName}))  ‖Δ‖ = ${sig(norm(s.delta))}`}
            fontSize={fs(11)}
            activeColour={subColour}
            testId="ln-delta"
          />
        )}
        {s.out && (
          <VectorStrip
            x={stripX}
            y={stripTop + 3 * stripGap}
            values={s.out}
            cell={cell}
            height={cell}
            max={maxAbs(s.out)}
            label={`${xName} + Δ, the stream out  ‖${s.sub === 1 ? "h" : "y"}‖ = ${sig(norm(s.out))}`}
            fontSize={fs(11)}
            testId="ln-sum"
          />
        )}
      </svg>
    </div>
  );

  const select =
    "focus-ring h-11 rounded border border-neutral-300 bg-white px-2 font-mono text-sm dark:border-neutral-700 dark:bg-neutral-950";
  const labelText = "font-medium text-neutral-700 dark:text-neutral-300";
  return (
    <AnimationPanel
      testId={testId}
      title="LayerNorm and the residual stream"
      summary="Each sub-layer reads a normalised copy of the stream (mean 0, spread 1, then γ and β) and adds its update back. Block 1, your input."
      stepper={st}
      stepLabel="step"
      caption={lnCaption(s, tr.tokens)}
      visual={visual}
      equation={children}
      hl={hl}
      onEquationHover={setHover}
      params={
        <>
          <label className="flex min-w-0 flex-col gap-1 text-sm">
            <span className={labelText}>
              Your input (up to {tr.tokens.length} characters)
            </span>
            <input
              type="text"
              value={text}
              maxLength={tr.tokens.length}
              onChange={(e) => setText(e.target.value.toLowerCase())}
              aria-label="LayerNorm animation input"
              className={select}
            />
          </label>
          <label className="flex min-w-0 flex-col gap-1 text-sm">
            <span className={labelText}>Position</span>
            <select
              value={pos}
              onChange={(e) => setPos(Number(e.target.value))}
              aria-label="LayerNorm position"
              className={select}
            >
              {tr.tokens.map((c, i) => (
                <option key={i} value={i}>
                  {i}: {showChar(c)}
                </option>
              ))}
            </select>
          </label>
          <label className="flex min-w-0 flex-col gap-1 text-sm">
            <span className={labelText}>γ (gain, every element)</span>
            <select
              value={gamma}
              onChange={(e) => setGamma(Number(e.target.value))}
              aria-label="LayerNorm gain"
              className={select}
            >
              {GAMMAS.map((v) => (
                <option key={v} value={v}>
                  {v === 1 ? "1 (the model's)" : v}
                </option>
              ))}
            </select>
          </label>
          <label className="flex min-w-0 flex-col gap-1 text-sm">
            <span className={labelText}>β (bias, every element)</span>
            <select
              value={beta}
              onChange={(e) => setBeta(Number(e.target.value))}
              aria-label="LayerNorm bias"
              className={select}
            >
              {BETAS.map((v) => (
                <option key={v} value={v}>
                  {v === 0 ? "0 (the model's)" : v}
                </option>
              ))}
            </select>
          </label>
        </>
      }
    />
  );
}
