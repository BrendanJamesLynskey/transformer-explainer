"use client";

/**
 * Chapter 03's centrepiece: one row of attention built step by step for a
 * chosen head and query — dot products one key at a time, the 1/√d_k
 * scale, the causal mask, the softmax (exponentials, their sum, the
 * weights), then the weighted sum of value rows — and on to the next row;
 * finally the heads are concatenated and projected through W_o. Every
 * state comes from `attentionStates` over the `/api/compute/attention`
 * trace, computed in the browser (`computeAttention`).
 */
import { useId, useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { VectorStrip } from "@/components/viz/VectorStrip";
import {
  PHONE_BELOW,
  useContainerWidth,
} from "@/components/viz/useContainerWidth";
import { useSvgFont } from "@/components/viz/useSvgFont";
import {
  ATTN_HL,
  attentionCaption,
  attentionStates,
  type AttnState,
} from "@/lib/anim/attention-steps";
import { maxAbs, showChar, sig } from "@/lib/anim/format";
import { DEFAULT_PARAMS, computeAttention } from "@/lib/compute/traces";
import { PART_COLOUR, heat } from "@/lib/viz/palette";

/** Which column of the table a phase has reached (0 = dots only). */
function stage(s: AttnState): number {
  switch (s.phase) {
    case "dot":
      return 0;
    case "scale":
    case "mask":
      return 1;
    case "exp":
    case "sum":
      return 2;
    default:
      return 3;
  }
}

export default function AttentionAnimation({
  testId = "attention-animation",
  children,
}: {
  testId?: string;
  children?: ReactNode;
}): JSX.Element {
  const [text, setText] = useState(DEFAULT_PARAMS.text);
  const [head, setHead] = useState<number | "all">(0);
  const [row, setRow] = useState(3);
  const [hover, setHover] = useState<string | null>(null);
  const hatch = `hatch-${useId().replace(/:/g, "")}`;
  const tr = useMemo(
    () => computeAttention({ ...DEFAULT_PARAMS, text }),
    [text],
  );
  const states = useMemo(
    () => attentionStates(tr, { head, row }),
    [tr, head, row],
  );
  const st = useStepper(states.length, {
    stepMs: 600,
    resetKey: `${text}|${head}|${row}`,
  });
  const box = useContainerWidth();
  const phone = box.width > 0 && box.width < PHONE_BELOW;
  const W = phone ? 360 : 720;
  const font = useSvgFont(W);
  const fs = font.fs;
  const s = states[st.step]!;
  const S = tr.tokens.length;
  const dk = s.dk;
  const tok = (j: number) => showChar(tr.tokens[j] ?? "?");
  const hl = ATTN_HL[s.phase];
  const on = (k: string) => hl === k || hover === k;

  // ─── layout ──────────────────────────────────────────────
  const cell = phone ? 28 : 30;
  const gx = fs(13) + 6;
  const gy = fs(11) * 2 + 14;
  const gridW = S * cell;
  const lineH = fs(11) + 7;
  const tx = phone ? 0 : gx + gridW + 30;
  const ty = phone ? gy + S * cell + fs(11) + 22 : gy - lineH + 4;
  const colW = (W - tx) / 5;
  const tableH = (S + 2) * lineH;
  const sy = Math.max(gy + S * cell, ty + tableH) + fs(11) + 22;
  const sc = phone ? 26 : 26;
  const stripGap = sc + fs(11) + 14;
  const H = phone ? sy + 3 * stripGap + 6 : sy + sc + 10;
  const tail = s.phase === "concat" || s.phase === "project";

  // the head's finished rows (weights) and this row (by phase)
  const wHead = s.head >= 0 ? tr.weights[s.head]! : [];
  const rowMax = maxAbs(stage(s) === 0 ? s.dots : (s.scores ?? []));
  const cellFill = (i: number, j: number) => {
    if (i === s.row) {
      if (s.weights) {
        const v = s.weights[j];
        return v === null || v === undefined
          ? null
          : { fill: PART_COLOUR.attn, fillOpacity: 0.08 + 0.9 * v };
      }
      const v = stage(s) === 0 ? s.dots[j] : s.scores?.[j];
      return v === null || v === undefined ? null : heat(v, rowMax);
    }
    if (s.rowsDone.includes(i)) {
      const v = wHead[i]![j]!;
      return j > i
        ? null
        : { fill: PART_COLOUR.attn, fillOpacity: 0.06 + 0.5 * v };
    }
    return null;
  };

  const qh =
    s.head >= 0 ? tr.Q[s.row]!.slice(s.head * dk, (s.head + 1) * dk) : [];
  const kOrV =
    s.phase === "dot"
      ? tr.K[s.j]!.slice(s.head * dk, (s.head + 1) * dk)
      : s.phase === "wsum"
        ? tr.V[s.j]!.slice(s.head * dk, (s.head + 1) * dk)
        : null;

  const cols = ["key", "q·k", "÷√d", "e^(s−m)", "a"];
  const val = (v: number | null | undefined, masked: boolean) =>
    masked ? "−∞" : v === null || v === undefined ? "" : sig(v);

  const grid = (
    <g data-testid="score-grid">
      <text x={gx} y={fs(11)} fontSize={fs(11)} fill="currentColor">
        head {s.head}: query rows × keys
      </text>
      {Array.from({ length: S }, (_, j) => (
        <text
          key={`c${j}`}
          x={gx + j * cell + cell / 2}
          y={gy - 5}
          textAnchor="middle"
          fontSize={fs(11)}
          fontFamily="monospace"
          fill="currentColor"
          fontWeight={s.phase === "dot" && j === s.j ? 700 : 400}
        >
          {tok(j)}
        </text>
      ))}
      {Array.from({ length: S }, (_, i) => (
        <g key={`r${i}`}>
          <text
            x={gx - 4}
            y={gy + i * cell + cell / 2 + fs(11) / 3}
            textAnchor="end"
            fontSize={fs(11)}
            fontFamily="monospace"
            fill="currentColor"
            fontWeight={i === s.row ? 700 : 400}
          >
            {tok(i)}
          </text>
          {Array.from({ length: S }, (_, j) => {
            const f = cellFill(i, j);
            const masked =
              j > i &&
              (s.rowsDone.includes(i) ||
                (i === s.row && s.phase !== "dot" && s.phase !== "scale"));
            const active =
              i === s.row &&
              j === s.j &&
              (s.phase === "dot" || s.phase === "wsum");
            return (
              <rect
                key={j}
                data-testid={i === s.row ? `cell-${j}` : undefined}
                data-masked={masked || undefined}
                x={gx + j * cell + 1}
                y={gy + i * cell + 1}
                width={cell - 2}
                height={cell - 2}
                rx={2}
                fill={masked ? `url(#${hatch})` : f ? f.fill : "none"}
                fillOpacity={masked ? 1 : f ? f.fillOpacity : 0}
                stroke={
                  active
                    ? "currentColor"
                    : i === s.row
                      ? PART_COLOUR.attn
                      : "currentColor"
                }
                strokeOpacity={active ? 1 : i === s.row ? 0.8 : 0.15}
                strokeWidth={active ? 2.5 : 1}
                strokeDasharray={!f && !masked ? "2 2" : undefined}
              />
            );
          })}
        </g>
      ))}
    </g>
  );

  const table = (
    <g data-testid="row-table" fontSize={fs(11)}>
      {cols.map((c, k) => (
        <text
          key={c}
          x={tx + k * colW + colW - 6}
          y={ty}
          textAnchor="end"
          fill="currentColor"
          fontWeight={600}
          opacity={k === 0 || k - 1 <= stage(s) ? 1 : 0.35}
        >
          {c}
        </text>
      ))}
      {Array.from({ length: S }, (_, j) => {
        const y = ty + (j + 1) * lineH;
        const masked = j > s.row && stage(s) >= 1 && s.phase !== "scale";
        const activeRow =
          (s.phase === "dot" || s.phase === "wsum") && j === s.j;
        const cells = [
          `${j} ${tok(j)}`,
          val(s.dots[j], false),
          stage(s) >= 1 ? val(s.scores?.[j], masked) : "",
          stage(s) >= 2 ? val(s.exps?.[j], masked) : "",
          stage(s) >= 3 ? val(s.weights?.[j], masked) : "",
        ];
        return (
          <g key={j} opacity={masked ? 0.55 : 1}>
            {activeRow && (
              <rect
                x={tx + 2}
                y={y - lineH + 5}
                width={W - tx - 4}
                height={lineH}
                rx={3}
                fill={PART_COLOUR.attn}
                fillOpacity={0.15}
              />
            )}
            {cells.map((c, k) => (
              <text
                key={k}
                x={tx + k * colW + colW - 6}
                y={y}
                textAnchor="end"
                fontFamily={k === 0 ? "monospace" : undefined}
                fill="currentColor"
              >
                {c}
              </text>
            ))}
          </g>
        );
      })}
      <text
        x={W - 6}
        y={ty + (S + 1) * lineH + 2}
        textAnchor="end"
        fill="currentColor"
        fontWeight={on("sum") ? 700 : 400}
      >
        {s.sum !== null ? `Σ e = ${sig(s.sum)}` : ""}
      </text>
    </g>
  );

  const strips = !tail && (
    <g>
      <VectorStrip
        x={phone ? (W - dk * sc) / 2 : 0}
        y={sy}
        values={qh}
        cell={sc}
        height={sc}
        max={maxAbs(qh)}
        label={`q${s.row} (query '${tok(s.row)}')`}
        fontSize={fs(11)}
      />
      {kOrV && (
        <VectorStrip
          x={phone ? (W - dk * sc) / 2 : dk * sc + 24}
          y={phone ? sy + stripGap : sy}
          values={kOrV}
          cell={sc}
          height={sc}
          max={maxAbs(kOrV)}
          label={`${s.phase === "dot" ? "k" : "v"}${s.j} ('${tok(s.j)}')`}
          fontSize={fs(11)}
        />
      )}
      {s.out && (
        <VectorStrip
          x={phone ? (W - dk * sc) / 2 : 2 * (dk * sc + 24)}
          y={phone ? sy + 2 * stripGap : sy}
          values={s.out}
          cell={sc}
          height={sc}
          max={maxAbs(s.out)}
          label={`o${s.row} = Σ a·v so far`}
          fontSize={fs(11)}
          testId="out-strip"
        />
      )}
    </g>
  );

  const mats = tail && (
    <g data-testid="heads-out">
      {[
        { m: s.concat!, label: "[o⁽⁰⁾ ‖ o⁽¹⁾]: heads side by side", y: 0 },
        ...(s.projected
          ? [{ m: s.projected, label: "× W_o: the attention output", y: 1 }]
          : []),
      ].map(({ m, label, y }) => {
        const mc = Math.floor((W - gx - 10) / m[0]!.length);
        const rh = 16;
        const y0 = gy + y * (S * rh + fs(11) + 26);
        const mx = maxAbs(m.flat());
        return (
          <g key={label}>
            <text x={gx} y={y0 - 6} fontSize={fs(11)} fill="currentColor">
              {label}
            </text>
            {m.map((r, i) =>
              r.map((v, k) => {
                const h = heat(v, mx);
                return (
                  <rect
                    key={`${i}-${k}`}
                    x={gx + k * mc + 1}
                    y={y0 + i * rh + 1}
                    width={mc - 2}
                    height={rh - 2}
                    fill={h.fill}
                    fillOpacity={h.fillOpacity}
                  />
                );
              }),
            )}
            {y === 0 && (
              <line
                x1={gx + dk * mc}
                x2={gx + dk * mc}
                y1={y0 - 2}
                y2={y0 + S * rh + 2}
                stroke="currentColor"
                strokeWidth={2}
              />
            )}
          </g>
        );
      })}
    </g>
  );

  const tailH = gy + 2 * (S * 16 + fs(11) + 26);
  const visual = (
    <div ref={box.ref} className="min-w-0">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${Math.max(H, tailH)}`}
        className="mx-auto w-full text-neutral-800 dark:text-neutral-200"
        role="img"
        aria-label="The attention pattern as a grid of query rows by keys, a table of this row's dot products, scaled scores, exponentials and weights, and the query, key or value and output vectors as rows of coloured cells"
      >
        <defs>
          <pattern
            id={hatch}
            width={6}
            height={6}
            patternUnits="userSpaceOnUse"
            patternTransform="rotate(45)"
          >
            <rect width={6} height={6} fill="currentColor" fillOpacity={0.08} />
            <line
              x1={0}
              y1={0}
              x2={0}
              y2={6}
              stroke="currentColor"
              strokeOpacity={0.45}
              strokeWidth={2}
            />
          </pattern>
        </defs>
        {tail ? (
          mats
        ) : (
          <>
            {grid}
            {table}
            {strips}
          </>
        )}
      </svg>
    </div>
  );

  return (
    <AnimationPanel
      testId={testId}
      title="Attention, one query row at a time"
      summary="Dot products, scale, mask, softmax, weighted sum; then the next row. The numbers are block 1's, for your input."
      stepper={st}
      stepLabel="step"
      caption={attentionCaption(s, tr.tokens)}
      visual={visual}
      equation={children}
      hl={hl}
      onEquationHover={setHover}
      params={
        <>
          <label className="flex min-w-0 flex-col gap-1 text-sm">
            <span className="font-medium text-neutral-700 dark:text-neutral-300">
              Your input (up to {S} characters)
            </span>
            <input
              type="text"
              value={text}
              maxLength={S}
              onChange={(e) => setText(e.target.value.toLowerCase())}
              aria-label="Attention animation input"
              className="focus-ring h-11 rounded border border-neutral-300 bg-white px-2 font-mono text-sm dark:border-neutral-700 dark:bg-neutral-950"
            />
          </label>
          <div className="grid min-w-0 grid-cols-2 gap-2">
            <label className="flex min-w-0 flex-col gap-1 text-sm">
              <span className="font-medium text-neutral-700 dark:text-neutral-300">
                Head
              </span>
              <select
                value={String(head)}
                onChange={(e) =>
                  setHead(
                    e.target.value === "all" ? "all" : Number(e.target.value),
                  )
                }
                aria-label="Attention head"
                className="focus-ring h-11 rounded border border-neutral-300 bg-white px-2 text-sm dark:border-neutral-700 dark:bg-neutral-950"
              >
                {Array.from({ length: tr.nHeads }, (_, h) => (
                  <option key={h} value={h}>
                    head {h}
                  </option>
                ))}
                <option value="all">play all heads</option>
              </select>
            </label>
            <label className="flex min-w-0 flex-col gap-1 text-sm">
              <span className="font-medium text-neutral-700 dark:text-neutral-300">
                Start at query
              </span>
              <select
                value={row}
                onChange={(e) => setRow(Number(e.target.value))}
                aria-label="First query row"
                className="focus-ring h-11 rounded border border-neutral-300 bg-white px-2 font-mono text-sm dark:border-neutral-700 dark:bg-neutral-950"
              >
                {tr.tokens.map((c, i) => (
                  <option key={i} value={i}>
                    {i}: {showChar(c)}
                  </option>
                ))}
              </select>
            </label>
          </div>
        </>
      }
    />
  );
}
