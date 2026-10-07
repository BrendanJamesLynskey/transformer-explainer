"use client";

/**
 * Chapter 02's animation: the embedding lookup and the positional add.
 * The token id at the chosen position selects a row of the embedding
 * table E; the positional vector P[p] joins it; the sum is built element
 * by element; then the other rows of X = E + P fill in. Every state comes
 * from `embedStates` over the `/api/compute/embed` trace, computed in the
 * browser (`computeEmbed`).
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
import { maxAbs, showChar } from "@/lib/anim/format";
import {
  embedCaption,
  embedStates,
  type EmbedKind,
} from "@/lib/anim/embed-steps";
import { DEFAULT_PARAMS, computeEmbed, configFor } from "@/lib/compute/traces";
import { initModelWeights } from "@/lib/transformer/init";
import { PART_COLOUR, heat } from "@/lib/viz/palette";

const HL: Record<EmbedKind, string> = {
  ids: "tok",
  lookup: "emb",
  position: "pos",
  add: "xp",
  row: "xp",
};

export default function EmbeddingAnimation({
  testId = "embedding-animation",
  children,
}: {
  testId?: string;
  children?: ReactNode;
}): JSX.Element {
  const [text, setText] = useState(DEFAULT_PARAMS.text);
  const [pos, setPos] = useState(2);
  const [hover, setHover] = useState<string | null>(null);
  const p = useMemo(() => ({ ...DEFAULT_PARAMS, text }), [text]);
  const tr = useMemo(() => computeEmbed(p), [p]);
  // The table the lookup reads (the embedding table is the first draw of
  // the seeded initialiser, the same for every config with this seed).
  const table = useMemo(
    () =>
      initModelWeights(
        configFor(DEFAULT_PARAMS, { n_heads: 1, d_ff: 1, n_blocks: 0 }),
      ).tok_emb,
    [],
  );
  const states = useMemo(() => embedStates(tr, pos), [tr, pos]);
  const st = useStepper(states.length, {
    stepMs: 650,
    resetKey: `${text}|${pos}`,
  });
  const box = useContainerWidth();
  const phone = box.width > 0 && box.width < PHONE_BELOW;
  const W = phone ? 360 : 720;
  const font = useSvgFont(W);
  const fs = font.fs;
  const s = states[st.step]!;
  const S = tr.tokenIds.length;
  const D = s.tok.length;
  const V = table.length;

  // ─── layout ──────────────────────────────────────────────
  const slot = phone ? 40 : 48;
  const tokY = fs(11) + 6;
  const tabY = tokY + slot + fs(11) * 2 + 18;
  const cw = phone ? 4 : 6;
  const ch = phone ? 3 : 4;
  const tabX = phone ? 4 : 30;
  const sx = tabX + D * cw + (phone ? 14 : 70);
  const cell = Math.floor((W - sx - 4) / D);
  const gap = cell + fs(11) + 12;
  const xY = tabY + 3 * gap + fs(11) + 6;
  const rowH = phone ? 12 : 14;
  const H = Math.max(tabY + V * ch + 8, xY + S * rowH + 8);
  const tMax = useMemo(() => maxAbs(table.flat()), [table]);
  const xMax = maxAbs(tr.xAfterEmb.flat());
  // 1,024 cells that never change while the animation plays: drawn once
  // The 1,024 cells never change while the animation plays, so they are
  // painted once into a 16 × 64 pixel image (Canvas 2D) and drawn scaled
  // up: one element instead of 1,024 (this keeps the page's main thread
  // free; Lighthouse's total blocking time).
  const tableImage = useMemo(() => {
    if (typeof document === "undefined") return "";
    const c = document.createElement("canvas");
    c.width = table[0]!.length;
    c.height = table.length;
    const g = c.getContext("2d");
    if (!g) return "";
    table.forEach((row, r) =>
      row.forEach((v, k) => {
        const h = heat(v, tMax);
        g.globalAlpha = h.fillOpacity;
        g.fillStyle = h.fill;
        g.fillRect(k, r, 1, 1);
      }),
    );
    return c.toDataURL();
  }, [table, tMax]);
  const tableCells = (
    <image
      href={tableImage}
      x={tabX}
      y={tabY}
      width={D * cw}
      height={V * ch}
      preserveAspectRatio="none"
      style={{ imageRendering: "pixelated" }}
    />
  );
  const show = (k: string) => HL[s.kind] === k || hover === k;
  const id = s.id;
  const lit = s.kind !== "ids";

  const visual = (
    <div ref={box.ref} className="min-w-0">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="mx-auto w-full text-neutral-800 dark:text-neutral-200"
        role="img"
        aria-label="The input tokens with their ids, the embedding table with the selected row outlined, the token vector, the positional vector and their sum as rows of coloured cells, and the matrix X filling row by row"
      >
        {/* the tokens and their ids */}
        <text x={0} y={tokY - 4} fontSize={fs(11)} fill="currentColor">
          tokens → ids
        </text>
        {tr.tokens.map((c, i) => {
          const x = (W - S * slot) / 2 + i * slot;
          const on = i === s.pos;
          return (
            <g key={i}>
              <rect
                x={x + 2}
                y={tokY}
                width={slot - 4}
                height={slot - 4}
                rx={5}
                fill={PART_COLOUR.tok}
                fillOpacity={on ? 0.25 : 0.05}
                stroke={on ? PART_COLOUR.tok : "currentColor"}
                strokeOpacity={on ? 1 : 0.35}
                strokeWidth={on && show("tok") ? 3 : on ? 2 : 1}
              />
              <text
                x={x + slot / 2}
                y={tokY + slot / 2 + fs(13) / 3}
                textAnchor="middle"
                fontSize={fs(13)}
                fontFamily="monospace"
                fill="currentColor"
              >
                {showChar(c)}
              </text>
              <text
                x={x + slot / 2}
                y={tokY + slot + fs(11)}
                textAnchor="middle"
                fontSize={fs(11)}
                fill="currentColor"
              >
                {tr.tokenIds[i]}
              </text>
            </g>
          );
        })}

        {/* the embedding table */}
        <text x={tabX} y={tabY - 6} fontSize={fs(11)} fill="currentColor">
          E ({V}×{D})
        </text>
        {tableCells}
        {lit && (
          // dim the rest of the table (the panel's background, translucent)
          // and redraw the selected row on top
          <g>
            <rect
              x={tabX}
              y={tabY}
              width={D * cw}
              height={V * ch}
              className="fill-neutral-50 dark:fill-neutral-900"
              fillOpacity={0.65}
            />
            {table[id]!.map((v, k) => (
              <rect
                key={k}
                x={tabX + k * cw}
                y={tabY + id * ch}
                width={cw}
                height={ch}
                {...heat(v, tMax)}
              />
            ))}
          </g>
        )}
        {lit && (
          <g data-testid="lookup-row">
            <rect
              x={tabX - 2}
              y={tabY + id * ch - 2}
              width={D * cw + 4}
              height={ch + 4}
              fill="none"
              stroke={PART_COLOUR.emb}
              strokeWidth={2.5}
            />
            <line
              x1={tabX + D * cw + 3}
              y1={tabY + id * ch + ch / 2}
              x2={sx - 4}
              y2={tabY + cell / 2}
              stroke={PART_COLOUR.emb}
              strokeWidth={1.5}
              strokeOpacity={show("emb") ? 1 : 0.5}
            />
            {!phone && (
              <text
                x={tabX + D * cw + 6}
                y={tabY + id * ch + ch / 2 + fs(11) + 2}
                fontSize={fs(11)}
                fill="currentColor"
              >
                row {id}
              </text>
            )}
          </g>
        )}

        {/* E[t], P[p] and their sum */}
        {lit && (
          <VectorStrip
            x={sx}
            y={tabY}
            values={s.tok}
            cell={cell}
            height={cell}
            max={xMax}
            label={`E[${id}]: the row for '${showChar(s.char)}'`}
            fontSize={fs(11)}
            active={s.kind === "add" ? s.elem : -1}
            activeColour={PART_COLOUR.emb}
          />
        )}
        {(s.kind === "position" || s.kind === "add" || s.kind === "row") && (
          <VectorStrip
            x={sx}
            y={tabY + gap}
            values={s.pe}
            cell={cell}
            height={cell}
            max={xMax}
            label={`+ P[${s.pos}]: position ${s.pos}`}
            fontSize={fs(11)}
            active={s.kind === "add" ? s.elem : -1}
            activeColour={PART_COLOUR.pos}
          />
        )}
        {(s.kind === "add" || s.kind === "row") && (
          <VectorStrip
            x={sx}
            y={tabY + 2 * gap}
            values={s.sum}
            cell={cell}
            height={cell}
            max={xMax}
            label={`= x${s.pos}, the input to block 1`}
            fontSize={fs(11)}
            active={s.kind === "add" ? s.elem : -1}
            activeColour="currentColor"
            testId="sum-row"
          />
        )}

        {/* X = E + P, row by row */}
        <text x={sx} y={xY - 6} fontSize={fs(11)} fill="currentColor">
          X = E + P ({S}×{D})
        </text>
        {tr.xAfterEmb.map((row, i) => {
          const done = s.done.includes(i);
          return (
            <g key={i} data-testid={`x-row-${i}`} data-done={done || undefined}>
              <text
                x={sx - 4}
                y={xY + i * rowH + rowH - 2}
                textAnchor="end"
                fontSize={fs(11)}
                fontFamily="monospace"
                fill="currentColor"
              >
                {phone ? "" : showChar(tr.tokens[i]!)}
              </text>
              {row.map((v, k) => {
                const h = heat(v, xMax);
                return (
                  <rect
                    key={k}
                    x={sx + k * cell + 1}
                    y={xY + i * rowH + 1}
                    width={cell - 2}
                    height={rowH - 2}
                    fill={done ? h.fill : "none"}
                    fillOpacity={done ? h.fillOpacity : 0}
                    stroke="currentColor"
                    strokeOpacity={done ? 0.1 : 0.2}
                    strokeDasharray={done ? undefined : "2 2"}
                  />
                );
              })}
            </g>
          );
        })}
      </svg>
    </div>
  );

  return (
    <AnimationPanel
      testId={testId}
      title="From a character to a vector"
      summary="Lookup, then the positional add, element by element. The numbers are the model's, for your input."
      stepper={st}
      stepLabel="step"
      caption={embedCaption(s)}
      visual={visual}
      equation={children}
      hl={HL[s.kind]}
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
              aria-label="Embedding animation input"
              className="focus-ring h-11 rounded border border-neutral-300 bg-white px-2 font-mono text-sm dark:border-neutral-700 dark:bg-neutral-950"
            />
          </label>
          <label className="flex min-w-0 flex-col gap-1 text-sm">
            <span className="font-medium text-neutral-700 dark:text-neutral-300">
              Position to follow
            </span>
            <select
              value={pos}
              onChange={(e) => setPos(Number(e.target.value))}
              aria-label="Position to follow"
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
