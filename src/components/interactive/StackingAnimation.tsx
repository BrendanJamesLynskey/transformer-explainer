"use client";

/**
 * Chapter 06's animation: the residual stream at one position climbing
 * through the stack of blocks. Each row of the tower is the stream after
 * one more block, with the sizes of the two updates that block added and
 * a "logit lens" read-out (the final LayerNorm and output head applied
 * early); the last row is the model's real output. Every state comes from
 * `stackStates` over a traced forward pass, computed in the browser.
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
import { maxAbs, norm, pct, showChar, sig } from "@/lib/anim/format";
import {
  STACK_HL,
  stackCaption,
  stackStates,
  type LayerRow,
} from "@/lib/anim/stack-steps";
import type { TopToken } from "@/lib/anim/overview-steps";
import { DEFAULT_PARAMS, configFor } from "@/lib/compute/traces";
import { initModelWeights } from "@/lib/transformer/init";
import { ALPHABET, encode } from "@/lib/transformer/tokenizer";
import { PART_COLOUR } from "@/lib/viz/palette";

const BLOCKS = [1, 2, 3, 4] as const;

const tok = (id: number) => `'${showChar(ALPHABET[id] ?? "?")}'`;
const lensText = (top: TopToken[], n: number) =>
  top
    .slice(0, n)
    .map((t) => `${tok(t.id)} ${pct(t.prob)}`)
    .join("  ");

export default function StackingAnimation({
  testId = "stacking-animation",
  children,
}: {
  testId?: string;
  children?: ReactNode;
}): JSX.Element {
  const [text, setText] = useState(DEFAULT_PARAMS.text);
  const [pos, setPos] = useState(5);
  const [nBlocks, setNBlocks] = useState(4);
  const [hover, setHover] = useState<string | null>(null);
  const config = useMemo(
    () => configFor({ ...DEFAULT_PARAMS, nBlocks }),
    [nBlocks],
  );
  const states = useMemo(
    () => stackStates(text, config, initModelWeights(config), pos),
    [text, config, pos],
  );
  const tokens = useMemo(
    () => encode(text, config.seq_len).map((id) => ALPHABET[id] ?? "?"),
    [text, config],
  );
  const st = useStepper(states.length, {
    stepMs: 1100,
    smooth: true,
    resetKey: `${text}|${pos}|${nBlocks}`,
  });
  const box = useContainerWidth();
  const phone = box.width > 0 && box.width < PHONE_BELOW;
  const W = phone ? 360 : 720;
  const font = useSvgFont(W);
  const fs = font.fs;
  const s = states[st.step]!;
  const D = s.vec.length;
  const hl = STACK_HL[s.kind];
  const lit = (k: string) => hl === k || hover === k;
  const t = st.step === 0 ? 1 : Math.min(1, st.frac * 2 + (st.playing ? 0 : 1));

  // the rows on screen: the finished layers, plus the block in progress
  type Row = LayerRow & { pending: boolean };
  const rows: Row[] = s.rows.map((r) => ({ ...r, pending: false }));
  if (s.kind === "attn")
    rows.push({
      layer: s.block + 1,
      vec: s.vec,
      attnNorm: norm(s.delta!),
      ffnNorm: null,
      lens: [],
      pending: true,
    });
  const newest = rows.length - 1;
  const fresh = s.kind === "attn" || s.kind === "ffn" || st.step === 0;

  // ─── layout ──────────────────────────────────────────────
  const cell = phone ? 18 : 14;
  const stripW = D * cell;
  const labelW = phone ? 0 : 118;
  const normX = phone ? 0 : labelW + stripW + 14;
  const normW = phone ? 120 : 110;
  const lensX = phone ? normW + 30 : normX + normW + 16;
  const line = fs(11) + 6;
  const rowH = phone ? line + cell + 2 * line + 10 : Math.max(cell, line) + 14;
  const top0 = fs(11) * 2 + 16;
  const nRows = config.n_blocks + 2;
  const deltaY = top0 + nRows * rowH + fs(11) + 10;
  const H = deltaY + cell + 10;
  const normMax = useMemo(() => {
    let m = 1e-9;
    for (const x of states)
      for (const r of x.rows) m = Math.max(m, r.attnNorm ?? 0, r.ffnNorm ?? 0);
    return m;
  }, [states]);
  const rowY = (i: number) => top0 + i * rowH;
  const finalRow = s.kind === "final";

  const visual = (
    <div ref={box.ref} className="min-w-0">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="mx-auto w-full text-neutral-800 dark:text-neutral-200"
        role="img"
        aria-label="A table with one row per layer: the residual stream at this position as a row of coloured cells, bars for the sizes of the attention and feed-forward updates, and the logit lens's three most likely next characters"
      >
        <text x={0} y={fs(11)} fontSize={fs(11)} fill="currentColor">
          position {s.pos} (&apos;{showChar(tokens[s.pos] ?? "?")}&apos;)
          through {s.nBlocks} block{s.nBlocks > 1 ? "s" : ""}
        </text>
        {!phone && (
          <>
            <text x={labelW} y={top0 - 8} fontSize={fs(11)} fill="currentColor">
              residual stream x
            </text>
            <text x={normX} y={top0 - 8} fontSize={fs(11)} fill="currentColor">
              ‖Δ‖ attn / FFN
            </text>
            <text x={lensX} y={top0 - 8} fontSize={fs(11)} fill="currentColor">
              logit lens: top 3
            </text>
          </>
        )}
        {rows.map((r, i) => {
          const y = rowY(i);
          const isNew = i === newest && fresh;
          const op = isNew ? Math.max(0.15, t) : 1;
          const active = i === newest && !finalRow;
          const label = r.layer === 0 ? "x₀ = E + P" : `after block ${r.layer}`;
          const sy = phone ? y + line : y;
          const ny = phone ? sy + cell + 4 : y;
          return (
            <g
              key={r.layer}
              data-testid={`layer-${r.layer}`}
              data-pending={r.pending || undefined}
              opacity={op}
            >
              {active && (
                <rect
                  x={-2}
                  y={y - 4}
                  width={W + 4}
                  height={rowH - 4}
                  rx={6}
                  fill={r.layer === 0 ? PART_COLOUR.emb : PART_COLOUR.ffn}
                  fillOpacity={0.08}
                  stroke={
                    lit(r.layer === 0 ? "x0" : "blk")
                      ? r.layer === 0
                        ? PART_COLOUR.emb
                        : PART_COLOUR.ffn
                      : "none"
                  }
                  strokeWidth={2}
                />
              )}
              <text
                x={0}
                y={phone ? y + fs(11) : y + cell / 2 + fs(11) / 3}
                fontSize={fs(11)}
                fill="currentColor"
              >
                {phone
                  ? `${label}  ‖x‖ = ${sig(norm(r.vec))}`
                  : `${label}${r.pending ? " (attn)" : ""}`}
              </text>
              <VectorStrip
                x={labelW}
                y={sy}
                values={r.vec}
                cell={cell}
                height={cell}
                max={maxAbs(r.vec)}
                fontSize={fs(11)}
              />
              {/* the two updates' sizes */}
              {r.attnNorm !== null && (
                <rect
                  x={normX}
                  y={phone ? ny + 2 : y + 1}
                  width={Math.max(1, (normW / 2 - 4) * (r.attnNorm / normMax))}
                  height={phone ? fs(11) : cell - 2}
                  fill={PART_COLOUR.attn}
                  fillOpacity={0.8}
                />
              )}
              {r.ffnNorm !== null && (
                <rect
                  x={normX + normW / 2}
                  y={phone ? ny + 2 : y + 1}
                  width={Math.max(1, (normW / 2 - 4) * (r.ffnNorm / normMax))}
                  height={phone ? fs(11) : cell - 2}
                  fill={PART_COLOUR.ffn}
                  fillOpacity={0.8}
                />
              )}
              {r.layer === 0 && (
                <text
                  x={normX}
                  y={phone ? ny + fs(11) : y + cell / 2 + fs(11) / 3}
                  fontSize={fs(11)}
                  fill="currentColor"
                >
                  (no block yet)
                </text>
              )}
              <text
                x={lensX}
                y={phone ? ny + fs(11) : y + cell / 2 + fs(11) / 3}
                fontSize={fs(11)}
                fontFamily="monospace"
                fill="currentColor"
                data-testid={`lens-${r.layer}`}
              >
                {r.lens.length > 0 ? lensText(r.lens, phone ? 2 : 3) : "…"}
              </text>
            </g>
          );
        })}
        {finalRow && s.top && (
          <g data-testid="final-row">
            <rect
              x={-2}
              y={rowY(rows.length) - 4}
              width={W + 4}
              height={rowH - 4}
              rx={6}
              fill={PART_COLOUR.logits}
              fillOpacity={0.1}
              stroke={lit("lens") ? PART_COLOUR.logits : "none"}
              strokeWidth={2}
            />
            <text
              x={0}
              y={rowY(rows.length) + (phone ? fs(11) : cell / 2 + fs(11) / 3)}
              fontSize={fs(11)}
              fill="currentColor"
            >
              {phone
                ? "LN_f, then × Eᵀ: the model's output"
                : "LN_f · Eᵀ: output"}
            </text>
            <text
              x={phone ? 0 : lensX}
              y={
                rowY(rows.length) +
                (phone ? line + fs(11) : cell / 2 + fs(11) / 3)
              }
              fontSize={fs(11)}
              fontFamily="monospace"
              fill="currentColor"
            >
              {lensText(s.top, phone ? 3 : 3)}
            </text>
          </g>
        )}
        {s.delta && (
          <VectorStrip
            x={labelW}
            y={deltaY}
            values={s.delta}
            cell={cell}
            height={cell}
            max={maxAbs(s.delta)}
            label={`this step's ${s.kind === "attn" ? "attention" : "FFN"} update Δ  ‖Δ‖ = ${sig(norm(s.delta))} (own colour scale)`}
            fontSize={fs(11)}
            activeColour={
              s.kind === "attn" ? PART_COLOUR.attn : PART_COLOUR.ffn
            }
            testId="stack-delta"
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
      title="The residual stream through the stack"
      summary="One position's stream gains an attention update and an FFN update per block; after each block the logit lens reads it out as next-token guesses. Your input."
      stepper={st}
      stepLabel="step"
      caption={stackCaption(s, tokens)}
      visual={visual}
      equation={children}
      hl={hl}
      onEquationHover={setHover}
      params={
        <>
          <label className="flex min-w-0 flex-col gap-1 text-sm">
            <span className={labelText}>
              Your input (up to {config.seq_len} characters)
            </span>
            <input
              type="text"
              value={text}
              maxLength={config.seq_len}
              onChange={(e) => setText(e.target.value.toLowerCase())}
              aria-label="Stacking animation input"
              className={select}
            />
          </label>
          <label className="flex min-w-0 flex-col gap-1 text-sm">
            <span className={labelText}>Position</span>
            <select
              value={pos}
              onChange={(e) => setPos(Number(e.target.value))}
              aria-label="Stacking position"
              className={select}
            >
              {tokens.map((c, i) => (
                <option key={i} value={i}>
                  {i}: {showChar(c)}
                </option>
              ))}
            </select>
          </label>
          <label className="flex min-w-0 flex-col gap-1 text-sm">
            <span className={labelText}>Blocks</span>
            <select
              value={nBlocks}
              onChange={(e) => setNBlocks(Number(e.target.value))}
              aria-label="Number of blocks"
              className={select}
            >
              {BLOCKS.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
        </>
      }
    />
  );
}
