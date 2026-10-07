"use client";

/**
 * Chapter 01's hero: one token's journey through the toy decoder. Every
 * frame is a state from `overviewStates` (a real forward pass on the
 * reader's input, computed in the browser); between two states the marker
 * glides from one stage of the pipeline to the next.
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
  OVERVIEW_HL,
  overviewCaption,
  overviewStates,
  type OverviewKind,
  type OverviewState,
} from "@/lib/anim/overview-steps";
import { DEFAULT_PARAMS, configFor } from "@/lib/compute/traces";
import { initModelWeights } from "@/lib/transformer/init";
import { ALPHABET } from "@/lib/transformer/tokenizer";
import { PART_COLOUR, type Part } from "@/lib/viz/palette";

type Stage = { kind: OverviewKind; block: number; label: string; part: Part };

/** The pipeline, left to right (one stage per kind of step). */
function stages(nBlocks: number): Stage[] {
  const out: Stage[] = [
    { kind: "token", block: -1, label: "id", part: "tok" },
    { kind: "embed", block: -1, label: "E[t]", part: "emb" },
    { kind: "position", block: -1, label: "+ P[p]", part: "pos" },
  ];
  for (let b = 0; b < nBlocks; b++) {
    out.push({ kind: "attn", block: b, label: `Attn ${b + 1}`, part: "attn" });
    out.push({ kind: "ffn", block: b, label: `FFN ${b + 1}`, part: "ffn" });
  }
  out.push({ kind: "norm", block: -1, label: "LN", part: "ln" });
  out.push({ kind: "logits", block: -1, label: "logits", part: "logits" });
  out.push({ kind: "sample", block: -1, label: "sample", part: "sample" });
  return out;
}

function stageIndex(st: Stage[], s: OverviewState): number {
  const kind = s.kind === "append" ? "sample" : s.kind;
  return st.findIndex((g) => g.kind === kind && g.block === s.block);
}

const ROUNDS = 3;

export default function OverviewHero({
  testId = "overview-hero",
  children,
}: {
  testId?: string;
  children?: ReactNode;
}): JSX.Element {
  const [text, setText] = useState("hello");
  const [seed, setSeed] = useState(DEFAULT_PARAMS.seed);
  const [hover, setHover] = useState<string | null>(null);
  const config = useMemo(() => configFor({ ...DEFAULT_PARAMS, seed }), [seed]);
  const states = useMemo(
    () =>
      overviewStates(text, config, initModelWeights(config), {
        rounds: ROUNDS,
      }),
    [text, config],
  );
  const st = useStepper(states.length, {
    stepMs: 1100,
    smooth: true,
    resetKey: `${text}|${seed}`,
  });
  const box = useContainerWidth();
  const phone = box.width > 0 && box.width < PHONE_BELOW;
  const W = phone ? 360 : 720;
  const font = useSvgFont(W);
  const fs = font.fs;
  const cur = states[st.step]!;
  const prev = states[Math.max(0, st.step - 1)]!;
  const pipe = stages(config.n_blocks);

  // ─── layout ──────────────────────────────────────────────
  const S = config.seq_len;
  const slot = phone ? 40 : 44;
  const ctxX = (W - S * slot) / 2;
  const ctxY = fs(14) + 8;
  const perRow = phone ? Math.ceil(pipe.length / 2) : pipe.length;
  const sw = (W - 8) / perRow;
  const pipeY = ctxY + slot + fs(14) + 22;
  const sh = Math.max(34, fs(11) + 18);
  const stageXY = (k: number) => ({
    x: 4 + (k % perRow) * sw,
    y: pipeY + Math.floor(k / perRow) * (sh + 18),
  });
  const pipeRows = Math.ceil(pipe.length / perRow);
  const stripY = pipeY + pipeRows * (sh + 18) + fs(11) + 14;
  const cell = phone ? 20 : 22;
  const D = config.d_model;
  const topX = phone ? 0 : D * cell + 40;
  const topY = phone
    ? stripY + 2 * (cell + fs(11) + 14) + fs(11)
    : stripY - fs(11);
  const barW = phone ? W - 80 : W - topX - 70;
  const H = topY + 6 * (fs(11) + 8) + 8;

  // the marker glides from the previous step's stage to this one's
  const a = stageXY(stageIndex(pipe, prev));
  const b = stageXY(stageIndex(pipe, cur));
  const t = st.step === 0 ? 1 : Math.min(1, st.frac * 3 + (st.playing ? 0 : 1));
  const mx = a.x + (b.x - a.x) * t + sw / 2;
  const my = a.y + (b.y - a.y) * t + sh + 8;

  const vec = cur.vec;
  const delta = cur.delta;
  const top = cur.top;
  const pMax = top ? Math.max(...top.map((x) => x.prob)) : 1;
  const hlKey = OVERVIEW_HL[cur.kind];

  const visual = (
    <div ref={box.ref} className="min-w-0">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="mx-auto w-full text-neutral-800 dark:text-neutral-200"
        role="img"
        aria-label="The context tokens, the decoder pipeline from token id to sample, the residual stream as a row of coloured cells, and the five most likely next tokens"
      >
        {/* context */}
        <text x={ctxX} y={ctxY - 6} fontSize={fs(11)} fill="currentColor">
          context (round {cur.round + 1})
        </text>
        {Array.from({ length: S }, (_, i) => {
          const id = cur.context[i];
          const isNew = cur.kind === "append" && i === cur.context.length - 1;
          const on = i === cur.pos && cur.kind !== "append";
          return (
            <g key={i} data-testid={`ctx-${i}`}>
              <rect
                x={ctxX + i * slot + 2}
                y={ctxY}
                width={slot - 4}
                height={slot - 4}
                rx={5}
                fill={isNew ? PART_COLOUR.sample : PART_COLOUR.tok}
                fillOpacity={id === undefined ? 0 : on || isNew ? 0.25 : 0.06}
                stroke={on ? PART_COLOUR.tok : "currentColor"}
                strokeOpacity={on ? 1 : id === undefined ? 0.25 : 0.4}
                strokeWidth={on ? 2.5 : 1}
                strokeDasharray={id === undefined ? "3 3" : undefined}
              />
              {id !== undefined && (
                <text
                  x={ctxX + i * slot + slot / 2}
                  y={ctxY + slot / 2 + fs(13) / 3}
                  textAnchor="middle"
                  fontSize={fs(13)}
                  fontFamily="monospace"
                  fill="currentColor"
                  opacity={isNew ? Math.max(0.2, st.playing ? st.frac : 1) : 1}
                >
                  {showChar(ALPHABET[id] ?? "?")}
                </text>
              )}
            </g>
          );
        })}

        {/* pipeline */}
        {pipe.map((g, k) => {
          const { x, y } = stageXY(k);
          const i = stageIndex(pipe, cur);
          const active = k === i;
          const lit =
            active || hover === g.part || hover === OVERVIEW_HL[g.kind];
          return (
            <g
              key={k}
              data-testid={`stage-${k}`}
              data-active={active || undefined}
            >
              <rect
                x={x + 2}
                y={y}
                width={sw - 4}
                height={sh}
                rx={6}
                fill={PART_COLOUR[g.part]}
                fillOpacity={active ? 0.3 : k < i ? 0.1 : 0.04}
                stroke={lit ? PART_COLOUR[g.part] : "currentColor"}
                strokeOpacity={lit ? 1 : 0.35}
                strokeWidth={lit ? 2.5 : 1}
              />
              <text
                x={x + sw / 2}
                y={y + sh / 2 + fs(11) / 3}
                textAnchor="middle"
                fontSize={fs(11)}
                fill="currentColor"
              >
                {g.label}
              </text>
            </g>
          );
        })}
        <circle cx={mx} cy={my} r={6} fill={PART_COLOUR.tok} />

        {/* the residual stream at the followed position */}
        {vec && (
          <VectorStrip
            x={phone ? (W - D * cell) / 2 : 0}
            y={stripY}
            values={vec}
            cell={cell}
            height={cell}
            max={maxAbs(vec)}
            label={`residual stream x at position ${cur.pos}  ‖x‖ = ${sig(norm(vec))}`}
            fontSize={fs(11)}
            testId="stream"
          />
        )}
        {delta && (
          <VectorStrip
            x={phone ? (W - D * cell) / 2 : 0}
            y={stripY + cell + fs(11) + 14}
            values={delta}
            cell={cell}
            height={cell}
            max={maxAbs(delta)}
            label={`added this step  ‖Δ‖ = ${sig(norm(delta))}`}
            fontSize={fs(11)}
            activeColour={PART_COLOUR.attn}
          />
        )}

        {/* the five most likely next tokens */}
        {top && (
          <g data-testid="top5">
            <text x={topX} y={topY - 6} fontSize={fs(11)} fill="currentColor">
              next-token probabilities (top 5 of 64)
            </text>
            {top.map((tt, r) => {
              const y = topY + r * (fs(11) + 8);
              const picked = cur.sampled === tt.id;
              return (
                <g key={tt.id}>
                  <text
                    x={topX + 14}
                    y={y + fs(11)}
                    textAnchor="middle"
                    fontFamily="monospace"
                    fontSize={fs(12)}
                    fill="currentColor"
                  >
                    {showChar(ALPHABET[tt.id] ?? "?")}
                  </text>
                  <rect
                    x={topX + 30}
                    y={y + 2}
                    width={Math.max(2, (barW * tt.prob) / pMax)}
                    height={fs(11)}
                    rx={2}
                    fill={PART_COLOUR.logits}
                    fillOpacity={picked ? 0.95 : 0.4}
                    stroke={picked ? "currentColor" : "none"}
                  />
                  <text
                    x={topX + 34 + Math.max(2, (barW * tt.prob) / pMax)}
                    y={y + fs(11)}
                    fontSize={fs(11)}
                    fill="currentColor"
                  >
                    {pct(tt.prob)}
                  </text>
                </g>
              );
            })}
            {cur.sampled !== null && !top.some((x) => x.id === cur.sampled) && (
              <text
                x={topX}
                y={topY + 5 * (fs(11) + 8) + fs(11)}
                fontSize={fs(11)}
                fill="currentColor"
              >
                sampled &apos;{showChar(ALPHABET[cur.sampled] ?? "?")}&apos; (
                {pct(cur.prob!)}), outside the top 5
              </text>
            )}
          </g>
        )}
      </svg>
    </div>
  );

  return (
    <AnimationPanel
      testId={testId}
      title="One token's journey through the decoder"
      summary="The newest character becomes a vector, collects an update from every block, and comes out as next-token probabilities; one is sampled and appended. Every number is a real forward pass on your input."
      stepper={st}
      stepLabel="step"
      caption={overviewCaption(cur)}
      visual={visual}
      equation={children}
      hl={hlKey}
      onEquationHover={setHover}
      params={
        <>
          <label className="flex min-w-0 flex-col gap-1 text-sm">
            <span className="font-medium text-neutral-700 dark:text-neutral-300">
              Your prompt (up to {S - 1} characters)
            </span>
            <input
              type="text"
              value={text}
              maxLength={S - 1}
              onChange={(e) => setText(e.target.value.toLowerCase())}
              aria-label="Hero prompt"
              className="focus-ring h-11 rounded border border-neutral-300 bg-white px-2 font-mono text-sm dark:border-neutral-700 dark:bg-neutral-950"
            />
          </label>
          <label className="flex min-w-0 flex-col gap-1 text-sm">
            <span className="font-medium text-neutral-700 dark:text-neutral-300">
              Seed (weights and sampling)
            </span>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Math.trunc(Number(e.target.value || 0)))}
              aria-label="Hero seed"
              className="focus-ring h-11 rounded border border-neutral-300 bg-white px-2 font-mono text-sm dark:border-neutral-700 dark:bg-neutral-950"
            />
          </label>
        </>
      }
    />
  );
}
