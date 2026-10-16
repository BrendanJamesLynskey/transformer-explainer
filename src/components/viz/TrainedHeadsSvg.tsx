/**
 * The landing page's hero: the trained model's four attention heads on one
 * input, with what each head does on the whole corpus, and the same input
 * at random init for contrast.
 *
 * Server Component — no `"use client"`. The weights come from
 * `lib/attention-preview` (the lessons' own forward pass, on Training's
 * trained weights) and the per-head numbers from
 * `src/data/decoder/trained-heads.json` (scripts/reference.py, checked
 * against the TypeScript model by tests/unit/attention-stats.test.ts).
 * Every cell has a `<title>` with its weight; masked cells are labelled.
 */
import { interpolateViridis } from "d3";

import type { HeadStats } from "@/lib/attention-stats";
import type { AttentionPreview, HeadPattern } from "@/lib/attention-preview";

const CELL = 13;
const LABEL = 12;
const GAP = 22;
const TITLE = 30;

/** Show a token readably: spaces become a visible "␣". */
function show(token: string): string {
  return token === " " ? "␣" : token;
}

const pct = (v: number) => `${Math.round(v * 100)}%`;

/** One line on what a head does, from its corpus statistics. */
export function habit(s: HeadStats): string {
  if (s.spaceStart > 0.6) return `spaces → first 2: ${pct(s.spaceStart)}`;
  if (s.entropy > 0.8) return `spread out (entropy ${s.entropy.toFixed(2)})`;
  return `this ${pct(s.self)} · previous ${pct(s.prev)}`;
}

function Grid({
  x,
  y,
  tokens,
  h,
  title,
  note,
  leftLabels,
  idPrefix,
}: {
  x: number;
  y: number;
  tokens: string[];
  h: HeadPattern;
  title: string;
  note: string;
  leftLabels: boolean;
  idPrefix: string;
}): JSX.Element {
  const gx = x + LABEL;
  const gy = y + TITLE + LABEL;
  return (
    <g data-head={`${h.block}-${h.head}`}>
      <text
        x={x}
        y={y + 10}
        fontSize={10}
        fontWeight={600}
        className="fill-neutral-800 dark:fill-neutral-200"
      >
        {title}
      </text>
      <text
        x={x}
        y={y + 22}
        fontSize={9}
        className="fill-neutral-600 dark:fill-neutral-400"
      >
        {note}
      </text>
      {tokens.map((t, i) => (
        <g key={`${idPrefix}-l${i}`} className="fill-neutral-500 font-mono">
          <text
            x={gx + i * CELL + CELL / 2}
            y={gy - 3}
            textAnchor="middle"
            fontSize={9}
          >
            {show(t)}
          </text>
          {leftLabels && (
            <text
              x={gx - 3}
              y={gy + i * CELL + CELL / 2 + 3}
              textAnchor="end"
              fontSize={9}
            >
              {show(t)}
            </text>
          )}
        </g>
      ))}
      {h.weights.map((row, i) =>
        row.map((w, j) => {
          const masked = j > i;
          return (
            <rect
              key={`${idPrefix}-${i}-${j}`}
              x={gx + j * CELL}
              y={gy + i * CELL}
              width={CELL - 1.5}
              height={CELL - 1.5}
              rx={1.5}
              className={
                masked ? "fill-neutral-200 dark:fill-neutral-800" : undefined
              }
              fill={masked ? undefined : interpolateViridis(w)}
            >
              <title>
                {masked
                  ? `"${show(tokens[i] ?? "")}" → "${show(tokens[j] ?? "")}": masked (a later token)`
                  : `"${show(tokens[i] ?? "")}" → "${show(tokens[j] ?? "")}": ${w.toFixed(3)}`}
              </title>
            </rect>
          );
        }),
      )}
    </g>
  );
}

export function TrainedHeadsSvg({
  trained,
  random,
  stats,
  randomStats,
}: {
  trained: AttentionPreview;
  random: AttentionPreview;
  stats: HeadStats[];
  randomStats: HeadStats[];
}): JSX.Element {
  const n = trained.tokens.length;
  const gw = LABEL + n * CELL;
  const gh = TITLE + LABEL + n * CELL;
  const W = 2 * gw + GAP + 8;
  const yRandom = 2 * (gh + 14) + 18;
  const H = yRandom + gh + 4;
  const stat = (b: number, h: number, from: HeadStats[]) =>
    from.find((s) => s.block === b && s.head === h)!;
  const r0 = random.heads[0]!;
  const rs = stat(0, 0, randomStats);
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-labelledby="trained-heads-title"
      className="h-auto w-full max-w-[340px]"
      data-trained-heads
    >
      <title id="trained-heads-title">
        {`Attention weights of the trained model's four heads on the text "${trained.tokens.join("")}", with what each head does across the corpus, and block 1 head 0 at random initialisation, whose rows are almost even.`}
      </title>
      {trained.heads.map((h, k) => {
        const col = k % 2;
        const row = Math.floor(k / 2);
        return (
          <Grid
            key={`t${k}`}
            idPrefix={`t${k}`}
            x={col * (gw + GAP)}
            y={row * (gh + 14)}
            tokens={trained.tokens}
            h={h}
            title={`block ${h.block + 1} · head ${h.head}`}
            note={habit(stat(h.block, h.head, stats))}
            leftLabels={col === 0}
          />
        );
      })}
      <line
        x1={0}
        x2={W}
        y1={yRandom - 10}
        y2={yRandom - 10}
        className="stroke-neutral-300 dark:stroke-neutral-700"
        strokeDasharray="3 3"
      />
      <Grid
        idPrefix="r"
        x={0}
        y={yRandom}
        tokens={random.tokens}
        h={r0}
        title="before training: block 1 · head 0"
        note={`random init: even (entropy ${rs.entropy.toFixed(2)})`}
        leftLabels
      />
      <g transform={`translate(${gw + GAP}, ${yRandom + TITLE + LABEL})`}>
        <text
          x={0}
          y={-6}
          fontSize={9}
          className="fill-neutral-600 dark:fill-neutral-400"
        >
          weight (one scale, 0 to 1)
        </text>
        {Array.from({ length: 10 }, (_, i) => (
          <rect
            key={`legend-${i}`}
            x={i * 10}
            y={0}
            width={10}
            height={10}
            fill={interpolateViridis(i / 9)}
          />
        ))}
        <text
          x={0}
          y={22}
          fontSize={9}
          className="fill-neutral-600 dark:fill-neutral-400"
        >
          0
        </text>
        <text
          x={100}
          y={22}
          fontSize={9}
          textAnchor="end"
          className="fill-neutral-600 dark:fill-neutral-400"
        >
          1
        </text>
        <text
          x={0}
          y={42}
          fontSize={9}
          className="fill-neutral-600 dark:fill-neutral-400"
        >
          grey: masked (later letters)
        </text>
      </g>
    </svg>
  );
}
