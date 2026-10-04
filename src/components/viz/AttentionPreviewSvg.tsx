/**
 * Static attention heatmap for the landing page.
 *
 * Server Component — no `"use client"`. The weights are computed on the
 * server by `lib/attention-preview` and drawn here as plain SVG, so the
 * landing page ships no D3 or React-hydration cost for this picture. The
 * interactive version is `AttentionMatrix` inside the attention lesson.
 *
 * Accessibility (SPEC §11): every cell has a `<title>` with its numeric
 * weight, the masked cells carry a "masked" label rather than relying on
 * colour, and the whole figure has a text summary.
 */
import { interpolateViridis } from "d3";

import type { AttentionPreview } from "@/lib/attention-preview";

const CELL = 22;
const LABEL = 18;

/** Show a token readably: spaces become a visible "␣". */
function show(token: string): string {
  return token === " " ? "␣" : token;
}

export function AttentionPreviewSvg({
  preview,
}: {
  preview: AttentionPreview;
}): JSX.Element {
  const { tokens, weights } = preview;
  const n = tokens.length;
  // One colour scale for the whole grid. (Scaling each row separately was
  // tried: with untrained weights every row is within ~1% of uniform, so
  // it turned the whole triangle one colour.)
  let max = 0;
  weights.forEach((row, i) =>
    row.forEach((w, j) => {
      if (j <= i && w > max) max = w;
    }),
  );
  const size = LABEL + n * CELL;

  return (
    <svg
      viewBox={`0 0 ${size} ${size}`}
      role="img"
      aria-labelledby="attn-preview-title"
      className="h-auto w-full max-w-[280px]"
    >
      <title id="attn-preview-title">
        {`Attention weights for head 0 on the text "${tokens.join("")}". Each row is a query position; it spreads its attention over itself and earlier positions only, and the cells above the diagonal are masked.`}
      </title>
      {tokens.map((t, i) => (
        <g key={`labels-${i}`} className="fill-neutral-500 font-mono">
          <text
            x={LABEL + i * CELL + CELL / 2}
            y={LABEL - 5}
            textAnchor="middle"
            fontSize={11}
          >
            {show(t)}
          </text>
          <text
            x={LABEL - 5}
            y={LABEL + i * CELL + CELL / 2 + 4}
            textAnchor="end"
            fontSize={11}
          >
            {show(t)}
          </text>
        </g>
      ))}
      {weights.map((row, i) =>
        row.map((w, j) => {
          const masked = j > i;
          return (
            <rect
              key={`${i}-${j}`}
              x={LABEL + j * CELL}
              y={LABEL + i * CELL}
              width={CELL - 2}
              height={CELL - 2}
              rx={2}
              className={
                masked ? "fill-neutral-200 dark:fill-neutral-800" : undefined
              }
              fill={masked ? undefined : interpolateViridis(w / (max || 1))}
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
    </svg>
  );
}
