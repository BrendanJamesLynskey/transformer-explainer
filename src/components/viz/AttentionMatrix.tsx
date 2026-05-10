"use client";

/**
 * Square heatmap for an `[S × S]` attention matrix.
 *
 * Cells include `<title>` tooltips with the numeric value (SPEC §11
 * accessibility — heatmaps must include numeric tooltips). An optional
 * `activeRow` index renders a frame around that query row, used by the
 * widget's hover-to-highlight interaction.
 */
import { interpolateViridis } from "d3";
import { useMemo } from "react";

type Props = {
  matrix: number[][];
  /** Token labels for both axes (queries on Y, keys on X). */
  tokens?: readonly string[];
  /** Highlight one query row's cells, e.g. on hover. */
  activeRow?: number | null;
  /** Optional notification when the user hovers a query row. */
  onHoverRow?: (row: number | null) => void;
  /** Cell side in CSS pixels. Default 28. */
  cellSize?: number;
  /** Fraction the value is scaled by; pass `null` to colour by min/max. */
  fixedRange?: { min: number; max: number } | null;
};

// ─── d3 logic ──────────────────────────────────────────────────────────
function buildColourFn(
  matrix: number[][],
  fixedRange: Props["fixedRange"],
): (v: number) => string {
  let min = Infinity;
  let max = -Infinity;
  if (fixedRange) {
    min = fixedRange.min;
    max = fixedRange.max;
  } else {
    for (const row of matrix)
      for (const v of row) {
        if (!Number.isFinite(v)) continue;
        if (v < min) min = v;
        if (v > max) max = v;
      }
  }
  // Non-finite cells (the mask's −∞ entries) always render as the "blocked"
  // colour, regardless of which branch we take below. Without this, a mask
  // matrix that's `0` on/below diagonal and `−∞` above would collapse to a
  // single colour (since min == max == 0 sends finite cells through the
  // constant fallback), erasing the triangle the user expects to see.
  const blocked = "#0a0a0a";
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) {
    return (v: number) =>
      Number.isFinite(v) ? interpolateViridis(0.5) : blocked;
  }
  return (v: number) => {
    if (!Number.isFinite(v)) return blocked;
    return interpolateViridis((v - min) / (max - min));
  };
}
// ───────────────────────────────────────────────────────────────────────

export function AttentionMatrix({
  matrix,
  tokens,
  activeRow = null,
  onHoverRow,
  cellSize = 28,
  fixedRange = null,
}: Props): JSX.Element {
  const rows = matrix.length;
  const cols = rows === 0 ? 0 : (matrix[0]?.length ?? 0);
  const colour = useMemo(
    () => buildColourFn(matrix, fixedRange),
    [matrix, fixedRange],
  );
  const labelSize = 18;

  const width = labelSize + cols * cellSize;
  const height = labelSize + rows * cellSize;

  return (
    <svg
      width={width}
      height={height}
      role="img"
      aria-label={`${rows} × ${cols} attention matrix.`}
      className="block max-w-full overflow-visible"
    >
      {/* Column labels (keys). */}
      {tokens?.map((t, j) => (
        <text
          key={`col-${j}`}
          x={labelSize + j * cellSize + cellSize / 2}
          y={labelSize - 4}
          textAnchor="middle"
          className="fill-neutral-700 font-mono text-[10px] dark:fill-neutral-300"
        >
          {t === "\n" ? "↵" : t === " " ? "␣" : t}
        </text>
      ))}

      {matrix.map((row, i) => (
        <g
          key={`row-${i}`}
          transform={`translate(0, ${labelSize + i * cellSize})`}
          onMouseEnter={() => onHoverRow?.(i)}
          onMouseLeave={() => onHoverRow?.(null)}
        >
          {tokens?.[i] !== undefined && (
            <text
              x={labelSize - 4}
              y={cellSize / 2}
              dominantBaseline="middle"
              textAnchor="end"
              className="fill-neutral-700 font-mono text-[10px] dark:fill-neutral-300"
            >
              {tokens[i] === "\n" ? "↵" : tokens[i] === " " ? "␣" : tokens[i]}
            </text>
          )}
          {row.map((v, j) => (
            <rect
              key={`cell-${i}-${j}`}
              x={labelSize + j * cellSize}
              y={0}
              width={cellSize}
              height={cellSize}
              fill={colour(v)}
              stroke="rgba(0,0,0,0.06)"
            >
              <title>
                q{i} · k{j}: {Number.isFinite(v) ? v.toFixed(3) : "−∞"}
              </title>
            </rect>
          ))}
          {activeRow === i && (
            <rect
              x={labelSize}
              y={0}
              width={cols * cellSize}
              height={cellSize}
              fill="none"
              stroke="rgb(99 102 241)"
              strokeWidth={2}
              pointerEvents="none"
            />
          )}
        </g>
      ))}
    </svg>
  );
}
