"use client";

/**
 * Compact heatmap of a [rows × cols] matrix using D3 colour scales.
 *
 * React owns DOM structure: we render `<rect>`s in JSX, keyed by index.
 * D3 owns the maths: it computes the colour scale and writes `fill` via a
 * useEffect. Per CLAUDE.md §5, "never have D3 create or remove DOM nodes
 * that React owns."
 */
import { interpolateViridis } from "d3";
import { useMemo } from "react";

type Props = {
  matrix: number[][];
  rowLabels?: readonly string[];
  /** Cell side in CSS pixels. Default 16. */
  cellSize?: number;
  /** Label height in CSS pixels. Default 18. */
  labelWidth?: number;
};

// ─── d3 logic ──────────────────────────────────────────────────────────
function buildColourFn(matrix: number[][]): (v: number) => string {
  let min = Infinity;
  let max = -Infinity;
  for (const row of matrix)
    for (const v of row) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) {
    return () => interpolateViridis(0.5);
  }
  return (v: number) => interpolateViridis((v - min) / (max - min));
}
// ───────────────────────────────────────────────────────────────────────

export function EmbeddingHeatmap({
  matrix,
  rowLabels,
  cellSize = 16,
  labelWidth = 36,
}: Props): JSX.Element {
  const rows = matrix.length;
  const cols = rows === 0 ? 0 : (matrix[0]?.length ?? 0);
  const colour = useMemo(() => buildColourFn(matrix), [matrix]);

  const width = labelWidth + cols * cellSize;
  const height = rows * cellSize;

  return (
    <svg
      width={width}
      height={height}
      role="img"
      aria-label={`Heatmap of a ${rows} × ${cols} matrix.`}
      className="block max-w-full overflow-visible"
    >
      {matrix.map((row, i) => (
        <g key={`row-${i}`} transform={`translate(0, ${i * cellSize})`}>
          {rowLabels?.[i] !== undefined && (
            <text
              x={labelWidth - 6}
              y={cellSize / 2}
              dominantBaseline="middle"
              textAnchor="end"
              className="fill-neutral-700 font-mono text-[10px] dark:fill-neutral-300"
            >
              {rowLabels[i]}
            </text>
          )}
          {row.map((v, j) => (
            <rect
              key={`cell-${i}-${j}`}
              x={labelWidth + j * cellSize}
              y={0}
              width={cellSize}
              height={cellSize}
              fill={colour(v)}
              stroke="rgba(0,0,0,0.05)"
            >
              <title>
                row {i}, col {j}: {v.toFixed(3)}
              </title>
            </rect>
          ))}
        </g>
      ))}
    </svg>
  );
}
