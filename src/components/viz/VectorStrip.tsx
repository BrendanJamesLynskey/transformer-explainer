/**
 * One vector drawn as a row of heat-map cells, inside an animation's SVG:
 * positive values blue, negative orange (palette.ts `heat`). A null value
 * (not computed yet) is an empty dashed cell; `active` outlines one cell.
 * The label sits above the strip.
 */
import { heat } from "@/lib/viz/palette";

export function VectorStrip({
  x,
  y,
  values,
  cell,
  height,
  max,
  label,
  fontSize,
  active = -1,
  activeColour = "currentColor",
  muted = false,
  testId,
}: {
  x: number;
  y: number;
  values: readonly (number | null)[];
  cell: number;
  height: number;
  /** The value that gets full colour (|v| ≥ max). */
  max: number;
  label?: string;
  fontSize: number;
  active?: number;
  activeColour?: string;
  muted?: boolean;
  testId?: string;
}): JSX.Element {
  return (
    <g data-testid={testId}>
      {label && (
        <text x={x} y={y - 4} fontSize={fontSize} fill="currentColor">
          {label}
        </text>
      )}
      {values.map((v, k) => {
        const h = v === null ? null : heat(v, max);
        return (
          <rect
            key={k}
            x={x + k * cell + 1}
            y={y}
            width={cell - 2}
            height={height}
            rx={2}
            fill={h ? h.fill : "none"}
            fillOpacity={h ? h.fillOpacity * (muted ? 0.45 : 1) : 0}
            stroke={k === active ? activeColour : "currentColor"}
            strokeOpacity={k === active ? 1 : v === null ? 0.3 : 0.15}
            strokeWidth={k === active ? 2.5 : 1}
            strokeDasharray={v === null ? "3 2" : undefined}
          />
        );
      })}
    </g>
  );
}
