/**
 * A server-rendered line chart for Part 2's measured results (no
 * JavaScript): lines, markers (a different shape per series, so colour is
 * never the only cue), optional error bars (mean ± sd over seeds), linear
 * or log axes. Each chart in chapters 6-10 is a thin wrapper that reads
 * results.json and hands its numbers here.
 */
import type { ReactNode } from "react";

export type Point = { x: number; y: number | null; lo?: number; hi?: number };
export type Marker = "circle" | "square" | "triangle" | "diamond" | "none";
export type Series = {
  name: string;
  colour: string;
  dash?: string;
  marker?: Marker;
  points: Point[];
};

function MarkerShape({
  kind,
  x,
  y,
  fill,
}: {
  kind: Marker;
  x: number;
  y: number;
  fill: string;
}): JSX.Element | null {
  const r = 4.5;
  if (kind === "none") return null;
  if (kind === "circle")
    return <circle cx={x} cy={y} r={r} fill={fill} stroke="currentColor" />;
  if (kind === "square")
    return (
      <rect
        x={x - r}
        y={y - r}
        width={2 * r}
        height={2 * r}
        fill={fill}
        stroke="currentColor"
      />
    );
  if (kind === "triangle")
    return (
      <path
        d={`M${x},${y - r * 1.2}L${x + r * 1.1},${y + r * 0.8}L${x - r * 1.1},${y + r * 0.8}Z`}
        fill={fill}
        stroke="currentColor"
      />
    );
  return (
    <path
      d={`M${x},${y - r * 1.3}L${x + r * 1.1},${y}L${x},${y + r * 1.3}L${x - r * 1.1},${y}Z`}
      fill={fill}
      stroke="currentColor"
    />
  );
}

export function LineChart({
  testId,
  label,
  series,
  xDomain,
  yDomain,
  xTicks,
  yTicks,
  xLabel,
  yLabel,
  xLog = false,
  yLog = false,
  xFmt = (v: number) => String(v),
  yFmt = (v: number) => String(v),
  caption,
  markers,
  width = 640,
  height = 300,
}: {
  testId: string;
  /** the chart's accessible description (what it shows, with the key numbers) */
  label: string;
  series: Series[];
  xDomain: [number, number];
  yDomain: [number, number];
  xTicks: number[];
  yTicks: number[];
  xLabel: string;
  yLabel: string;
  xLog?: boolean;
  yLog?: boolean;
  xFmt?: (v: number) => string;
  yFmt?: (v: number) => string;
  caption: ReactNode;
  /** extra marks, e.g. a vertical line at an event: [x, label] */
  markers?: [number, string][];
  width?: number;
  height?: number;
}): JSX.Element {
  // Two drawings of the same chart: the wide one for tablets and up, a
  // narrow one for phones, so the labels stay readable (at least 11 px) on
  // a 390 px screen without any JavaScript.
  const plot = (W: number, H: number, phone: boolean, cls: string) => {
    const left = phone ? 44 : 52;
    const right = 12;
    const top = 22;
    const bottom = 40;
    const tx = (v: number) => (xLog ? Math.log10(v) : v);
    const ty = (v: number) => (yLog ? Math.log10(v) : v);
    const [x0, x1] = [tx(xDomain[0]), tx(xDomain[1])];
    const [y0, y1] = [ty(yDomain[0]), ty(yDomain[1])];
    const X = (v: number) =>
      left + ((W - left - right) * (tx(v) - x0)) / (x1 - x0);
    const Y = (v: number) => {
      const c = Math.min(Math.max(ty(v), y0), y1);
      return top + ((H - top - bottom) * (y1 - c)) / (y1 - y0);
    };
    return (
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className={`${cls} w-full text-neutral-800 dark:text-neutral-200`}
        role="img"
        aria-label={label}
      >
        <text x={left} y={13} fontSize={12} fill="currentColor">
          {yLabel}
        </text>
        {yTicks.map((v) => (
          <g key={`y${v}`}>
            <line
              x1={left}
              x2={W - right}
              y1={Y(v)}
              y2={Y(v)}
              stroke="currentColor"
              strokeOpacity={0.12}
            />
            <text
              x={left - 6}
              y={Y(v) + 4}
              fontSize={12}
              textAnchor="end"
              fill="currentColor"
            >
              {yFmt(v)}
            </text>
          </g>
        ))}
        {xTicks.map((v) => (
          <text
            key={`x${v}`}
            x={X(v)}
            y={H - bottom + 18}
            fontSize={12}
            textAnchor="middle"
            fill="currentColor"
          >
            {xFmt(v)}
          </text>
        ))}
        <text
          x={(W + left) / 2}
          y={H - 4}
          fontSize={12}
          textAnchor="middle"
          fill="currentColor"
        >
          {xLabel}
        </text>
        {(markers ?? []).map(([v, text]) => (
          <g key={`m${v}`}>
            <line
              x1={X(v)}
              x2={X(v)}
              y1={top}
              y2={H - bottom}
              stroke="currentColor"
              strokeOpacity={0.4}
              strokeDasharray="4 3"
            />
            <text x={X(v) + 4} y={top + 12} fontSize={12} fill="currentColor">
              {text}
            </text>
          </g>
        ))}
        {series.map((s) => {
          const pts = s.points.filter((p) => p.y !== null);
          const d = pts
            .map(
              (p, i) =>
                `${i ? "L" : "M"}${X(p.x).toFixed(1)},${Y(p.y!).toFixed(1)}`,
            )
            .join("");
          return (
            <g key={s.name}>
              <path
                d={d}
                fill="none"
                stroke={s.colour}
                strokeWidth={2.5}
                strokeDasharray={s.dash}
              />
              {pts.map((p, i) =>
                p.lo !== undefined && p.hi !== undefined ? (
                  <line
                    key={`e${i}`}
                    x1={X(p.x)}
                    x2={X(p.x)}
                    y1={Y(p.lo)}
                    y2={Y(p.hi)}
                    stroke={s.colour}
                    strokeWidth={2}
                  />
                ) : null,
              )}
              {pts.map((p, i) => (
                <MarkerShape
                  key={`p${i}`}
                  kind={s.marker ?? "none"}
                  x={X(p.x)}
                  y={Y(p.y!)}
                  fill={s.colour}
                />
              ))}
            </g>
          );
        })}
      </svg>
    );
  };
  return (
    <figure className="my-6" data-testid={testId}>
      {plot(width, height, false, "hidden sm:block")}
      {plot(360, Math.round(height * 0.95), true, "sm:hidden")}
      <figcaption className="text-xs text-neutral-600 dark:text-neutral-400">
        <span className="mr-2">
          {series.map((s, i) => (
            <span key={s.name}>
              {i ? " · " : ""}
              <span style={{ color: s.colour }} aria-hidden>
                {s.marker === "square"
                  ? "■"
                  : s.marker === "triangle"
                    ? "▲"
                    : s.marker === "diamond"
                      ? "◆"
                      : s.marker === "circle"
                        ? "●"
                        : "━"}
              </span>{" "}
              {s.name}
            </span>
          ))}
          .
        </span>
        {caption}
      </figcaption>
    </figure>
  );
}
