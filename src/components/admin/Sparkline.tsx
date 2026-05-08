/**
 * Tiny dependency-free SVG sparkline.
 *
 * Pure server component — D3 would be overkill for one polyline.
 */
type Point = { day: string; users: number };

export function Sparkline({
  points,
  width = 480,
  height = 80,
}: {
  points: Point[];
  width?: number;
  height?: number;
}): JSX.Element {
  if (points.length === 0) {
    return (
      <p className="font-mono text-xs text-neutral-500">
        no events in window — interact with a page to populate
      </p>
    );
  }
  const max = Math.max(1, ...points.map((p) => p.users));
  const stepX = points.length > 1 ? width / (points.length - 1) : 0;
  const path = points
    .map((p, i) => {
      const x = i * stepX;
      const y = height - (p.users / max) * (height - 8) - 4;
      return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <figure className="space-y-2">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        width={width}
        height={height}
        role="img"
        aria-label={`Daily active sessions over the last ${points.length} days, peak ${max}`}
        className="block"
      >
        <path
          d={path}
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          className="text-accent"
        />
        {points.map((p, i) => (
          <circle
            key={p.day}
            cx={i * stepX}
            cy={height - (p.users / max) * (height - 8) - 4}
            r="2"
            className="fill-accent"
          >
            <title>
              {p.day} — {p.users} session{p.users === 1 ? "" : "s"}
            </title>
          </circle>
        ))}
      </svg>
      <figcaption className="font-mono text-[11px] text-neutral-500">
        {points[0]?.day} → {points.at(-1)?.day} · peak {max}
      </figcaption>
    </figure>
  );
}
