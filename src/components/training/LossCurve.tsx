/**
 * The Training site's landing figure: the validation loss of chapter 1's default run, from the
 * recorded, seeded run (src/data/training/results.json), falling from random guessing. Static
 * SVG, rendered on the server.
 */
import results from "@/data/training/results.json";

export function LossCurve(): JSX.Element {
  const run = results.defaultRun;
  const ev = run.evals;
  const steps = ev[ev.length - 1]!.step;
  const hi = Math.ceil(ev[0]!.val * 2) / 2;
  const lo = Math.floor(Math.min(...ev.map((e) => e.val)) * 2) / 2;
  const W = 300;
  const H = 180;
  const x = (st: number) => 34 + ((W - 44) * st) / steps;
  const y = (v: number) => 10 + ((H - 40) * (hi - v)) / (hi - lo);
  const pts = ev.map((e) => `${x(e.step)},${y(e.val)}`).join(" ");
  const ticks: number[] = [];
  for (let t = Math.ceil(lo); t <= hi; t++) ticks.push(t);
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="h-auto w-full"
      role="img"
      aria-label={`Validation loss falling from ${ev[0]!.val.toFixed(2)} to ${ev[ev.length - 1]!.val.toFixed(2)} over ${steps.toLocaleString("en-GB")} steps.`}
    >
      {ticks.map((t) => (
        <g key={t}>
          <line
            x1={34}
            x2={W - 10}
            y1={y(t)}
            y2={y(t)}
            className="stroke-neutral-200 dark:stroke-neutral-800"
          />
          <text
            x={28}
            y={y(t) + 4}
            textAnchor="end"
            fontSize={11}
            className="fill-neutral-600 dark:fill-neutral-400"
          >
            {t.toFixed(1)}
          </text>
        </g>
      ))}
      <polyline
        points={pts}
        fill="none"
        stroke="#E69F00"
        strokeWidth={2.5}
        strokeLinejoin="round"
      />
      {ev.map((e) => (
        <circle
          key={e.step}
          cx={x(e.step)}
          cy={y(e.val)}
          r={2.5}
          fill="#E69F00"
        />
      ))}
      <text
        x={34}
        y={H - 8}
        fontSize={11}
        className="fill-neutral-600 dark:fill-neutral-400"
      >
        step 0
      </text>
      <text
        x={W - 10}
        y={H - 8}
        textAnchor="end"
        fontSize={11}
        className="fill-neutral-600 dark:fill-neutral-400"
      >
        {steps.toLocaleString("en-GB")}
      </text>
    </svg>
  );
}
