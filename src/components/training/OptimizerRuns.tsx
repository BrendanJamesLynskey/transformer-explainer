/**
 * Chapter 5's measured comparison on the real model (results.md §5): the
 * validation loss of each optimiser at its best learning rate, every 100
 * steps, from the recorded runs. Server-rendered SVG: no JavaScript.
 */
import results from "@/data/training/results.json";
import { OKABE_ITO } from "@/lib/viz/palette";

const ORDER = ["sgd", "momentum", "adam", "adamw"] as const;
const LABEL = {
  sgd: "SGD",
  momentum: "Momentum",
  adam: "Adam",
  adamw: "AdamW",
} as const;
const COLOUR = {
  sgd: OKABE_ITO.orange,
  momentum: OKABE_ITO.green,
  adam: OKABE_ITO.blue,
  adamw: OKABE_ITO.purple,
} as const;
const DASH = {
  sgd: "",
  momentum: "6 3",
  adam: "2 3",
  adamw: "10 3 2 3",
} as const;

export function OptimizerRuns(): JSX.Element {
  const best = ORDER.map((k) => {
    const rows = results.optimizerCompare.filter((r) => r.kind === k);
    return rows.reduce((a, b) => (b.final < a.final ? b : a));
  });
  const W = 640;
  const H = 300;
  const left = 44;
  const bottom = 34;
  const steps = best[0]!.evals[best[0]!.evals.length - 1]!.step;
  const yLo = 2.0;
  const yHi = 4.3;
  const x = (t: number) => left + ((W - left - 10) * t) / steps;
  const y = (v: number) =>
    8 + ((H - bottom - 8) * (yHi - Math.min(yHi, v))) / (yHi - yLo);
  return (
    <figure className="my-6" data-testid="optimizer-runs">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full text-neutral-800 dark:text-neutral-200"
        role="img"
        aria-label={`Validation loss against step for four optimisers at their best learning rates: ${best
          .map(
            (b) =>
              `${LABEL[b.kind as keyof typeof LABEL]} ${b.final.toFixed(3)}`,
          )
          .join(", ")} after ${steps} steps`}
      >
        {[2.0, 2.5, 3.0, 3.5, 4.0].map((v) => (
          <g key={v}>
            <line
              x1={left}
              x2={W - 10}
              y1={y(v)}
              y2={y(v)}
              stroke="currentColor"
              strokeOpacity={0.12}
            />
            <text
              x={left - 6}
              y={y(v) + 4}
              fontSize={12}
              textAnchor="end"
              fill="currentColor"
            >
              {v.toFixed(1)}
            </text>
          </g>
        ))}
        {[0, 500, 1000, 1500].map((t) => (
          <text
            key={t}
            x={x(t)}
            y={H - bottom + 18}
            fontSize={12}
            textAnchor="middle"
            fill="currentColor"
          >
            {t}
          </text>
        ))}
        <text
          x={(W + left) / 2}
          y={H - 2}
          fontSize={12}
          textAnchor="middle"
          fill="currentColor"
        >
          step (validation loss every 100 steps)
        </text>
        {best.map((b) => {
          const k = b.kind as keyof typeof COLOUR;
          const d = b.evals
            .map(
              (e, i) =>
                `${i ? "L" : "M"}${x(e.step).toFixed(1)},${y(e.val).toFixed(1)}`,
            )
            .join("");
          return (
            <path
              key={k}
              d={d}
              fill="none"
              stroke={COLOUR[k]}
              strokeWidth={2.5}
              strokeDasharray={DASH[k]}
            />
          );
        })}
      </svg>
      <figcaption className="text-xs text-neutral-600 dark:text-neutral-400">
        {best.map((b, i) => {
          const k = b.kind as keyof typeof COLOUR;
          return (
            <span key={k}>
              {i ? " · " : ""}
              <span style={{ color: COLOUR[k] }}>━</span> {LABEL[k]} (lr {b.lr}
              ): {b.final.toFixed(3)}
            </span>
          );
        })}
        . Constant learning rate, 1,500 steps, the same batches; results.md §5.
      </figcaption>
    </figure>
  );
}
