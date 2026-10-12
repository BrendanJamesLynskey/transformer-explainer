/**
 * Chapter 9: the default run in each emulated precision recipe
 * (results.md §9a): final validation loss, mean ± sd over three seeds, as
 * horizontal bars. Server-rendered twice: a wide drawing (labels beside the
 * bars) and a phone drawing (labels above them).
 */
import results from "@/data/training/results.json";
import { OKABE_ITO } from "@/lib/viz/palette";

const LO = 2.0;
const HI = 3.0;

function Plot({ phone }: { phone: boolean }): JSX.Element {
  const rows = results.precision.rows;
  const W = phone ? 360 : 640;
  const labelW = phone ? 0 : 250;
  const rowH = phone ? 46 : 30;
  const top = 24;
  const H = top + rows.length * rowH + 30;
  const X = (v: number) =>
    labelW + 8 + ((W - labelW - 8 - 70) * (Math.min(HI, v) - LO)) / (HI - LO);
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className={`${phone ? "sm:hidden" : "hidden sm:block"} w-full text-neutral-800 dark:text-neutral-200`}
      role="img"
      aria-label={`Final validation loss by precision recipe: ${rows
        .map(
          (r) =>
            `${r.name}: ${r.mean === null ? "diverged" : r.mean.toFixed(3)}`,
        )
        .join("; ")}`}
    >
      <text x={X(LO)} y={14} fontSize={12} fill="currentColor">
        final validation loss (mean ± sd, 3 seeds)
      </text>
      {[2.0, 2.2, 2.4, 2.6, 2.8, 3.0].map((v) => (
        <g key={v}>
          <line
            x1={X(v)}
            x2={X(v)}
            y1={top}
            y2={H - 26}
            stroke="currentColor"
            strokeOpacity={0.12}
          />
          <text
            x={X(v)}
            y={H - 10}
            fontSize={12}
            textAnchor="middle"
            fill="currentColor"
          >
            {v.toFixed(1)}
          </text>
        </g>
      ))}
      {rows.map((r, i) => {
        const y = top + i * rowH;
        const barY = phone ? y + 18 : y + 6;
        const barH = phone ? rowH - 24 : rowH - 12;
        const mid = barY + barH / 2;
        return (
          <g key={r.name}>
            <text
              x={phone ? X(LO) : labelW}
              y={phone ? y + 13 : mid + 4}
              fontSize={12}
              textAnchor={phone ? "start" : "end"}
              fill="currentColor"
            >
              {r.name}
            </text>
            {r.mean === null ? (
              <text
                x={X(LO) + 4}
                y={mid + 4}
                fontSize={12}
                fill={OKABE_ITO.vermillion}
                fontWeight={600}
              >
                diverged ({r.diverged} of 3 seeds)
              </text>
            ) : (
              <>
                <rect
                  x={X(LO)}
                  y={barY}
                  width={X(r.mean) - X(LO)}
                  height={barH}
                  fill={OKABE_ITO.blue}
                  fillOpacity={0.8}
                />
                {r.sd !== null && (
                  <line
                    x1={X(r.mean - r.sd)}
                    x2={X(r.mean + r.sd)}
                    y1={mid}
                    y2={mid}
                    stroke="currentColor"
                    strokeWidth={2}
                  />
                )}
                <text
                  x={X(r.mean + (r.sd ?? 0)) + 6}
                  y={mid + 4}
                  fontSize={12}
                  fill="currentColor"
                >
                  {r.mean.toFixed(3)}
                </text>
              </>
            )}
          </g>
        );
      })}
    </svg>
  );
}

export function PrecisionChart(): JSX.Element {
  return (
    <figure className="my-6" data-testid="precision-chart">
      <Plot phone={false} />
      <Plot phone />
      <figcaption className="text-xs text-neutral-600 dark:text-neutral-400">
        Emulated on the CPU by rounding in float64
        (scripts/training/precision.py); the GPU run of the same comparison is
        pending (notebook 09). results.md §9a.
      </figcaption>
    </figure>
  );
}
