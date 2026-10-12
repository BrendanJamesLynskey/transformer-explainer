/**
 * Chapter 12: the systems model against published runs (results.md §11a).
 * For each run, the published throughput per GPU (circle) and the model's
 * prediction (square), as a percentage of the GPU's peak; the fitted run is
 * marked. Server-rendered SVG, with a narrow drawing for phones.
 */
import results from "@/data/training/results.json";
import { OKABE_ITO } from "@/lib/viz/palette";

export function ValidationChart(): JSX.Element {
  const v = results.systems.validation;
  const peak = (hw: string) =>
    results.systems.hardware[hw as "a100" | "h100"].peak;
  // the published throughput as a share of peak, from the TFLOP/s (the
  // papers round the percentage)
  const rows = v.rows.map((r) => ({
    ...r,
    pubPct: (100 * r.published) / peak(r.hw),
  }));
  const lo = 30;
  const hi = 55;
  const plot = (W: number, phone: boolean, cls: string) => {
    const left = phone ? 150 : 210;
    const right = 14;
    const top = 24;
    const rowH = phone ? 24 : 22;
    const H = top + rows.length * rowH + 40;
    const X = (pct: number) =>
      left + ((W - left - right) * (pct - lo)) / (hi - lo);
    return (
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className={`${cls} w-full text-neutral-800 dark:text-neutral-200`}
        role="img"
        aria-label={`Published and predicted throughput per GPU as a share of peak for ${rows.length} runs; largest error ${(100 * v.maxAbsError).toFixed(1)}%`}
      >
        <text x={left} y={13} fontSize={12} fill="currentColor">
          throughput per GPU, % of peak
        </text>
        {[30, 35, 40, 45, 50, 55].map((t) => (
          <g key={t}>
            <line
              x1={X(t)}
              x2={X(t)}
              y1={top}
              y2={top + rows.length * rowH}
              stroke="currentColor"
              strokeOpacity={0.12}
            />
            <text
              x={X(t)}
              y={top + rows.length * rowH + 16}
              fontSize={12}
              textAnchor="middle"
              fill="currentColor"
            >
              {t}%
            </text>
          </g>
        ))}
        {rows.map((r, i) => {
          const y = top + i * rowH + rowH / 2;
          const llama = `${r.gpus >= 16384 ? "16K" : "8K"} GPUs${r.label.includes("131") ? ", 128K" : ""}`;
          const label =
            r.source === "megatron"
              ? `Megatron ${r.label}`
              : phone
                ? `Llama 3 ${llama}`
                : `Llama 3 405B, ${llama}`;
          return (
            <g key={i} data-testid="validation-row">
              <text
                x={left - 6}
                y={y + 4}
                fontSize={12}
                textAnchor="end"
                fill="currentColor"
              >
                {label}
                {r.fitted ? " (fit)" : ""}
              </text>
              <line
                x1={X(r.pubPct)}
                x2={X(r.predictedPct)}
                y1={y}
                y2={y}
                stroke="currentColor"
                strokeOpacity={0.4}
              />
              <circle
                cx={X(r.pubPct)}
                cy={y}
                r={5}
                fill={OKABE_ITO.blue}
                stroke="currentColor"
              />
              <rect
                x={X(r.predictedPct) - 4.5}
                y={y - 4.5}
                width={9}
                height={9}
                fill={OKABE_ITO.orange}
                stroke="currentColor"
              />
            </g>
          );
        })}
        <text x={left} y={H - 4} fontSize={12} fill="currentColor">
          A100 (Megatron, peak 312 TFLOP/s) · H100 (Llama 3, 989)
        </text>
      </svg>
    );
  };
  return (
    <figure className="my-6" data-testid="validation-chart">
      {plot(640, false, "hidden sm:block")}
      {plot(360, true, "sm:hidden")}
      <figcaption className="text-xs text-neutral-600 dark:text-neutral-400">
        <span style={{ color: OKABE_ITO.blue }} aria-hidden>
          ●
        </span>{" "}
        published ·{" "}
        <span style={{ color: OKABE_ITO.orange }} aria-hidden>
          ■
        </span>{" "}
        the systems model. One constant (η = {v.eta.toFixed(3)}) fitted on
        Megatron&rsquo;s 1.7B run; the other {rows.length - 1} rows are
        predictions, every one within {(100 * v.maxAbsError).toFixed(1)}% of the
        published number and below it (mean error{" "}
        {(100 * v.meanAbsError).toFixed(1)}%). Narayanan et al. 2021, Table 1;
        Grattafiori et al. 2024, Table 4; results.md §11a. Megatron counts
        hardware FLOPs (with recomputation), Llama 3 model FLOPs (MFU); the
        model counts each the same way as its paper.
      </figcaption>
    </figure>
  );
}
