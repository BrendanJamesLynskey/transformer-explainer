/**
 * Chapter 13's measured comparison (results.md §13): four simulated workers
 * synchronised four ways, each at its best learning rate, mean of three
 * seeds. Left: validation loss against step. Right: against the bytes each
 * worker has sent (log scale), where the compressors and DiLoCo pull ahead.
 * Server-rendered (LineChart), no JavaScript.
 */
import results from "@/data/training/results.json";
import { OKABE_ITO } from "@/lib/viz/palette";

import { LineChart, type Marker } from "./LineChart";

const STYLE: Record<string, { colour: string; dash?: string; marker: Marker }> =
  {
    dense: { colour: OKABE_ITO.purple, marker: "none" },
    topk: { colour: OKABE_ITO.orange, dash: "6 3", marker: "square" },
    demo: { colour: OKABE_ITO.vermillion, marker: "triangle" },
    "diloco H=10": { colour: OKABE_ITO.blue, dash: "2 3", marker: "circle" },
    "diloco H=50": {
      colour: OKABE_ITO.green,
      dash: "10 3 2 3",
      marker: "diamond",
    },
  };
const NAME: Record<string, string> = {
  dense: "dense AdamW",
  topk: "top-k + error feedback",
  demo: "DeMo",
  "diloco H=10": "DiLoCo, H = 10",
  "diloco H=50": "DiLoCo, H = 50",
};

export function SlowLinksCharts(): JSX.Element {
  const x = results.slowLinks;
  const series = (byBytes: boolean) =>
    x.rows.map((r) => {
      const st = STYLE[r.name] ?? {
        colour: OKABE_ITO.purple,
        marker: "none" as Marker,
      };
      return {
        name: `${NAME[r.name] ?? r.name} (lr ${r.bestLr})`,
        colour: st.colour,
        dash: st.dash,
        marker: "none" as Marker,
        points: r.evalSteps
          .map((t, i) => ({
            x: byBytes ? Math.max(1, t * r.bytes.wire) : t,
            y: r.meanEvals[i] ?? null,
          }))
          .filter((p, i) => (byBytes ? i > 0 : true)),
      };
    });
  const maxBytes = Math.max(...x.rows.map((r) => r.bytes.wire * x.steps));
  const minBytes = Math.min(...x.rows.map((r) => r.bytes.wire * x.evalEvery));
  const decades: number[] = [];
  for (
    let e = Math.floor(Math.log10(minBytes));
    e <= Math.ceil(Math.log10(maxBytes));
    e++
  )
    decades.push(10 ** e);
  return (
    <div data-testid="slow-links-charts">
      <LineChart
        testId="slow-links-steps"
        label={`Validation loss against step for four ways of syncing four workers: ${x.rows
          .map(
            (r) =>
              `${r.name} ${r.mean === null ? "diverged" : r.mean.toFixed(3)}`,
          )
          .join(", ")}`}
        series={series(false)}
        xDomain={[0, x.steps]}
        yDomain={[2.0, 4.2]}
        xTicks={[0, 500, 1000, 1500, 2000]}
        yTicks={[2.0, 2.5, 3.0, 3.5, 4.0]}
        xLabel="step"
        yLabel="validation loss (mean of 3 seeds)"
        caption={
          <>
            Each method at the best of three learning rates; final validation
            loss (mean ± sd):{" "}
            {x.rows
              .map(
                (r) =>
                  `${NAME[r.name] ?? r.name} ${r.mean === null ? "–" : r.mean.toFixed(3)} ± ${r.sd === null ? "–" : r.sd.toFixed(3)}`,
              )
              .join("; ")}
            . results.md §13.
          </>
        }
      />
      <LineChart
        testId="slow-links-bytes"
        label="Validation loss against the bytes each worker has sent, log scale"
        series={series(true)}
        xDomain={[decades[0]!, maxBytes * 1.6]}
        yDomain={[2.0, 4.2]}
        xLog
        xTicks={decades.filter((d) => d <= maxBytes)}
        yTicks={[2.0, 2.5, 3.0, 3.5, 4.0]}
        xFmt={(v) => {
          if (v >= 1e9) return `${v / 1e9} GB`;
          if (v >= 1e6) return `${v / 1e6} MB`;
          if (v >= 1e3) return `${v / 1e3} kB`;
          return `${v} B`;
        }}
        xLabel="bytes sent per worker so far (log scale)"
        yLabel="validation loss"
        caption={
          <>
            The same runs against what each worker has put on the wire (BF16
            values, ring collectives among 4 workers):{" "}
            {x.rows
              .map(
                (r) =>
                  `${NAME[r.name] ?? r.name} ${r.bytes.wire.toLocaleString("en-GB", { maximumFractionDigits: 0 })} bytes per step`,
              )
              .join("; ")}
            .
          </>
        }
      />
    </div>
  );
}
