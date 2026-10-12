/**
 * Chapter 8: one bad batch (results.md §8a). Validation loss around the
 * corrupted step for each guard, seed 1 (the table gives all three seeds).
 */
import results from "@/data/training/results.json";
import { OKABE_ITO } from "@/lib/viz/palette";

import { LineChart, type Marker } from "./LineChart";

const STYLE: { colour: string; dash: string; marker: Marker }[] = [
  { colour: OKABE_ITO.vermillion, dash: "", marker: "triangle" },
  { colour: OKABE_ITO.blue, dash: "6 3", marker: "circle" },
  { colour: OKABE_ITO.green, dash: "2 3", marker: "square" },
  { colour: OKABE_ITO.purple, dash: "10 3 2 3", marker: "diamond" },
];

export function BadBatchChart(): JSX.Element {
  const x = results.badBatch;
  const lo = x.at - 100;
  return (
    <LineChart
      testId="bad-batch-chart"
      label={`Validation loss around the bad batch at step ${x.at}: ${x.rows
        .map(
          (r) =>
            `${r.name}: ${r.after[0] === null ? "diverged" : r.after[0]!.toFixed(3)} just after`,
        )
        .join("; ")}`}
      series={x.rows.map((r, k) => ({
        name: r.name,
        ...STYLE[k]!,
        marker: "none" as Marker,
        points: r.runs[0]!.evals.filter((e) => e.step >= lo).map((e) => ({
          x: e.step,
          y: e.val === null ? 1e6 : Math.min(e.val, 1e6),
        })),
      }))}
      xDomain={[lo, x.steps]}
      yDomain={[2, 1e6]}
      xTicks={[lo, x.at, x.at + 100, x.at + 200, x.steps]}
      yTicks={[2, 10, 100, 1e3, 1e4, 1e5, 1e6]}
      yLog
      yFmt={(v) => (v >= 1000 ? `1e${Math.log10(v)}` : String(v))}
      markers={[[x.at, "bad batch"]]}
      xLabel="step"
      yLabel="validation loss (log scale; 1e6 = diverged)"
      caption={<>Seed 1; all three seeds are in results.md §8a.</>}
    />
  );
}
