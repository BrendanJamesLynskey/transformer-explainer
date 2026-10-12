/**
 * Chapter 7: muP's coordinate check (results.md §7d): how far 10 AdamW
 * steps at one learning rate move the logits, against the width.
 */
import results from "@/data/training/results.json";
import { OKABE_ITO } from "@/lib/viz/palette";

import { LineChart } from "./LineChart";

export function CoordCheckChart(): JSX.Element {
  const x = results.coordCheck;
  return (
    <LineChart
      testId="coord-check-chart"
      label={`RMS change of the logits after ${x.steps} steps against width; standard: ${x.sp
        .map((p) => p.logitsDelta.toFixed(2))
        .join(
          ", ",
        )}; muP: ${x.mup.map((p) => p.logitsDelta.toFixed(2)).join(", ")}`}
      series={[
        {
          name: "standard parameterisation",
          colour: OKABE_ITO.vermillion,
          marker: "triangle",
          points: x.sp.map((p) => ({ x: p.width, y: p.logitsDelta })),
        },
        {
          name: "muP (base width 16)",
          colour: OKABE_ITO.green,
          dash: "6 3",
          marker: "square",
          points: x.mup.map((p) => ({ x: p.width, y: p.logitsDelta })),
        },
      ]}
      xDomain={[12, 340]}
      yDomain={[0.3, 30]}
      xTicks={x.widths}
      yTicks={[0.3, 1, 3, 10, 30]}
      xLog
      yLog
      xLabel="width d_model (log scale)"
      yLabel={`RMS change of the logits after ${x.steps} steps (log scale)`}
      caption={
        <>
          Learning rate {x.lr}, the same batches for every width. results.md
          §7d.
        </>
      }
    />
  );
}
