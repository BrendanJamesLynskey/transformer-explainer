/**
 * Chapter 6: the learning-rate sweep (results.md §6a). Final validation
 * loss against the peak learning rate, mean ± sd over three seeds; diverged
 * runs are counted in the caption.
 */
import results from "@/data/training/results.json";
import { OKABE_ITO } from "@/lib/viz/palette";

import { LineChart } from "./LineChart";

export function LrSweepChart(): JSX.Element {
  const x = results.lrSweep;
  const pts = x.rows.map((r) => ({
    x: r.lr,
    y: r.mean,
    lo: r.mean !== null && r.sd !== null ? r.mean - r.sd : undefined,
    hi: r.mean !== null && r.sd !== null ? r.mean + r.sd : undefined,
  }));
  const div = x.rows.filter((r) => r.diverged > 0);
  return (
    <LineChart
      testId="lr-sweep-chart"
      label={`Final validation loss against peak learning rate for AdamW: ${x.rows
        .map(
          (r) => `${r.lr}: ${r.mean === null ? "diverged" : r.mean.toFixed(3)}`,
        )
        .join(", ")}`}
      series={[
        {
          name: "AdamW, warmup + cosine, 2,000 steps",
          colour: OKABE_ITO.blue,
          marker: "circle",
          points: pts,
        },
      ]}
      xDomain={[0.0007, 1.4]}
      yDomain={[2.0, 3.4]}
      xTicks={x.rows.map((r) => r.lr)}
      yTicks={[2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2]}
      xLog
      xLabel="peak learning rate (log scale)"
      yLabel="final validation loss"
      caption={
        <>
          Mean ± standard deviation over three seeds; best mean at {x.bestLr}.
          {div.length
            ? ` Diverged: ${div.map((r) => `${r.diverged} of 3 at ${r.lr}`).join(", ")}.`
            : " None diverged here, though 1.0 is on the edge (results.md §6)."}{" "}
          results.md §6a.
        </>
      }
    />
  );
}
