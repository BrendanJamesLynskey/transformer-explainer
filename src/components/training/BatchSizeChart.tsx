/**
 * Chapter 6: steps to reach a target validation loss against the batch
 * size (results.md §6c), with the fit S = S_min (1 + B_crit / B)
 * (McCandlish et al. 2018).
 */
import results from "@/data/training/results.json";
import { OKABE_ITO } from "@/lib/viz/palette";

import { LineChart } from "./LineChart";

export function BatchSizeChart(): JSX.Element {
  const x = results.batchSize;
  const fit = x.fit;
  const rows = x.rows.filter((r) => r.best !== null);
  const fitPts = [];
  for (let b = 2; b <= 128; b *= Math.SQRT2)
    fitPts.push({ x: b, y: fit.sMin * (1 + fit.bCrit / b) });
  return (
    <LineChart
      testId="batch-size-chart"
      label={`Steps to reach validation loss ${x.target} against batch size: ${rows
        .map((r) => `${r.batch} windows: ${Math.round(r.steps!)} steps`)
        .join(", ")}; fitted critical batch ${fit.bCrit.toFixed(0)} windows`}
      series={[
        {
          name: "measured (mean ± sd, 3 seeds)",
          colour: OKABE_ITO.blue,
          marker: "circle",
          points: rows.map((r) => ({
            x: r.batch,
            y: r.steps,
            lo: r.best!.mean! - r.best!.sd!,
            hi: r.best!.mean! + r.best!.sd!,
          })),
        },
        {
          name: `fit: S_min (1 + B_crit / B), B_crit ≈ ${fit.bCrit.toFixed(0)}`,
          colour: OKABE_ITO.orange,
          dash: "6 3",
          points: fitPts,
        },
      ]}
      xDomain={[3, 90]}
      yDomain={[50, 12000]}
      xTicks={x.rows.map((r) => r.batch)}
      yTicks={[100, 300, 1000, 3000, 10000]}
      xLog
      yLog
      xFmt={(v) => String(v)}
      yFmt={(v) => v.toLocaleString("en-GB")}
      xLabel="batch size (windows of 8 characters, log scale)"
      yLabel={`steps to validation loss ${x.target} (log scale)`}
      caption={
        <>
          Each batch size at the better of two learning rates. Doubling a small
          batch nearly halves the steps; past the critical batch it barely
          helps, so the extra windows are wasted. results.md §6c.
        </>
      }
    />
  );
}
