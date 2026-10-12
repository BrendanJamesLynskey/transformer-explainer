/**
 * Chapter 7: learning-rate transfer across width (results.md §7e). Final
 * validation loss against the peak learning rate for width 16 and width 64
 * under the standard parameterisation and under muP (base width 16).
 */
import results from "@/data/training/results.json";
import { OKABE_ITO } from "@/lib/viz/palette";

import { LineChart } from "./LineChart";

export function MupChart(): JSX.Element {
  const x = results.mupTransfer;
  const pick = (param: string, width: number) =>
    x.rows
      .filter((r) => r.param === param && r.width === width)
      .map((r) => ({
        x: r.lr,
        y: r.mean,
        lo: r.mean !== null && r.sd !== null ? r.mean - r.sd : undefined,
        hi: r.mean !== null && r.sd !== null ? r.mean + r.sd : undefined,
      }));
  const [w0, w1] = x.widths;
  return (
    <LineChart
      testId="mup-chart"
      label={`Final validation loss against learning rate: width ${w0} (either parameterisation), width ${w1} standard and width ${w1} muP`}
      series={[
        {
          name: `width ${w0} (both)`,
          colour: OKABE_ITO.blue,
          marker: "circle",
          points: pick("sp", w0!),
        },
        {
          name: `width ${w1}, standard`,
          colour: OKABE_ITO.vermillion,
          dash: "6 3",
          marker: "triangle",
          points: pick("sp", w1!),
        },
        {
          name: `width ${w1}, muP`,
          colour: OKABE_ITO.green,
          dash: "2 3",
          marker: "square",
          points: pick("mup", w1!),
        },
      ]}
      xDomain={[0.007, 0.14]}
      yDomain={[2.0, 3.0]}
      xTicks={x.lrs}
      yTicks={[2.0, 2.2, 2.4, 2.6, 2.8, 3.0]}
      xLog
      xLabel="peak learning rate (log scale)"
      yLabel={`validation loss after ${x.steps.toLocaleString("en-GB")} steps`}
      caption={<>Mean ± sd over three seeds. results.md §7e.</>}
    />
  );
}
