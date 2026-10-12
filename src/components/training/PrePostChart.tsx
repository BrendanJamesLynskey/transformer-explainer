/**
 * Chapter 7: pre-LN vs post-LN, 6 blocks, with and without warmup
 * (results.md §7b): mean validation curves over three seeds.
 */
import results from "@/data/training/results.json";
import { OKABE_ITO } from "@/lib/viz/palette";

import { LineChart, type Marker } from "./LineChart";

const STYLE: Record<string, { colour: string; dash: string; marker: Marker }> =
  {
    "pre-0": { colour: OKABE_ITO.blue, dash: "", marker: "circle" },
    "pre-200": { colour: OKABE_ITO.sky, dash: "6 3", marker: "square" },
    "post-0": { colour: OKABE_ITO.vermillion, dash: "", marker: "triangle" },
    "post-200": { colour: OKABE_ITO.orange, dash: "6 3", marker: "diamond" },
  };

export function PrePostChart(): JSX.Element {
  const x = results.prePost;
  return (
    <LineChart
      testId="pre-post-chart"
      label={`Validation loss over ${x.steps} steps for a ${x.depth}-block model: ${x.rows
        .map(
          (r) =>
            `${r.norm}-LN, warmup ${r.warmup}: ${r.mean === null ? "diverged" : r.mean.toFixed(3)}`,
        )
        .join("; ")}`}
      series={x.rows.map((r) => ({
        name: `${r.norm}-LN, warmup ${r.warmup}`,
        ...STYLE[`${r.norm}-${r.warmup}`]!,
        points: r.evalSteps.map((t, i) => ({
          x: t,
          y: r.meanEvals[i] ?? null,
        })),
      }))}
      xDomain={[0, x.steps]}
      yDomain={[2.0, 4.3]}
      xTicks={[0, 500, 1000, 1500]}
      yTicks={[2.0, 2.5, 3.0, 3.5, 4.0]}
      xLabel="step"
      yLabel="validation loss (mean of 3 seeds)"
      caption={<>Peak learning rate {x.lr}, cosine decay. results.md §7b.</>}
    />
  );
}
