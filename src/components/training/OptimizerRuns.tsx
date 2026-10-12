/**
 * Chapter 5's measured comparison on the real model (results.md §5): the
 * validation loss of each optimiser at its best learning rate, every 100
 * steps, from the recorded runs. Server-rendered (LineChart, with its
 * phone drawing so the labels stay readable at 390 px): no JavaScript.
 */
import results from "@/data/training/results.json";
import { OKABE_ITO } from "@/lib/viz/palette";

import { LineChart } from "./LineChart";

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
  const steps = best[0]!.evalSteps[best[0]!.evalSteps.length - 1]!;
  return (
    <LineChart
      testId="optimizer-runs"
      label={`Validation loss against step for four optimisers at their best learning rates: ${best
        .map(
          (b) => `${LABEL[b.kind as keyof typeof LABEL]} ${b.final.toFixed(3)}`,
        )
        .join(", ")} after ${steps} steps`}
      series={best.map((b) => {
        const k = b.kind as keyof typeof COLOUR;
        return {
          name: `${LABEL[k]} (lr ${b.lr}): ${b.final.toFixed(3)} ± ${b.sd.toFixed(3)}`,
          colour: COLOUR[k],
          dash: DASH[k],
          points: b.meanEvals.map((v, i) => ({ x: b.evalSteps[i]!, y: v })),
        };
      })}
      xDomain={[0, steps]}
      yDomain={[2.0, 4.3]}
      xTicks={[0, 500, 1000, 1500]}
      yTicks={[2.0, 2.5, 3.0, 3.5, 4.0]}
      xLabel="step (validation loss every 100 steps)"
      yLabel="validation loss"
      caption={
        <>
          Mean over three data seeds (± their standard deviation) at each
          optimiser&rsquo;s best of three constant learning rates,{" "}
          {steps.toLocaleString("en-GB")} steps; results.md §5.
        </>
      }
    />
  );
}
