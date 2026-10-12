/**
 * Chapter 15's measured charts (results.md §11c, §15): what a dashboard
 * shows during the default run (loss, gradient norm, update-to-weight
 * ratio), the resume experiment, and goodput against the checkpoint
 * interval. Server-rendered (LineChart), no JavaScript.
 */
import results from "@/data/training/results.json";
import { OKABE_ITO } from "@/lib/viz/palette";

import { LineChart } from "./LineChart";

export function MonitorCharts(): JSX.Element {
  const lg = results.monitor.log;
  const pick = (ys: number[]) =>
    lg.steps.map((x, i) => ({ x, y: ys[i]! })).filter((_, i) => i % 2 === 0);
  return (
    <div data-testid="monitor-charts">
      <LineChart
        testId="monitor-loss"
        label={`Training loss and gradient norm during the default run: loss ${lg.loss[0]!.toFixed(2)} at step 0, ${lg.loss.at(-1)!.toFixed(2)} at the end`}
        series={[
          {
            name: "training loss (one batch)",
            colour: OKABE_ITO.vermillion,
            points: pick(lg.loss),
          },
          {
            name: "gradient norm before clipping",
            colour: OKABE_ITO.orange,
            dash: "6 3",
            points: pick(lg.gradNorm),
          },
        ]}
        xDomain={[0, 2000]}
        yDomain={[0, 4.5]}
        xTicks={[0, 500, 1000, 1500, 2000]}
        yTicks={[0, 1, 2, 3, 4]}
        xLabel="step"
        yLabel="loss (nats) · gradient norm"
        caption={
          <>
            The default run as a training dashboard sees it, every 20 steps. The
            gradient norm spikes as the model leaves chapter 1&rsquo;s plateau;
            clipping (at 1.0) acts whenever the dashed line is above 1.
            results.md §15.
          </>
        }
      />
      <LineChart
        testId="monitor-ratio"
        label={`Update-to-weight ratio during the default run: ${lg.updateRatio[0]!.toExponential(1)} at the start, ${lg.updateRatio.at(-1)!.toExponential(1)} at the end`}
        series={[
          {
            name: "all parameters",
            colour: OKABE_ITO.green,
            points: pick(lg.updateRatio),
          },
          {
            name: "block 1 FFN W1",
            colour: OKABE_ITO.blue,
            dash: "6 3",
            points: pick(lg.updateRatioW1),
          },
        ]}
        xDomain={[0, 2000]}
        yDomain={[1e-4, 1e-1]}
        yLog
        xTicks={[0, 500, 1000, 1500, 2000]}
        yTicks={[1e-4, 1e-3, 1e-2, 1e-1]}
        yFmt={(v) => v.toExponential(0).replace("e-", "e−")}
        xLabel="step"
        yLabel="|Δθ| / |θ| per step (log)"
        caption={
          <>
            How far each step moves the weights, relative to their size: up
            through the warmup, a jump as the model leaves the plateau, then
            down with the cosine decay. results.md §15.
          </>
        }
      />
    </div>
  );
}

export function ResumeChart(): JSX.Element {
  const r = results.resume;
  const S = [
    {
      key: "uninterrupted",
      name: "never stopped",
      colour: OKABE_ITO.blue,
      marker: "none" as const,
    },
    {
      key: "weights",
      name: "resumed with the weights only",
      colour: OKABE_ITO.vermillion,
      marker: "square" as const,
    },
    {
      key: "freshData",
      name: "resumed, data restarted",
      colour: OKABE_ITO.orange,
      marker: "triangle" as const,
    },
  ] as const;
  const from = r.stop - 200;
  return (
    <LineChart
      testId="resume-chart"
      label={`Validation loss around a resume at step ${r.stop}: never stopped ${r.uninterrupted.mean.toFixed(3)}, weights only ${r.weights.mean.toFixed(3)}, data restarted ${r.freshData.mean.toFixed(3)} at the end`}
      series={S.map((s) => {
        const x = r[s.key];
        return {
          name: s.name,
          colour: s.colour,
          marker: s.marker,
          dash: s.key === "uninterrupted" ? "" : undefined,
          points: x.evalSteps
            .map((t, i) => ({ x: t, y: x.meanEvals[i]! }))
            .filter((p) => p.x >= from),
        };
      })}
      markers={[[r.stop, "stop and resume"]]}
      xDomain={[from, 2000]}
      yDomain={[2.2, 3.0]}
      xTicks={[from, r.stop, 1400, 1700, 2000]}
      yTicks={[2.2, 2.4, 2.6, 2.8, 3.0]}
      xLabel="step"
      yLabel="validation loss (mean of 3 seeds)"
      caption={
        <>
          Resuming with the weights, AdamW&rsquo;s moments and step count, and
          the data position continues the run bit for bit (
          {r.bitIdentical ? "checked: identical" : "NOT identical"}), so it is
          the &ldquo;never stopped&rdquo; line. Finals: weights only{" "}
          {r.weights.mean.toFixed(3)} ± {r.weights.sd.toFixed(3)}; data
          restarted {r.freshData.mean.toFixed(3)} ± {r.freshData.sd.toFixed(3)};
          never stopped {r.uninterrupted.mean.toFixed(3)} ±{" "}
          {r.uninterrupted.sd.toFixed(3)}. results.md §15.
        </>
      }
    />
  );
}

export function GoodputChart(): JSX.Element {
  const r = results.systems.reliability;
  return (
    <LineChart
      testId="goodput-chart"
      label={`Goodput against checkpoint interval: best simulated at ${r.bestTau} s, Young's interval ${Math.round(r.young)} s`}
      series={[
        {
          name: `simulated (${r.seeds.length} seeds × ${r.llama.days} days)`,
          colour: OKABE_ITO.green,
          marker: "circle",
          points: r.rows.map((x) => ({ x: x.tau, y: x.simulated * 100 })),
        },
        {
          name: "first-order formula",
          colour: OKABE_ITO.blue,
          dash: "6 3",
          points: r.rows.map((x) => ({ x: x.tau, y: x.formula * 100 })),
        },
      ]}
      markers={[[r.young, "Young's τ*"]]}
      xDomain={[50, 6000]}
      yDomain={[60, 90]}
      xLog
      xTicks={[60, 240, 960, 3840]}
      yTicks={[60, 70, 80, 90]}
      xFmt={(v) =>
        v < 3600 ? `${Math.round(v / 60)} min` : `${(v / 3600).toFixed(1)} h`
      }
      xLabel="checkpoint every (log scale)"
      yLabel="goodput, %"
      caption={
        <>
          A failure every {r.llama.mtbfHours.toFixed(1)} h (Llama 3 405B&rsquo;s
          rate), a {Math.round(r.delta)} s checkpoint and a{" "}
          {Math.round(r.restart / 60)} min restart (illustrative). Checkpoint
          too often and the checkpoints eat the time; too rarely and failures
          do. Simulated best {Math.round(r.bestTau / 60)} min; Young&rsquo;s
          √(2δM) = {Math.round(r.young / 60)} min. results.md §11c.
        </>
      }
    />
  );
}
