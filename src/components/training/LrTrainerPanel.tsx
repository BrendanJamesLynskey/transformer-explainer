/**
 * Chapter 6's live trainer: the default run at four peak learning rates
 * (the first is the recorded default). Each label quotes the sweep's
 * recorded seed-1 result (results.md §6a), which the browser's run with
 * the same settings reproduces.
 */
import results from "@/data/training/results.json";
import type { OptimizerKind } from "@/lib/training/optimizers";

import { LiveTrainer } from "./lazy";

export function LrTrainerPanel(): JSX.Element {
  const run = results.defaultRun;
  const at = (lr: number) => results.lrSweep.rows.find((r) => r.lr === lr)!;
  const label = (lr: number) => {
    const f = at(lr).finals[0];
    return `AdamW lr ${lr} · recorded ${f === null || f === undefined ? "diverged" : f.toFixed(3)}`;
  };
  const choices = [0.01, 0.001, 0.1, 1.0].map((lr) => ({
    kind: "adamw" as OptimizerKind,
    lr,
    label: lr === 0.01 ? `${label(lr)} (default)` : label(lr),
  }));
  return (
    <LiveTrainer
      recorded={run.evals.map((e) => ({ step: e.step, val: e.val }))}
      choices={choices}
      prompt={run.samples.prompt}
    />
  );
}
