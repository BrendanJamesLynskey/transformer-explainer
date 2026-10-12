/**
 * Server wrapper for the live trainer: hands it the recorded run and each
 * optimiser's best learning rate from results.json, so the client bundle
 * does not carry the whole results file.
 */
import results from "@/data/training/results.json";
import type { OptimizerKind } from "@/lib/training/optimizers";

import { LiveTrainer } from "./lazy";

const LABEL: Record<string, string> = {
  sgd: "SGD",
  momentum: "Momentum",
  adam: "Adam",
  adamw: "AdamW",
};

export function LiveTrainerPanel(): JSX.Element {
  const run = results.defaultRun;
  const best = new Map<string, { lr: number; final: number }>();
  for (const r of results.optimizerCompare) {
    const b = best.get(r.kind);
    if (!b || r.final < b.final) best.set(r.kind, r);
  }
  const choices = [
    {
      kind: "adamw" as OptimizerKind,
      lr: run.config.schedule.lr,
      label: `AdamW, lr ${run.config.schedule.lr} (the recorded run)`,
    },
    ...["sgd", "momentum", "adam"].map((k) => ({
      kind: k as OptimizerKind,
      lr: best.get(k)!.lr,
      label: `${LABEL[k]}, lr ${best.get(k)!.lr} (its best in §5)`,
    })),
  ];
  return (
    <LiveTrainer
      recorded={run.evals.map((e) => ({ step: e.step, val: e.val }))}
      choices={choices}
      prompt={run.samples.prompt}
    />
  );
}
