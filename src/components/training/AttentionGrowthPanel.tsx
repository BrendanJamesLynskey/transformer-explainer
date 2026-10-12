/**
 * Server wrapper for chapter 8's animation: the snapshot run's validation
 * losses and the recorded largest-score curves (baseline and QK-norm at
 * peak learning rate 0.1, mean of three seeds; results.md §8b).
 */
import type { ReactNode } from "react";

import results from "@/data/training/results.json";

import { AttentionGrowthAnimation } from "./lazy";

export function AttentionGrowthPanel({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const curves = results.attnGrowth.rows
    .filter(
      (r) => r.lr === 0.1 && (r.name === "baseline" || r.name === "QK-norm"),
    )
    .map((r) => ({ name: `${r.name} (lr 0.1)`, stats: r.meanStats }));
  const evals = results.attnRun.evals.map((e) => ({
    step: e.step,
    val: e.val,
  }));
  return (
    <AttentionGrowthAnimation evals={evals} curves={curves}>
      {children}
    </AttentionGrowthAnimation>
  );
}
