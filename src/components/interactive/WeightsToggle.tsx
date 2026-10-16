"use client";

/**
 * The "Weights" control every Decoder widget carries: Training's trained
 * weights (the default) or the seeded random initialiser they started
 * from. Flipping it re-runs the widget's model on the other weights, so
 * the reader can see what training changed: sharp attention instead of
 * near-uniform rows, a logit lens that moves, samples that look like words.
 */
import { useState } from "react";

import type { WeightKind } from "@/lib/transformer/trained";

/** The toggle's state, shared shape for every widget. */
export function useWeightKind(
  initial: WeightKind = "trained",
): [WeightKind, (k: WeightKind) => void] {
  return useState<WeightKind>(initial);
}

export function WeightsToggle({
  value,
  onChange,
  label,
  disabled = false,
}: {
  value: WeightKind;
  onChange: (k: WeightKind) => void;
  /** Accessible name, e.g. "Attention animation weights". */
  label: string;
  /** Shapes that don't fit the trained model: random init only. */
  disabled?: boolean;
}): JSX.Element {
  return (
    <label className="flex min-w-0 flex-col gap-1 text-sm">
      <span className="font-medium text-neutral-700 dark:text-neutral-300">
        Weights
      </span>
      <select
        value={disabled ? "random" : value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value as WeightKind)}
        aria-label={label}
        data-weights={disabled ? "random" : value}
        className="focus-ring h-11 rounded border border-neutral-300 bg-white px-2 text-sm disabled:opacity-60 dark:border-neutral-700 dark:bg-neutral-950"
      >
        <option value="trained">trained</option>
        <option value="random">random init</option>
      </select>
    </label>
  );
}
