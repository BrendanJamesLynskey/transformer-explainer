"use client";

/** The "which weights?" selector shared by chapters 3 and 4. */
export type ModelKind = "trained" | "init";

export function ModelChoice({
  value,
  onChange,
  label,
}: {
  value: ModelKind;
  onChange: (v: ModelKind) => void;
  label: string;
}): JSX.Element {
  return (
    <label className="flex min-w-0 flex-col gap-1 text-sm">
      <span className="font-medium text-neutral-700 dark:text-neutral-300">
        Weights
      </span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as ModelKind)}
        aria-label={label}
        className="focus-ring h-11 rounded border border-neutral-300 bg-white px-2 text-sm dark:border-neutral-700 dark:bg-neutral-950"
      >
        <option value="trained">after the default run (2,000 steps)</option>
        <option value="init">at initialisation (Part 1&apos;s)</option>
      </select>
    </label>
  );
}
