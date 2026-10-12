/**
 * Chapter 6's hero: four learning-rate schedules side by side, with the
 * validation loss each one measured (results.md §6, the mean over three
 * seeds, every 50 steps). One state per validation point: each schedule's
 * learning rate at that step, computed by the trainer's own schedules.ts,
 * and its mean validation loss so far. scripts/training/anim_b.py:
 * schedule_states builds the same list in Python.
 */
import { lrAt, type Schedule } from "@/lib/training/schedules";

/** A schedule as results.json stores it (the Python reference's keys). */
export type PySchedule = {
  kind: "constant" | "cosine" | "wsd";
  lr: number;
  warmup?: number;
  min_ratio?: number;
  decay_frac?: number;
};

export type ScheduleRow = {
  name: string;
  spec: PySchedule;
  evalSteps: number[];
  meanEvals: (number | null)[];
};

export type ScheduleState = {
  i: number;
  step: number;
  lrs: number[];
  vals: (number | null)[];
};

/** results.json's schedule spec as schedules.ts's Schedule. */
export function toSchedule(s: PySchedule, total: number): Schedule {
  return {
    kind: s.kind,
    lr: s.lr,
    total,
    warmup: s.warmup,
    minRatio: s.min_ratio,
    decayFrac: s.decay_frac,
  };
}

export function scheduleStates(
  rows: ScheduleRow[],
  total = 2000,
): ScheduleState[] {
  return rows[0]!.evalSteps.map((step, i) => ({
    i,
    step,
    lrs: rows.map((r) =>
      lrAt(Math.min(step, total - 1), toSchedule(r.spec, total)),
    ),
    vals: rows.map((r) => r.meanEvals[i] ?? null),
  }));
}

export function scheduleCaption(s: ScheduleState, names: string[]): string {
  const parts = names.map((n, k) => {
    const v = s.vals[k];
    return `${n} lr ${s.lrs[k]!.toPrecision(2)}, loss ${v === null ? "–" : v!.toFixed(3)}`;
  });
  if (s.step === 0)
    return `Step 0: every run starts from Part 1's weights at loss ${s.vals[0]!.toFixed(3)}. ${names.length} schedules, peak learning rate ${s.lrs[0]}.`;
  return `Step ${s.step.toLocaleString("en-GB")}: ${parts.join(" · ")}.`;
}
