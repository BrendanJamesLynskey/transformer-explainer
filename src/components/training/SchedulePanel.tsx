/**
 * Server wrapper for chapter 6's schedule animation: hands it the measured
 * schedule runs from results.json (results.md §6b), so the client bundle
 * does not carry the whole results file.
 */
import type { ReactNode } from "react";

import results from "@/data/training/results.json";
import type { ScheduleRow } from "@/lib/anim/training/schedule-steps";

import { ScheduleAnimation } from "./lazy";

export function SchedulePanel({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const rows: ScheduleRow[] = results.schedules.rows.map((r) => ({
    name: r.name,
    spec: r.spec as ScheduleRow["spec"],
    evalSteps: r.evalSteps,
    meanEvals: r.meanEvals,
  }));
  return <ScheduleAnimation rows={rows}>{children}</ScheduleAnimation>;
}
