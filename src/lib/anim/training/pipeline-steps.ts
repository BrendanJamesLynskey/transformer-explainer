/**
 * Chapter 12's hero: one training step of a pipeline, from the
 * discrete-event simulation in systems.ts (`pipeline`): GPipe, 1F1B or
 * interleaved 1F1B on p devices with m micro-batches. A state is a moment
 * in time (every time a work item starts or ends): which item each device
 * is running, how many are done, and the idle time so far (the bubble).
 * scripts/training/anim_c.py: pipeline_fixture holds the same schedules
 * (tests: anim-frames-c.test.ts).
 */
import type { PipeItem, PipeResult } from "@/lib/training/systems";

export type PipeState = {
  t: number;
  /** per device: the item it runs during [t, next t), or null (idle) */
  running: (PipeItem | null)[];
  done: number;
  /** idle device-time before t, over all devices */
  idle: number;
};

/** The moments at which anything starts or ends, in order. */
export function pipeTimes(r: PipeResult): number[] {
  const set = new Set<number>();
  for (const it of r.items) {
    set.add(it.start);
    set.add(it.end);
  }
  return [...set].sort((a, b) => a - b);
}

export function pipelineStates(r: PipeResult): PipeState[] {
  const times = pipeTimes(r);
  return times.map((t) => {
    const running: (PipeItem | null)[] = new Array<PipeItem | null>(r.p).fill(
      null,
    );
    let done = 0;
    let busy = 0;
    for (const it of r.items) {
      if (it.start <= t && t < it.end) running[it.device] = it;
      if (it.end <= t) done++;
      busy += Math.max(0, Math.min(it.end, t) - it.start);
    }
    return { t, running, done, idle: r.p * t - busy };
  });
}

const label = (it: PipeItem, v: number) =>
  `${it.op === "F" ? "forward" : "backward"} of micro-batch ${it.mb + 1}${v > 1 ? ` (chunk ${it.chunk + 1})` : ""}`;

export function pipelineCaption(s: PipeState, r: PipeResult): string {
  const total = r.items.length;
  if (s.done === total)
    return `t = ${fmt(s.t)}: all ${total} passes done. The step took ${fmt(r.makespan)} time units against ${fmt(r.ideal)} of work per device: bubble ${(100 * r.bubble).toFixed(1)}% (the formula (p − 1)/(${r.v > 1 ? "v·" : ""}m) gives ${(100 * r.closedForm).toFixed(1)}%).`;
  const busy = s.running
    .map((it, d) => (it ? `device ${d + 1} ${label(it, r.v)}` : null))
    .filter((x) => x !== null);
  const idle = s.running
    .map((it, d) => (it ? null : d + 1))
    .filter((x) => x !== null);
  return `t = ${fmt(s.t)}: ${busy.join("; ")}${idle.length ? `; idle: device${idle.length > 1 ? "s" : ""} ${idle.join(", ")}` : ""}. ${s.done} of ${total} passes done; idle so far ${fmt(s.idle)} device-time units.`;
}

function fmt(x: number): string {
  return Number(x.toFixed(3)).toString();
}
