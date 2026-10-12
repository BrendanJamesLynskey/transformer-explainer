/**
 * Chapter 15's hero: a week of a large training job, failure by failure
 * (systems.ts `simulateGoodput`): work in intervals of τ, a checkpoint after
 * each, failures at random (exponential) times at Llama 3 405B's measured
 * rate, each throwing away the work since the last checkpoint and costing a
 * restart. A state is one segment of the timeline with the running totals.
 * scripts/training/anim_c.py: goodput_fixture runs the same simulation
 * (tests: anim-frames-c.test.ts).
 */
import type { Segment } from "@/lib/training/systems";

export type GoodputState = {
  index: number;
  segment: Segment;
  useful: number;
  lost: number;
  checkpointing: number;
  restarting: number;
  failures: number;
};

export function goodputStates(
  segments: Segment[],
  horizon: number,
): GoodputState[] {
  let useful = 0;
  let lost = 0;
  let ck = 0;
  let rs = 0;
  let failures = 0;
  return segments.map((g, index) => {
    const len = Math.min(g.end, horizon) - g.start;
    if (g.kind === "work") useful += len;
    else if (g.kind === "lost") {
      lost += len;
      failures++;
    } else if (g.kind === "checkpoint") ck += len;
    else rs += len;
    return {
      index,
      segment: g,
      useful,
      lost,
      checkpointing: ck,
      restarting: rs,
      failures,
    };
  });
}

/** The equation term each segment highlights (training itself: none). */
export const GOODPUT_HL: Record<Segment["kind"], string> = {
  work: "",
  checkpoint: "ckpt",
  lost: "lost",
  restart: "lost",
};

const hm = (s: number) => {
  const h = Math.floor(s / 3600);
  const m = Math.round((s - h * 3600) / 60);
  return m === 60
    ? `${h + 1} h 00 min`
    : `${h} h ${String(m).padStart(2, "0")} min`;
};
const day = (s: number) => `day ${Math.floor(s / 86400) + 1}, ${hm(s % 86400)}`;

export function goodputCaption(s: GoodputState): string {
  const g = s.segment;
  const gp = (100 * s.useful) / g.end;
  const what =
    g.kind === "work"
      ? `trained for ${hm(g.end - g.start)}`
      : g.kind === "checkpoint"
        ? `saved a checkpoint (${Math.round(g.end - g.start)} s)`
        : g.kind === "lost"
          ? `a failure: ${hm(g.end - g.start)} of training since the last checkpoint is lost`
          : `restarted from the last checkpoint (${Math.round((g.end - g.start) / 60)} min)`;
  return `${day(g.start)}: ${what}. So far ${s.failures} failure${s.failures === 1 ? "" : "s"}; useful work ${gp.toFixed(1)}% of the time so far.`;
}
