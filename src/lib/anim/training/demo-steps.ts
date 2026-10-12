/**
 * Chapter 13's hero: two DeMo syncs of four workers from the trained model
 * (slowlinks.ts `demoSync`), shown for the first chunk of one tensor: each
 * worker's gradient and momentum, the chunk's DCT, the top-k coefficients it
 * sends and the residual it keeps (its error feedback), the coefficients
 * gathered from everyone, their average and its inverse DCT, the sign the
 * update uses. In the second sync the momentum already holds the first
 * sync's residual. scripts/training/slowlinks.py: sync_states records the
 * same (tests: slowlinks.test.ts, anim-frames-c.test.ts).
 */
import { sig } from "@/lib/anim/format";
import type { DemoRecord } from "@/lib/training/slowlinks";

export const DEMO_STAGES = [
  "grad",
  "momentum",
  "dct",
  "topk",
  "residual",
  "gather",
  "mean",
  "inverse",
  "sign",
] as const;
export type DemoStage = (typeof DEMO_STAGES)[number];

export type DemoState = { sync: number; stage: DemoStage };

export function demoStates(records: DemoRecord[]): DemoState[] {
  const out: DemoState[] = [];
  records.forEach((_, sync) => {
    for (const stage of DEMO_STAGES) out.push({ sync, stage });
  });
  return out;
}

/** The equation term each stage highlights. */
export const DEMO_HL: Record<DemoStage, string> = {
  grad: "grad",
  momentum: "ef",
  dct: "dct",
  topk: "topk",
  residual: "ef",
  gather: "agg",
  mean: "agg",
  inverse: "dct",
  sign: "sign",
};

const W = (r: DemoRecord) => r.workers.length;

export function demoCaption(s: DemoState, records: DemoRecord[]): string {
  const r = records[s.sync]!;
  const w0 = r.workers[0]!;
  const n = r.chunk;
  const k = r.k;
  const sentCount = new Set(r.workers.flatMap((w) => w.idx)).size;
  const tag = `Sync ${s.sync + 1}`;
  switch (s.stage) {
    case "grad":
      return `${tag}: each of ${W(r)} workers computes its own gradient on its own windows; the first ${n} values of ${r.name} (worker 1's first: ${sig(w0.grad[0]!)}).`;
    case "momentum":
      return s.sync === 0
        ? `${tag}: each worker folds the gradient into its own momentum, M ← βM + g; empty before this step, so M = g.`
        : `${tag}: M ← βM + g. The momentum still holds what sync 1 did not send (β = 0.999), so it is not just this gradient: error feedback.`;
    case "dct":
      return `${tag}: a DCT turns each chunk of ${n} momentum values into ${n} frequency coefficients (orthonormal, so nothing is lost). Worker 1's largest: ${sig(Math.max(...w0.coef.map(Math.abs)))}.`;
    case "topk":
      return `${tag}: each worker picks its ${k} largest coefficients of the ${n} (1 in 16), different ones per worker: worker 1 sends positions ${w0.idx.map((i) => i + 1).join(", ")}.`;
    case "residual":
      return `${tag}: what was not sent stays in each worker's momentum (M ← M − IDCT(sent)): delayed, not lost. Worker 1 keeps ${sig(norm(w0.residual))} of ${sig(norm(w0.momentum))} (by size, ‖·‖₂).`;
    case "gather":
      return `${tag}: an all-gather: every worker receives every other worker's ${k} values and positions. Different workers chose different positions, so they cannot simply be added on the way (an all-reduce): ${sentCount} positions of ${n} were sent by someone.`;
    case "mean":
      return `${tag}: average each position over the workers that sent it; positions nobody sent stay 0.`;
    case "inverse":
      return `${tag}: the inverse DCT turns the averaged coefficients back into ${n} momentum values, the same on every worker.`;
    case "sign":
      return `${tag}: DeMo steps by the sign: θ ← θ − η (sign(M*) + λθ). ${r.aggregate.filter((v) => v > 0).length} of ${n} values move down, ${r.aggregate.filter((v) => v < 0).length} up, each by the learning rate.`;
  }
}

function norm(v: number[]): number {
  let s = 0;
  for (const x of v) s += x * x;
  return Math.sqrt(s);
}
