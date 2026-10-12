/**
 * Chapter 11's hero: a ring all-reduce of four workers' real gradients
 * (the trained model's final LayerNorm gain, each worker's own windows),
 * run chunk by chunk by systems.ts `ringStates`: three reduce-scatter steps
 * leave each worker holding the full sum of one chunk, three all-gather
 * steps hand those sums round the ring. scripts/training/anim_c.py:
 * ring_fixture builds the same states (tests: anim-frames-c.test.ts).
 */
import { sig } from "@/lib/anim/format";
import { ringStates, type RingState } from "@/lib/training/systems";

export { ringStates };
export type { RingState };

/** Which chunks of each worker's buffer are complete sums now. */
export function completeChunks(s: RingState, n: number): boolean[][] {
  // after reduce-scatter step k, rank r has summed (k + 1) contributions
  // into chunk (r - k) mod n ... (r + 1) mod n after n - 1 steps; after
  // all-gather step k it also holds k + 1 finished chunks from upstream
  const out = Array.from({ length: n }, () =>
    new Array<boolean>(n).fill(false),
  );
  if (s.phase === "start") return out;
  for (let r = 0; r < n; r++) {
    if (s.phase === "reduce-scatter") {
      if (s.step === n - 1) out[r]![(r + 1) % n] = true;
    } else {
      const k = s.step - n; // 0-based all-gather step
      for (let j = 0; j <= k + 1; j++)
        out[r]![(((r + 1 - j) % n) + n) % n] = true;
    }
  }
  return out;
}

/** Bytes (as values) each worker has sent so far: one chunk per step. */
export function valuesSent(s: RingState, size: number, n: number): number {
  return (s.phase === "start" ? 0 : s.step) * (size / n);
}

export function ringCaption(s: RingState, n: number, size: number): string {
  if (s.phase === "start")
    return `Start: each of ${n} workers holds its own gradient (${size} values in ${n} chunks of ${size / n}). Nobody has the sum yet.`;
  const sends = s.sends
    .map((x) => `${x.from + 1}→${x.to + 1} chunk ${x.chunk + 1}`)
    .join(", ");
  const sent = valuesSent(s, size, n);
  if (s.phase === "reduce-scatter") {
    const last = s.step === n - 1;
    return `Reduce-scatter step ${s.step} of ${n - 1}: ${sends}; each receiver adds the chunk to its own. ${last ? "Each worker now holds the full sum of one chunk." : `Partial sums of ${s.step + 1} gradients.`} Sent so far: ${sent} values per worker.`;
  }
  const k = s.step - n + 1;
  const done = s.bufs[0]![0]!;
  return `All-gather step ${k} of ${n - 1}: ${sends}; each receiver copies the finished chunk. ${k === n - 1 ? `Every worker holds the whole sum (first value ${sig(done)}): ${sent} values sent each, 2(n−1)/n of the gradient.` : `Sent so far: ${sent} values per worker.`}`;
}
