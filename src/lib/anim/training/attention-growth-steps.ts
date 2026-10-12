/**
 * Chapter 8's hero: the run at peak learning rate 0.1 (results.md §8),
 * replayed from weight snapshots (public/training/attention-snapshots.json).
 * At each snapshot the site's own forward pass (model-grad.ts) runs on
 * "alice was" and the state keeps block 2, head 1's attention weights, the
 * largest |attention score| and the mean attention entropy over every
 * block and head (rows from position 1; row 0 can only look at itself).
 * scripts/training/anim_b.py:attention_states builds the same list.
 */
import { encodeChars } from "@/lib/training/data";
import { forwardCached } from "@/lib/training/model-grad";
import type { ParamMap } from "@/lib/training/params";
import { PART1_MODEL } from "@/lib/training/trainer";

export const ATTN_TEXT = "alice was";
export const ATTN_BLOCK = 1;
export const ATTN_HEAD = 0;

export type Snapshots = {
  steps: number[];
  params: Record<string, ParamMap>;
};

export type AttentionGrowthState = {
  step: number;
  weights: number[][];
  maxScore: number;
  entropy: number;
  val: number | null;
};

/** Largest |score| (causal entries) and mean row entropy, over blocks and heads. */
export function attentionStats(
  blocks: { attn: { scores: number[][][]; weights: number[][][] } }[],
): { maxScore: number; entropy: number } {
  let mx = 0;
  let ent = 0;
  let n = 0;
  for (const b of blocks) {
    b.attn.scores.forEach((sc, h) => {
      const P = b.attn.weights[h]!;
      sc.forEach((row, i) => {
        for (let j = 0; j <= i; j++) mx = Math.max(mx, Math.abs(row[j]!));
        if (i === 0) return;
        let hrow = 0;
        for (let j = 0; j < P[i]!.length; j++) {
          const p = P[i]![j]!;
          if (p > 0) hrow -= p * Math.log(p);
        }
        ent += hrow;
        n++;
      });
    });
  }
  return { maxScore: mx, entropy: ent / n };
}

export function attentionGrowthStates(
  snap: Snapshots,
  evals: { step: number; val: number }[],
): AttentionGrowthState[] {
  const ids = encodeChars(ATTN_TEXT).slice(0, -1);
  const byStep = new Map(evals.map((e) => [e.step, e.val]));
  return snap.steps.map((step) => {
    const c = forwardCached(snap.params[String(step)]!, PART1_MODEL, ids);
    const st = attentionStats(c.blocks);
    return {
      step,
      weights: c.blocks[ATTN_BLOCK]!.attn.weights[ATTN_HEAD]!.map((r) =>
        r.slice(),
      ),
      maxScore: st.maxScore,
      entropy: st.entropy,
      val: byStep.get(step) ?? null,
    };
  });
}

export function attentionGrowthCaption(s: AttentionGrowthState): string {
  const loss = s.val === null ? "" : `, validation loss ${s.val.toFixed(3)}`;
  return `Step ${s.step.toLocaleString("en-GB")}: largest attention score ${s.maxScore < 100 ? s.maxScore.toFixed(1) : Math.round(s.maxScore).toLocaleString("en-GB")}, mean attention entropy ${s.entropy.toFixed(2)} nats${loss}.`;
}
