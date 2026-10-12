/**
 * Chapter 1's hero: three real training steps of Part 1's model on a batch
 * of 4 windows, as a list of states. Each step:
 *
 *   batch     4 windows of 8 characters; the targets are the next characters
 *   forward   the model's probability for each right answer, p(y)
 *   loss      −log p(y) at all 32 positions; their mean is the loss
 *   backward  the gradient reaches each group of weights, last layer first
 *             (one state per group, with ‖g‖ for that group)
 *   update    AdamW moves every weight; ‖Δθ‖/‖θ‖ per group, and the loss on
 *             the same batch after the step (lower)
 *
 * The numbers come from the trainer itself (model-grad.ts, optimizers.ts);
 * scripts/training/anim.py:hero_states builds the same list in Python and
 * the frame tests compare them.
 */
import { sampleBatch, splitIds } from "@/lib/training/data";
import { crossEntropy } from "@/lib/training/loss";
import {
  forwardCached,
  lossAndGrads,
  lossOnly,
} from "@/lib/training/model-grad";
import {
  clipByGlobalNorm,
  createOptimizer,
  optimizerStep,
} from "@/lib/training/optimizers";
import {
  cloneParams,
  fromWeights,
  sumSquares,
  type ParamMap,
} from "@/lib/training/params";
import { PART1_MODEL } from "@/lib/training/trainer";
import { initModelWeights } from "@/lib/transformer/init";
import { mulberry32 } from "@/lib/transformer/random";

import { sig } from "../format";

/** The hero's run: AdamW at a constant 0.01 (anim.py:HERO). */
export const HERO = {
  batch: 4,
  steps: 3,
  lr: 0.01,
  beta2: 0.99,
  weightDecay: 0.01,
  clip: 1.0,
  dataSeed: 1,
} as const;

/** Parameter groups, in the order the gradient reaches them going back. */
export const GROUPS: { label: string; names: string[] }[] = [
  { label: "final LayerNorm", names: ["ln_final.gamma", "ln_final.beta"] },
  ...[1, 0].flatMap((b) => [
    {
      label: `block ${b} · FFN`,
      names: [
        `block_${b}.ln2.gamma`,
        `block_${b}.ln2.beta`,
        `block_${b}.ffn.W1`,
        `block_${b}.ffn.b1`,
        `block_${b}.ffn.W2`,
        `block_${b}.ffn.b2`,
      ],
    },
    {
      label: `block ${b} · attention`,
      names: [
        `block_${b}.ln1.gamma`,
        `block_${b}.ln1.beta`,
        `block_${b}.attn.W_q`,
        `block_${b}.attn.W_k`,
        `block_${b}.attn.W_v`,
        `block_${b}.attn.W_o`,
      ],
    },
  ]),
  { label: "embedding E", names: ["tok_emb"] },
];

export type HeroPhase = "batch" | "forward" | "loss" | "backward" | "update";

export type HeroState = {
  step: number;
  phase: HeroPhase;
  /** the group the gradient just reached (backward), else −1 */
  group: number;
  inputs: number[][];
  targets: number[][];
  lr: number;
  /** p(right answer) per position (from forward on) */
  pTarget: number[][] | null;
  /** −log p(y) per position (from loss on) */
  perPosition: number[][] | null;
  loss: number | null;
  /** ‖g‖ per group; null until the gradient has reached it */
  gradNorms: (number | null)[];
  /** global ‖g‖ before clipping, and the clip scale (update) */
  gradNorm: number | null;
  clipScale: number | null;
  /** ‖Δθ‖ / ‖θ‖ per group (update) */
  updateRatios: number[] | null;
  /** the same batch's loss after the update */
  lossAfter: number | null;
};

function groupNorm(t: ParamMap, names: string[]): number {
  let s = 0;
  for (const n of names) s += sumSquares(t[n]!);
  return Math.sqrt(s);
}

function diff(a: ParamMap, b: ParamMap): ParamMap {
  const out: ParamMap = {};
  for (const k of Object.keys(a)) {
    const x = a[k]!;
    const y = b[k]!;
    out[k] = Array.isArray(x[0])
      ? (x as number[][]).map((r, i) =>
          r.map((v, j) => v - (y as number[][])[i]![j]!),
        )
      : (x as number[]).map((v, j) => v - (y as number[])[j]!);
  }
  return out;
}

/** The hero's states, from the corpus ids. */
export function heroStates(corpusIds: number[]): HeroState[] {
  const { train } = splitIds(corpusIds);
  const S = PART1_MODEL.seq_len;
  const params = fromWeights(initModelWeights(PART1_MODEL));
  const opt = createOptimizer(params, {
    kind: "adamw",
    beta2: HERO.beta2,
    weightDecay: HERO.weightDecay,
  });
  const rng = mulberry32(HERO.dataSeed);
  const G = GROUPS.length;
  const states: HeroState[] = [];
  for (let t = 0; t < HERO.steps; t++) {
    const b = sampleBatch(train, HERO.batch, S, rng);
    const pTarget = b.inputs.map((ids, i) => {
      const { probs } = crossEntropy(
        forwardCached(params, PART1_MODEL, ids).logits,
        b.targets[i]!,
      );
      return probs.map((p, s) => p[b.targets[i]![s]!]!);
    });
    const { loss, grads, perPosition } = lossAndGrads(
      params,
      PART1_MODEL,
      b.inputs,
      b.targets,
    );
    const norms = GROUPS.map((g) => groupNorm(grads, g.names));
    const clip = clipByGlobalNorm(grads, HERO.clip);
    const before = cloneParams(params);
    optimizerStep(opt, params, grads, HERO.lr);
    const delta = diff(params, before);
    const ratios = GROUPS.map(
      (g) => groupNorm(delta, g.names) / groupNorm(before, g.names),
    );
    const after = lossOnly(params, PART1_MODEL, b.inputs, b.targets);
    const base: HeroState = {
      step: t,
      phase: "batch",
      group: -1,
      inputs: b.inputs,
      targets: b.targets,
      lr: HERO.lr,
      pTarget: null,
      perPosition: null,
      loss: null,
      gradNorms: new Array<number | null>(G).fill(null),
      gradNorm: null,
      clipScale: null,
      updateRatios: null,
      lossAfter: null,
    };
    states.push(base);
    const fwd = { ...base, pTarget };
    states.push({ ...fwd, phase: "forward" });
    const ls = { ...fwd, perPosition, loss };
    states.push({ ...ls, phase: "loss" });
    for (let g = 0; g < G; g++) {
      states.push({
        ...ls,
        phase: "backward",
        group: g,
        gradNorms: norms.map((n, k) => (k <= g ? n : null)),
      });
    }
    states.push({
      ...ls,
      phase: "update",
      gradNorms: norms,
      gradNorm: clip.norm,
      clipScale: clip.scale,
      updateRatios: ratios,
      lossAfter: after,
    });
  }
  return states;
}

/** The KaTeX term each phase highlights. */
export const HERO_HL: Record<HeroPhase, string> = {
  batch: "batch",
  forward: "fwd",
  loss: "loss",
  backward: "grad",
  update: "update",
};

/** The live caption for one state. */
export function heroCaption(s: HeroState): string {
  const n = s.inputs.length * (s.inputs[0]?.length ?? 0);
  const head = `Step ${s.step + 1}: `;
  switch (s.phase) {
    case "batch":
      return `${head}draw ${s.inputs.length} windows of ${s.inputs[0]!.length} characters; each position's target is the character after it (${n} predictions).`;
    case "forward": {
      let m = 0;
      for (const r of s.pTarget!) for (const p of r) m += p;
      return `${head}forward pass: the model gives the right next character ${sig((m / n) * 100)}% on average (a uniform guess gives 1.56%).`;
    }
    case "loss":
      return `${head}loss = mean of −log p(right answer) over ${n} positions = ${sig(s.loss!, 4)} (ln 64 = 4.16).`;
    case "backward":
      return `${head}backward: the gradient reaches ${GROUPS[s.group]!.label}, ‖g‖ = ${sig(s.gradNorms[s.group]!)}.`;
    case "update":
      return `${head}AdamW step at lr ${s.lr}: ‖g‖ = ${sig(s.gradNorm!)}${s.clipScale! < 1 ? ` (clipped ×${sig(s.clipScale!)})` : ""}; the same batch's loss falls from ${sig(s.loss!, 4)} to ${sig(s.lossAfter!, 4)}.`;
  }
}
