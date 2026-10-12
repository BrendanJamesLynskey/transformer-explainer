/**
 * Chapter 3: next-token cross-entropy on one window ("alice was": 8 inputs,
 * 8 targets), position by position:
 *
 *   logits   the model's 64 scores for the next character
 *   softmax  turned into probabilities
 *   target   the character that actually came next, and its probability
 *   loss     −log p(target) at this position
 *   grad     d loss / d logits = (p − onehot(target)) / 8: every logit is
 *            pushed down by its probability, the right one up by 1 − p
 *
 * and finally the mean. Run on Part 1's model at initialisation or after
 * the default training run. scripts/training/anim.py:objective_states
 * builds the same list.
 */
import { crossEntropy, crossEntropyGrad } from "@/lib/training/loss";
import { forwardCached } from "@/lib/training/model-grad";
import type { ParamMap } from "@/lib/training/params";
import { PART1_MODEL } from "@/lib/training/trainer";
import { encodeChars } from "@/lib/training/data";
import { ALPHABET } from "@/lib/transformer/tokenizer";

import { pct, showChar, sig } from "../format";

export const WINDOW_TEXT = "alice was";

export type ObjectivePhase =
  | "logits"
  | "softmax"
  | "target"
  | "loss"
  | "grad"
  | "mean";

export type ObjectiveState = {
  phase: ObjectivePhase;
  pos: number;
  inputs: number[];
  targets: number[];
  target: number;
  logits: number[];
  probs: number[] | null;
  pTarget: number | null;
  loss: number | null;
  grad: number[] | null;
  losses: (number | null)[];
  mean: number | null;
};

/** The window's input and target ids. */
export function windowIds(text = WINDOW_TEXT): {
  inputs: number[];
  targets: number[];
} {
  const ids = encodeChars(text);
  return { inputs: ids.slice(0, -1), targets: ids.slice(1) };
}

export function objectiveStates(params: ParamMap): ObjectiveState[] {
  const { inputs, targets } = windowIds();
  const { logits } = forwardCached(params, PART1_MODEL, inputs);
  const { perPosition, probs } = crossEntropy(logits, targets);
  const grads = crossEntropyGrad(probs, targets, targets.length);
  const S = inputs.length;
  const losses: (number | null)[] = new Array<number | null>(S).fill(null);
  const out: ObjectiveState[] = [];
  for (let i = 0; i < S; i++) {
    const p = probs[i]!;
    const tgt = targets[i]!;
    const base = {
      pos: i,
      inputs,
      targets,
      target: tgt,
      logits: logits[i]!,
      probs: null,
      pTarget: null,
      loss: null,
      grad: null,
      mean: null,
    };
    out.push({ ...base, phase: "logits", losses: losses.slice() });
    out.push({ ...base, phase: "softmax", probs: p, losses: losses.slice() });
    out.push({
      ...base,
      phase: "target",
      probs: p,
      pTarget: p[tgt]!,
      losses: losses.slice(),
    });
    losses[i] = perPosition[i]!;
    const l = { ...base, probs: p, pTarget: p[tgt]!, loss: perPosition[i]! };
    out.push({ ...l, phase: "loss", losses: losses.slice() });
    out.push({ ...l, phase: "grad", grad: grads[i]!, losses: losses.slice() });
  }
  let sum = 0;
  for (const l of perPosition) sum += l;
  out.push({ ...out[out.length - 1]!, phase: "mean", mean: sum / S });
  return out;
}

export const OBJECTIVE_HL: Record<ObjectivePhase, string> = {
  logits: "z",
  softmax: "p",
  target: "y",
  loss: "loss",
  grad: "grad",
  mean: "mean",
};

export function objectiveCaption(s: ObjectiveState): string {
  const c = (id: number) => `'${showChar(ALPHABET[id]!)}'`;
  const at = `Position ${s.pos} (after ${c(s.inputs[s.pos]!)})`;
  switch (s.phase) {
    case "logits": {
      let best = 0;
      s.logits.forEach((v, i) => {
        if (v > s.logits[best]!) best = i;
      });
      return `${at}: the model scores all 64 characters; the highest logit is ${c(best)} (${sig(s.logits[best]!)}).`;
    }
    case "softmax":
      return `${at}: softmax turns the scores into probabilities that sum to 1.`;
    case "target":
      return `${at}: the next character was ${c(s.target)}; the model gave it ${pct(s.pTarget!)}.`;
    case "loss":
      return `${at}: loss = −log ${sig(s.pTarget!)} = ${sig(s.loss!)}.`;
    case "grad":
      return `${at}: gradient (p − y) ÷ 8: the logit of ${c(s.target)} is pushed up by ${sig(-s.grad![s.target]!)}, every other logit down by its probability ÷ 8.`;
    case "mean":
      return `The loss is the mean over the 8 positions: ${sig(s.mean!, 4)} (perplexity e^loss = ${sig(Math.exp(s.mean!))}).`;
  }
}
