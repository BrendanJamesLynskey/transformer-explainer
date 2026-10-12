/*
 * src/lib/training/loss.ts
 *
 * Operation: next-token cross-entropy and its gradient.
 * Shapes:    crossEntropy(logits: [S, V], targets: [S]) →
 *              { perPosition: [S], probs: [S, V] }
 *            crossEntropyGrad(probs, targets, n) → dlogits [S, V]
 * Intuition: the loss at a position is −log of the probability the model
 *            gave the character that actually came next:
 *              ℓ_i = logsumexp(z_i) − z_i[y_i]
 *            Its gradient with respect to the logits is the famous p − y:
 *            push the right answer's logit up by (1 − p_y), every other
 *            logit down by its probability. The batch loss is the mean over
 *            all n positions, so each gradient is divided by n.
 * MDX:       /training/03-objective.
 */
import type { Matrix, Vector } from "@/lib/transformer/types";

/** Per-position cross-entropy and the softmax probabilities. */
export function crossEntropy(
  logits: Matrix,
  targets: number[],
): { perPosition: Vector; probs: Matrix } {
  const perPosition: Vector = [];
  const probs: Matrix = [];
  for (let i = 0; i < logits.length; i++) {
    const z = logits[i]!;
    let m = -Infinity;
    for (const v of z) if (v > m) m = v;
    const e = z.map((v) => Math.exp(v - m));
    let s = 0;
    for (const v of e) s += v;
    // logsumexp(z) = m + log Σ exp(z − m): the max shift keeps exp finite
    perPosition.push(m + Math.log(s) - z[targets[i]!]!);
    probs.push(e.map((v) => v / s));
  }
  return { perPosition, probs };
}

/** d loss / d logits = (p − onehot(y)) / n for the mean over n positions. */
export function crossEntropyGrad(
  probs: Matrix,
  targets: number[],
  n: number,
): Matrix {
  return probs.map((p, i) =>
    p.map((v, j) => (v - (j === targets[i] ? 1 : 0)) / n),
  );
}
