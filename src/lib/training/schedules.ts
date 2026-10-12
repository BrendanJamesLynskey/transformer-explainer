/*
 * src/lib/training/schedules.ts
 *
 * Operation: learning-rate schedules, lr(t) for steps t = 0 … total − 1.
 * Shapes:    scalar → scalar.
 * Intuition: start small (warmup, while Adam's moment estimates settle),
 *            then either hold (constant), glide down a half cosine to a
 *            floor (cosine), or hold and drop linearly at the end
 *            (warmup–stable–decay, "WSD"). The floor is min_ratio · lr.
 *              warmup   lr · (t + 1) / W                       for t < W
 *              cosine   lr_min + ½ (lr − lr_min)(1 + cos(π (t − W)/(T − W)))
 *              WSD      lr until T − D, then linear to lr_min over D steps
 *            The cosine decay is checked against PyTorch's
 *            CosineAnnealingLR (scripts/training/checks.py).
 * MDX:       /training/01-training-loop (the default run's schedule).
 */
export type Schedule = {
  kind: "constant" | "cosine" | "wsd";
  lr: number;
  total: number;
  warmup?: number;
  /** lr_min = min_ratio · lr (default 0.1). */
  minRatio?: number;
  /** WSD: the fraction of the run spent decaying (default 0.2). */
  decayFrac?: number;
};

/** The learning rate at step t. Matches scripts/training/optim.py:lr_at. */
export function lrAt(t: number, s: Schedule): number {
  const warm = s.warmup ?? 0;
  const lrMin = s.lr * (s.minRatio ?? 0.1);
  if (s.kind === "constant") return s.lr;
  if (t < warm) return (s.lr * (t + 1)) / warm;
  if (s.kind === "cosine") {
    const progress = (t - warm) / Math.max(1, s.total - warm);
    return lrMin + 0.5 * (s.lr - lrMin) * (1 + Math.cos(Math.PI * progress));
  }
  const decayStart = s.total - Math.floor((s.decayFrac ?? 0.2) * s.total);
  if (t < decayStart) return s.lr;
  const frac = (t - decayStart) / Math.max(1, s.total - decayStart);
  return s.lr + (lrMin - s.lr) * frac;
}
