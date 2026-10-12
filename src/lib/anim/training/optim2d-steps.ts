/**
 * Chapter 5: four optimisers on a 2-D valley,
 *   f(x, y) = ½ (a (x − x*)² + b (y − y*)²),  a = 1, b = 50,
 * steep across (y) and shallow along (x), the shape that makes plain SGD
 * zig-zag. Each optimiser is the trainer's own (optimizers.ts), stepping a
 * 1×2 matrix θ = [x, y] (a matrix, so AdamW's weight decay applies).
 * scripts/training/anim.py:optim2d_states builds the same trajectories.
 */
import {
  createOptimizer,
  optimizerStep,
  type OptimizerKind,
  type OptimizerState,
} from "@/lib/training/optimizers";

export const VALLEY = {
  a: 1,
  b: 50,
  xStar: 2,
  yStar: 1,
  start: [-2.5, -1.5],
  steps: 60,
};

export type Optim2dSpec = {
  kind: OptimizerKind;
  lr: number;
  momentum?: number;
  weightDecay?: number;
  label: string;
};

export const OPTIM2D: Optim2dSpec[] = [
  { kind: "sgd", lr: 0.036, label: "SGD" },
  { kind: "momentum", lr: 0.02, momentum: 0.8, label: "Momentum" },
  { kind: "adam", lr: 0.1, label: "Adam" },
  { kind: "adamw", lr: 0.1, weightDecay: 0.3, label: "AdamW" },
];

export type Optim2dState = {
  t: number;
  points: [number, number][];
  losses: number[];
  /** Adam's first and second moments for x and y */
  adamM: number[];
  adamV: number[];
};

export function valleyLoss([x, y]: [number, number]): number {
  const v = VALLEY;
  return 0.5 * (v.a * (x - v.xStar) ** 2 + v.b * (y - v.yStar) ** 2);
}

function valleyGrad(theta: number[]): number[][] {
  const v = VALLEY;
  return [[v.a * (theta[0]! - v.xStar), v.b * (theta[1]! - v.yStar)]];
}

/** Every optimiser's position after each step (t = 0 … steps). */
export function optim2dStates(specs: Optim2dSpec[] = OPTIM2D): Optim2dState[] {
  const params = specs.map(() => ({ theta: [VALLEY.start.slice()] }));
  const opts: OptimizerState[] = specs.map((s, i) =>
    createOptimizer(params[i]!, {
      kind: s.kind,
      momentum: s.momentum ?? 0.9,
      weightDecay: s.weightDecay ?? 0,
    }),
  );
  const adam = specs.findIndex((s) => s.kind === "adam");
  const out: Optim2dState[] = [];
  for (let t = 0; t <= VALLEY.steps; t++) {
    const points = params.map(
      (p) => [p.theta[0]![0]!, p.theta[0]![1]!] as [number, number],
    );
    out.push({
      t,
      points,
      losses: points.map(valleyLoss),
      adamM: (opts[adam]!.m["theta"] as number[][])[0]!.slice(),
      adamV: (opts[adam]!.v["theta"] as number[][])[0]!.slice(),
    });
    if (t === VALLEY.steps) break;
    specs.forEach((s, i) => {
      optimizerStep(
        opts[i]!,
        params[i]!,
        { theta: valleyGrad(params[i]!.theta[0]!) },
        s.lr,
      );
    });
  }
  return out;
}

export function optim2dCaption(
  s: Optim2dState,
  specs: Optim2dSpec[] = OPTIM2D,
): string {
  if (s.t === 0)
    return `Step 0: all four start at (−2.5, −1.5). The valley is 50× steeper across (y) than along (x); the minimum is at (2, 1).`;
  const parts = specs.map(
    (sp, i) =>
      `${sp.label} ${s.losses[i]! < 1e-3 ? s.losses[i]!.toExponential(1) : s.losses[i]!.toFixed(3)}`,
  );
  return `Step ${s.t}: loss ${parts.join(" · ")}.`;
}
