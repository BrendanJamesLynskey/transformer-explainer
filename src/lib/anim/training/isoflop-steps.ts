/**
 * Chapter 14's hero: the toy IsoFLOP sweep (results.md §14) drawn budget by
 * budget: each measured point (final validation loss against the model's
 * size, at a fixed compute budget C = 6 N D), then the parabola through the
 * lowest point and its two neighbours and its minimum N_opt, then the
 * frontier: a line through log N_opt against log C. The fits are computed
 * here, in the same order as scripts/training/isoflop.py (tests:
 * anim-frames-c.test.ts against anim_c.py's isoflop_states).
 */
import { sig } from "@/lib/anim/format";

export type IsoPoint = {
  width: number;
  params: number;
  steps: number;
  tokens: number;
  mean: number | null;
};
export type IsoRow = { budget: number; points: IsoPoint[] };

export type IsoState =
  | {
      kind: "point";
      budget: number;
      params: number;
      loss: number;
      shown: number;
    }
  | {
      kind: "fit";
      budget: number;
      c: number[];
      nOpt: number | null;
      lossOpt: number | null;
      fitParams: number[];
      shown: number;
    }
  | { kind: "nofit"; budget: number; shown: number }
  | { kind: "frontier"; a: number; intercept: number; shown: number };

/** Least squares y = c0 + c1 x + c2 x², normal equations by Cramer's rule. */
export function fitParabola(
  xs: number[],
  ys: number[],
): { c: number[]; xMin: number | null; yMin: number | null } {
  const s = [0, 0, 0, 0, 0];
  const t = [0, 0, 0];
  xs.forEach((x, i) => {
    let p = 1;
    for (let k = 0; k < 5; k++) {
      s[k] = s[k]! + p;
      if (k < 3) t[k] = t[k]! + p * ys[i]!;
      p *= x;
    }
  });
  const a = [
    [s[0]!, s[1]!, s[2]!],
    [s[1]!, s[2]!, s[3]!],
    [s[2]!, s[3]!, s[4]!],
  ];
  const det3 = (m: number[][]) =>
    m[0]![0]! * (m[1]![1]! * m[2]![2]! - m[1]![2]! * m[2]![1]!) -
    m[0]![1]! * (m[1]![0]! * m[2]![2]! - m[1]![2]! * m[2]![0]!) +
    m[0]![2]! * (m[1]![0]! * m[2]![1]! - m[1]![1]! * m[2]![0]!);
  const d = det3(a);
  const c: number[] = [];
  for (let j = 0; j < 3; j++) {
    const mj = a.map((row) => row.slice());
    for (let i = 0; i < 3; i++) mj[i]![j] = t[i]!;
    c.push(det3(mj) / d);
  }
  const xMin = c[2]! > 0 ? -c[1]! / (2 * c[2]!) : null;
  return {
    c,
    xMin,
    yMin: xMin !== null ? c[0]! + c[1]! * xMin + c[2]! * xMin * xMin : null,
  };
}

/** The lowest mean and its two neighbours; null if the lowest is at an edge. */
export function fitPoints(ok: IsoPoint[]): IsoPoint[] | null {
  if (ok.length < 3) return null;
  let i = 0;
  ok.forEach((p, j) => {
    if (p.mean! < ok[i]!.mean!) i = j;
  });
  if (i === 0 || i === ok.length - 1) return null;
  return ok.slice(i - 1, i + 2);
}

export function isoflopStates(
  rows: IsoRow[],
  frontier: { a: number; intercept: number },
): IsoState[] {
  const states: IsoState[] = [];
  let shown = 0;
  rows.forEach((row, bi) => {
    for (const p of row.points) {
      if (p.mean === null) continue;
      shown++;
      states.push({
        kind: "point",
        budget: bi,
        params: p.params,
        loss: p.mean,
        shown,
      });
    }
    const near = fitPoints(row.points.filter((p) => p.mean !== null));
    if (!near) {
      states.push({ kind: "nofit", budget: bi, shown });
      return;
    }
    const fit = fitParabola(
      near.map((p) => Math.log10(p.params)),
      near.map((p) => p.mean!),
    );
    states.push({
      kind: "fit",
      budget: bi,
      c: fit.c,
      nOpt: fit.xMin !== null ? Math.pow(10, fit.xMin) : null,
      lossOpt: fit.yMin,
      fitParams: near.map((p) => p.params),
      shown,
    });
  });
  states.push({
    kind: "frontier",
    a: frontier.a,
    intercept: frontier.intercept,
    shown,
  });
  return states;
}

export const ISO_HL: Record<IsoState["kind"], string> = {
  point: "iso",
  fit: "iso",
  nofit: "iso",
  frontier: "front",
};

const fmtC = (c: number) => c.toExponential(0).replace("e+", "e");

export function isoflopCaption(s: IsoState, rows: IsoRow[]): string {
  if (s.kind === "frontier")
    return `The frontier: N_opt grows as C^${sig(s.a, 2)} on the toy. Chinchilla measured about 0.5 (half of each extra FLOP into parameters, half into tokens); Kaplan et al. 0.73.`;
  const row = rows[s.budget]!;
  if (s.kind === "point") {
    const p = row.points.find((x) => x.params === s.params)!;
    return `Budget ${fmtC(row.budget)} FLOPs: width ${p.width}, ${p.params.toLocaleString("en-GB")} parameters, trained for ${p.tokens.toLocaleString("en-GB")} tokens (${p.steps.toLocaleString("en-GB")} steps): validation loss ${sig(s.loss)}.`;
  }
  if (s.kind === "nofit")
    return `Budget ${fmtC(row.budget)}: the lowest loss is at the edge of the sizes tried, so this budget has no minimum to fit (the larger models never left the plateau in the steps they had).`;
  return `Budget ${fmtC(row.budget)}: a parabola in log N through the three best sizes; its minimum, ${s.nOpt === null ? "none" : `N_opt ≈ ${Math.round(s.nOpt).toLocaleString("en-GB")} parameters`}, is the compute-optimal size for this budget.`;
}
