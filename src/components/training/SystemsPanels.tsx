/**
 * Server wrappers for chapters 11-15's widgets: each hands its client
 * component the few numbers it needs from results.json (results.md §11-15),
 * so the browser does not download the whole results file.
 */
import type { ReactNode } from "react";

import results from "@/data/training/results.json";
import type { IsoRow } from "@/lib/anim/training/isoflop-steps";
import { gptArch } from "@/lib/training/systems";

import {
  DataParallelCalculator,
  GoodputAnimation,
  IsoflopAnimation,
  LayoutExplorer,
  WanTimeToTarget,
} from "./lazy";

const LLAMA7B = gptArch("LLaMA 7B", 32, 4096, 32, 4096, {
  V: 32000,
  f: 11008,
  swiglu: true,
  tied: false,
});

export function DataParallelPanel(): JSX.Element {
  return (
    <DataParallelCalculator
      arch={LLAMA7B}
      eta={results.systems.validation.eta}
    />
  );
}

export function LayoutPanel(): JSX.Element {
  return <LayoutExplorer eta={results.systems.validation.eta} />;
}

export function IsoflopPanel({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const iso = results.isoflop;
  const rows: IsoRow[] = iso.rows.map((r) => ({
    budget: r.budget,
    points: r.points.map((p) => ({
      width: p.width,
      params: p.params,
      steps: p.steps,
      tokens: p.tokens,
      mean: p.mean,
    })),
  }));
  return (
    <IsoflopAnimation
      rows={rows}
      frontier={{ a: iso.a, intercept: iso.frontier.intercept }}
    >
      {children}
    </IsoflopAnimation>
  );
}

export function GoodputPanel({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  return (
    <GoodputAnimation params={results.systems.goodputAnim}>
      {children}
    </GoodputAnimation>
  );
}

export function WanPanel(): JSX.Element {
  const x = results.slowLinks;
  return (
    <WanTimeToTarget
      rows={x.rows.map((r) => ({
        name: r.name,
        method: r.method,
        H: r.H,
        evalSteps: r.evalSteps,
        meanEvals: r.meanEvals,
        wirePerParam: r.bytes.wire / r.bytes.params,
      }))}
    />
  );
}
