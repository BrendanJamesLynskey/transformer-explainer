"use client";

/**
 * Chapter 14's hero: the toy IsoFLOP sweep (results.md §14), budget by
 * budget. Left: final validation loss against parameters (log scale), one
 * colour per compute budget; each budget's parabola through its three best
 * sizes and its minimum N_opt. Right (below on phones): log N_opt against
 * log C, the toy's frontier beside Chinchilla's slope (0.5) and Kaplan's
 * (0.73). The fits are recomputed in the browser (isoflop-steps.ts).
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import {
  PHONE_BELOW,
  useContainerWidth,
} from "@/components/viz/useContainerWidth";
import { useSvgFont } from "@/components/viz/useSvgFont";
import {
  ISO_HL,
  isoflopCaption,
  isoflopStates,
  type IsoRow,
  type IsoState,
} from "@/lib/anim/training/isoflop-steps";
import { OKABE_ITO } from "@/lib/viz/palette";

const BUDGET_COLOUR = [
  OKABE_ITO.sky,
  OKABE_ITO.green,
  OKABE_ITO.orange,
  OKABE_ITO.vermillion,
  OKABE_ITO.purple,
];
const MARK = ["circle", "square", "triangle", "diamond", "circle"] as const;

function Mark({
  kind,
  x,
  y,
  fill,
}: {
  kind: (typeof MARK)[number];
  x: number;
  y: number;
  fill: string;
}): JSX.Element {
  const r = 4.5;
  if (kind === "square")
    return (
      <rect
        x={x - r}
        y={y - r}
        width={2 * r}
        height={2 * r}
        fill={fill}
        stroke="currentColor"
      />
    );
  if (kind === "triangle")
    return (
      <path
        d={`M${x},${y - r * 1.2}L${x + r * 1.1},${y + r * 0.8}L${x - r * 1.1},${y + r * 0.8}Z`}
        fill={fill}
        stroke="currentColor"
      />
    );
  if (kind === "diamond")
    return (
      <path
        d={`M${x},${y - r * 1.3}L${x + r * 1.1},${y}L${x},${y + r * 1.3}L${x - r * 1.1},${y}Z`}
        fill={fill}
        stroke="currentColor"
      />
    );
  return <circle cx={x} cy={y} r={r} fill={fill} stroke="currentColor" />;
}

export default function IsoflopAnimation({
  rows,
  frontier,
  children,
  testId = "isoflop-animation",
}: {
  rows: IsoRow[];
  frontier: { a: number; intercept: number };
  children?: ReactNode;
  testId?: string;
}): JSX.Element {
  const states = useMemo(() => isoflopStates(rows, frontier), [rows, frontier]);
  const st = useStepper(states.length, { stepMs: 650 });
  const [, setHover] = useState<string | null>(null);
  const box = useContainerWidth();
  const phone = box.width > 0 && box.width < PHONE_BELOW;
  const W = phone ? 360 : 720;
  const font = useSvgFont(W);
  const fs = font.fs;
  const s = states[st.step]!;
  const upto = states.slice(0, st.step + 1);
  const points = upto.filter(
    (x): x is Extract<IsoState, { kind: "point" }> => x.kind === "point",
  );
  const fits = upto.filter(
    (x): x is Extract<IsoState, { kind: "fit" }> => x.kind === "fit",
  );
  const showFrontier = s.kind === "frontier";

  // ─── left panel: loss vs log N ──────────────────────────────────────
  const allP = rows.flatMap((r) => r.points.filter((p) => p.mean !== null));
  const lx0 = Math.log10(Math.min(...allP.map((p) => p.params))) - 0.1;
  const lx1 = Math.log10(Math.max(...allP.map((p) => p.params))) + 0.1;
  const y0 = Math.min(...allP.map((p) => p.mean!)) - 0.05;
  const y1 = 3.0;
  const panelW = phone ? W : W * 0.6;
  const left = 44;
  const top = fs(12) + 10;
  const ph = phone ? 210 : 250;
  const X = (n: number) =>
    left + ((panelW - left - 10) * (Math.log10(n) - lx0)) / (lx1 - lx0);
  const Y = (v: number) =>
    top + (ph * (y1 - Math.min(y1, Math.max(y0, v)))) / (y1 - y0);
  // ─── right panel (or below): log N_opt vs log C ─────────────────────
  const fx0 = phone ? 0 : panelW + 20;
  const fy0 = phone ? top + ph + fs(11) * 3 + 44 : top;
  const fw = phone ? W : W - panelW - 20;
  const fh = phone ? 170 : ph;
  const H = fy0 + fh + fs(11) * 4 + 28;
  const lc0 = Math.log10(rows[0]!.budget) - 0.2;
  const lc1 = Math.log10(rows[rows.length - 1]!.budget) + 0.2;
  const ln0 = lx0;
  const ln1 = lx1;
  const FX = (c: number) => fx0 + 40 + ((fw - 50) * (c - lc0)) / (lc1 - lc0);
  const FY = (n: number) => fy0 + (fh * (ln1 - n)) / (ln1 - ln0);
  const xTicks = [1e3, 3e3, 1e4, 3e4].filter(
    (t) => Math.log10(t) > lx0 && Math.log10(t) < lx1,
  );

  const visual = (
    <div ref={box.ref} className="min-w-0">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="mx-auto w-full text-neutral-800 dark:text-neutral-200"
        role="img"
        aria-label={`IsoFLOP curves of the toy model: ${points.length} runs shown`}
      >
        <defs>
          <clipPath id="iso-frontier-clip">
            <rect x={fx0 + 40} y={fy0} width={fw - 50} height={fh} />
          </clipPath>
        </defs>
        <text x={left} y={fs(12)} fontSize={fs(12)} fill="currentColor">
          {phone
            ? "loss vs parameters (log)"
            : "validation loss vs parameters (log), one curve per budget"}
        </text>
        {[2.0, 2.4, 2.8]
          .filter((v) => v > y0)
          .map((v) => (
            <g key={v}>
              <line
                x1={left}
                x2={panelW - 10}
                y1={Y(v)}
                y2={Y(v)}
                stroke="currentColor"
                strokeOpacity={0.12}
              />
              <text
                x={left - 6}
                y={Y(v) + 4}
                fontSize={fs(11)}
                textAnchor="end"
                fill="currentColor"
              >
                {v.toFixed(1)}
              </text>
            </g>
          ))}
        {xTicks.map((t) => (
          <text
            key={t}
            x={X(t)}
            y={top + ph + fs(11) + 4}
            fontSize={fs(11)}
            textAnchor="middle"
            fill="currentColor"
          >
            {t >= 1e4 ? `${t / 1e3}k` : `${t / 1e3}k`}
          </text>
        ))}
        {fits.map((f) => {
          const [c0, c1, c2] = f.c as [number, number, number];
          const xs = f.fitParams.map((p) => Math.log10(p));
          const a = xs[0]! - 0.08;
          const b = xs[xs.length - 1]! + 0.08;
          const pts = Array.from(
            { length: 30 },
            (_, i) => a + ((b - a) * i) / 29,
          );
          const d = pts
            .map(
              (x, i) =>
                `${i ? "L" : "M"}${X(10 ** x).toFixed(1)},${Y(c0 + c1 * x + c2 * x * x).toFixed(1)}`,
            )
            .join("");
          return (
            <g key={`f${f.budget}`}>
              <path
                d={d}
                fill="none"
                stroke={BUDGET_COLOUR[f.budget]}
                strokeWidth={2}
              />
              {f.nOpt !== null && f.lossOpt !== null && (
                <line
                  x1={X(f.nOpt)}
                  x2={X(f.nOpt)}
                  y1={Y(f.lossOpt) - 8}
                  y2={Y(f.lossOpt) + 8}
                  stroke="currentColor"
                  strokeWidth={2.5}
                  data-testid="iso-nopt"
                />
              )}
            </g>
          );
        })}
        {points.map((p, i) => (
          <Mark
            key={i}
            kind={MARK[p.budget]!}
            x={X(p.params)}
            y={Y(p.loss)}
            fill={BUDGET_COLOUR[p.budget]!}
          />
        ))}
        {rows.map((r, bi) => (
          <g key={`l${bi}`}>
            <Mark
              kind={MARK[bi]!}
              x={left + 8 + bi * ((panelW - left) / rows.length)}
              y={top + ph + fs(11) * 2 + 12}
              fill={BUDGET_COLOUR[bi]!}
            />
            <text
              x={left + 18 + bi * ((panelW - left) / rows.length)}
              y={top + ph + fs(11) * 2 + 16}
              fontSize={fs(11)}
              fill="currentColor"
            >
              {r.budget.toExponential(0).replace("e+", "e")}
            </text>
          </g>
        ))}
        {/* frontier panel */}
        <text x={fx0 + 40} y={fy0 - 4} fontSize={fs(11)} fill="currentColor">
          N_opt vs budget C (log–log)
        </text>
        <rect
          x={fx0 + 40}
          y={fy0}
          width={fw - 50}
          height={fh}
          fill="none"
          stroke="currentColor"
          strokeOpacity={0.3}
        />
        {rows.map((r) => (
          <text
            key={`cx${r.budget}`}
            x={FX(Math.log10(r.budget))}
            y={fy0 + fh + fs(11) + 2}
            fontSize={fs(10)}
            textAnchor="middle"
            fill="currentColor"
          >
            {r.budget.toExponential(0).replace("e+", "e")}
          </text>
        ))}
        {xTicks.map((t) => (
          <text
            key={`ny${t}`}
            x={fx0 + 36}
            y={FY(Math.log10(t)) + 4}
            fontSize={fs(10)}
            textAnchor="end"
            fill="currentColor"
          >
            {`${t / 1e3}k`}
          </text>
        ))}
        {fits
          .filter((f) => f.nOpt !== null)
          .map((f) => (
            <Mark
              key={`n${f.budget}`}
              kind={MARK[f.budget]!}
              x={FX(Math.log10(rows[f.budget]!.budget))}
              y={FY(Math.log10(f.nOpt!))}
              fill={BUDGET_COLOUR[f.budget]!}
            />
          ))}
        {showFrontier && (
          <g data-testid="iso-frontier" clipPath="url(#iso-frontier-clip)">
            <line
              x1={FX(lc0)}
              x2={FX(lc1)}
              y1={FY(frontier.intercept + frontier.a * lc0)}
              y2={FY(frontier.intercept + frontier.a * lc1)}
              stroke="currentColor"
              strokeWidth={2}
            />
            {[
              { a: 0.5, dash: "6 4", name: "0.5" },
              { a: 0.73, dash: "2 3", name: "0.73" },
            ].map((ref) => {
              const mid = (lc0 + lc1) / 2;
              const n0 = frontier.intercept + frontier.a * mid;
              return (
                <line
                  key={ref.name}
                  x1={FX(lc0)}
                  x2={FX(lc1)}
                  y1={FY(n0 + ref.a * (lc0 - mid))}
                  y2={FY(n0 + ref.a * (lc1 - mid))}
                  stroke={OKABE_ITO.blue}
                  strokeDasharray={ref.dash}
                  strokeWidth={1.5}
                />
              );
            })}
          </g>
        )}
        <text
          x={fx0 + 40}
          y={fy0 + fh + 2 * fs(11) + 8}
          fontSize={fs(11)}
          fill="currentColor"
        >
          {showFrontier
            ? `slope: toy ${frontier.a.toFixed(2)} (solid)`
            : "budget C (FLOPs) →"}
        </text>
        {showFrontier && (
          <text
            x={fx0 + 40}
            y={fy0 + fh + 3 * fs(11) + 12}
            fontSize={fs(11)}
            fill="currentColor"
          >
            0.5 dashed, 0.73 dotted
          </text>
        )}
      </svg>
    </div>
  );

  return (
    <AnimationPanel
      testId={testId}
      title="IsoFLOP curves on a toy model"
      summary="Part 1's model at eight widths, each trained for exactly the tokens a compute budget allows (C = 6ND), three seeds each (means shown). Bigger budgets favour bigger models; the parabolas' minima trace the compute-optimal frontier."
      stepper={st}
      stepLabel="step"
      caption={isoflopCaption(s, rows)}
      visual={visual}
      equation={children}
      hl={ISO_HL[s.kind]}
      onEquationHover={setHover}
    />
  );
}
