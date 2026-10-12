"use client";

/**
 * Chapter 1's hero: three real training steps of Part 1's model. A batch of
 * 4 windows flows forward (each target's probability lights its cell), the
 * loss lights up per position, the gradient flows back group by group (last
 * layer first, with ‖g‖), and AdamW updates the weights (‖Δθ‖/‖θ‖ per group,
 * and the batch's loss after the step). Every state comes from
 * `heroStates`, which runs the trainer itself in the browser.
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import {
  PHONE_BELOW,
  useContainerWidth,
} from "@/components/viz/useContainerWidth";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { showChar, sig } from "@/lib/anim/format";
import {
  GROUPS,
  HERO,
  HERO_HL,
  heroCaption,
  heroStates,
} from "@/lib/anim/training/hero-steps";
import { ALPHABET } from "@/lib/transformer/tokenizer";
import { MUTED, OKABE_ITO } from "@/lib/viz/palette";

import { useCorpus } from "./useTrainingData";

/** Box colours for the groups (final LN, FFN, attention, …, embedding). */
const GROUP_COLOUR = [
  OKABE_ITO.sky,
  OKABE_ITO.green,
  OKABE_ITO.vermillion,
  OKABE_ITO.green,
  OKABE_ITO.vermillion,
  OKABE_ITO.blue,
];

export default function TrainingLoopHero({
  children,
  testId = "training-hero",
}: {
  children?: ReactNode;
  testId?: string;
}): JSX.Element {
  const corpus = useCorpus();
  const ids = corpus && "ids" in corpus ? corpus.ids : null;
  const states = useMemo(() => (ids ? heroStates(ids) : []), [ids]);
  const st = useStepper(Math.max(1, states.length), {
    stepMs: 900,
    resetKey: ids ? "ready" : "",
  });
  const [hover, setHover] = useState<string | null>(null);
  const box = useContainerWidth();
  const phone = box.width > 0 && box.width < PHONE_BELOW;
  const W = phone ? 360 : 720;
  const font = useSvgFont(W);
  const fs = font.fs;

  if (!ids || states.length === 0) {
    return (
      <p
        data-pending-widget
        className="my-8 min-h-[40rem] text-sm text-neutral-600 dark:text-neutral-400"
      >
        {corpus && "error" in corpus
          ? `Could not load the corpus: ${corpus.error}`
          : "Loading the corpus…"}
      </p>
    );
  }
  const s = states[st.step]!;
  const hl = HERO_HL[s.phase];
  const on = (k: string) => hl === k || hover === k;
  const B = s.inputs.length;
  const S = s.inputs[0]!.length;

  // ─── layout ──────────────────────────────────────────────
  const cw = phone ? 40 : 42;
  const ch = phone ? 40 : 44;
  const gridW = cw * S;
  const gx = phone ? (W - gridW) / 2 : 0;
  const gy = fs(12) + 14;
  const gridH = ch * B;
  const stackX = phone ? 20 : gridW + 60;
  const stackW = phone ? W - 40 : W - stackX;
  const stackY = phone ? gy + gridH + fs(12) * 2 + 30 : gy;
  const bh = phone ? 34 : 40;
  const gap = 6;
  const stackH = GROUPS.length * (bh + gap);
  const lossY = Math.max(gy + gridH, stackY + stackH) + fs(12) + 22;
  const H = lossY + 34;

  let maxLoss = 0;
  for (const r of s.perPosition ?? [])
    for (const v of r) maxLoss = Math.max(maxLoss, v);
  const maxNorm = Math.max(1e-12, ...s.gradNorms.map((n) => n ?? 0));

  const visual = (
    <div ref={box.ref} className="min-w-0">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="mx-auto w-full text-neutral-800 dark:text-neutral-200"
        role="img"
        aria-label="A batch of four windows of characters as a grid, each cell coloured by the model's probability for the right next character or by its loss; beside it the model's six groups of weights, from the embedding at the bottom to the final LayerNorm at the top, each showing its gradient's size and, after the update, how much it moved; below, the batch's loss before and after the step"
      >
        <text x={gx} y={gy - 10} fontSize={fs(12)} fill="currentColor">
          {`batch: ${B} windows × ${S} characters (small: the target, the next character)`}
        </text>
        {s.inputs.map((row, b) =>
          row.map((id, i) => {
            const pT = s.pTarget?.[b]?.[i];
            const l = s.perPosition?.[b]?.[i];
            let fill: string = "none";
            let op = 0;
            if (s.phase === "forward" && pT !== undefined) {
              fill = OKABE_ITO.purple;
              op = 0.1 + 0.85 * Math.min(1, pT / 0.2);
            } else if (s.phase !== "batch" && l !== undefined) {
              fill = OKABE_ITO.vermillion;
              op = 0.08 + 0.85 * (maxLoss > 0 ? l / maxLoss : 0);
            }
            return (
              <g key={`${b}-${i}`} data-testid={`cell-${b}-${i}`}>
                <rect
                  x={gx + i * cw + 1}
                  y={gy + b * ch + 1}
                  width={cw - 2}
                  height={ch - 2}
                  rx={3}
                  fill={fill}
                  fillOpacity={op}
                  stroke={on("batch") ? OKABE_ITO.blue : "currentColor"}
                  strokeOpacity={on("batch") ? 1 : 0.25}
                />
                <text
                  x={gx + i * cw + cw / 2}
                  y={gy + b * ch + ch * 0.48}
                  fontSize={fs(14)}
                  textAnchor="middle"
                  fill="currentColor"
                  fontFamily="monospace"
                >
                  {showChar(ALPHABET[id]!)}
                </text>
                <text
                  x={gx + i * cw + cw / 2}
                  y={gy + b * ch + ch * 0.85}
                  fontSize={fs(10)}
                  textAnchor="middle"
                  fill="currentColor"
                  fillOpacity={0.75}
                  fontFamily="monospace"
                >
                  {showChar(ALPHABET[s.targets[b]![i]!]!)}
                </text>
              </g>
            );
          }),
        )}

        {/* the model's groups of weights, embedding at the bottom */}
        <text x={stackX} y={stackY - 10} fontSize={fs(12)} fill="currentColor">
          {s.phase === "update"
            ? "weights: ‖Δθ‖ / ‖θ‖ this step"
            : "the model: ‖gradient‖ per group"}
        </text>
        {GROUPS.map((g, k) => {
          const y = stackY + k * (bh + gap);
          const n = s.gradNorms[k];
          const reached = n !== null && n !== undefined;
          const active = s.phase === "backward" && s.group === k;
          const fwd = s.phase === "forward";
          const ratio = s.updateRatios?.[k];
          const barW = reached
            ? Math.max(2, (stackW * 0.45 * n!) / maxNorm)
            : 0;
          return (
            <g
              key={g.label}
              data-testid={`group-${k}`}
              data-reached={reached ? "true" : "false"}
            >
              <rect
                x={stackX}
                y={y}
                width={stackW}
                height={bh}
                rx={4}
                fill={GROUP_COLOUR[k]}
                fillOpacity={active ? 0.35 : fwd ? 0.22 : 0.1}
                stroke={active ? OKABE_ITO.orange : GROUP_COLOUR[k]}
                strokeWidth={active ? 3 : 1}
              />
              <text
                x={stackX + 8}
                y={y + bh / 2 + fs(11) / 3}
                fontSize={fs(11)}
                fill="currentColor"
              >
                {g.label}
              </text>
              {reached && s.phase !== "update" && (
                <>
                  <rect
                    x={stackX + stackW * 0.5}
                    y={y + bh / 2 - 5}
                    width={barW}
                    height={10}
                    fill={OKABE_ITO.orange}
                    fillOpacity={0.9}
                  />
                  <text
                    x={stackX + stackW - 6}
                    y={y + bh - 4}
                    fontSize={fs(10)}
                    textAnchor="end"
                    fill="currentColor"
                  >
                    {sig(n!)}
                  </text>
                </>
              )}
              {ratio !== undefined && (
                <text
                  x={stackX + stackW - 6}
                  y={y + bh / 2 + fs(11) / 3}
                  fontSize={fs(11)}
                  textAnchor="end"
                  fill="currentColor"
                >
                  {sig(ratio)}
                </text>
              )}
            </g>
          );
        })}
        {/* direction of travel */}
        {(s.phase === "forward" || s.phase === "backward") && (
          <text
            x={phone ? stackX + stackW / 2 : stackX - 30}
            y={phone ? stackY + stackH + fs(12) : stackY + stackH / 2}
            fontSize={fs(18)}
            textAnchor="middle"
            fill={s.phase === "forward" ? OKABE_ITO.purple : OKABE_ITO.orange}
          >
            {phone
              ? s.phase === "forward"
                ? "↑ forward"
                : "↓ backward"
              : s.phase === "forward"
                ? "↑"
                : "↓"}
          </text>
        )}

        {/* the loss, before and after */}
        <text
          x={0}
          y={lossY}
          fontSize={fs(12)}
          fill="currentColor"
          data-testid="hero-loss"
        >
          {s.loss === null
            ? `step ${s.step + 1} of ${HERO.steps}: loss not computed yet`
            : s.lossAfter === null
              ? `step ${s.step + 1}: loss ${sig(s.loss, 4)}`
              : `step ${s.step + 1}: loss ${sig(s.loss, 4)} → ${sig(s.lossAfter, 4)} after the update`}
        </text>
        <rect
          x={0}
          y={lossY + 8}
          width={W}
          height={10}
          fill={MUTED.light}
          fillOpacity={0.25}
        />
        {s.loss !== null && (
          <rect
            x={0}
            y={lossY + 8}
            width={(W * Math.min(s.loss, Math.log(64))) / Math.log(64)}
            height={10}
            fill={OKABE_ITO.vermillion}
            fillOpacity={on("loss") ? 0.9 : 0.6}
          />
        )}
        {s.lossAfter !== null && (
          <rect
            x={0}
            y={lossY + 20}
            width={(W * Math.min(s.lossAfter, Math.log(64))) / Math.log(64)}
            height={6}
            fill={OKABE_ITO.green}
          />
        )}
      </svg>
    </div>
  );

  return (
    <AnimationPanel
      testId={testId}
      title="One training step, three times"
      summary={`Part 1's model, at its initial weights, takes ${HERO.steps} AdamW steps (learning rate ${HERO.lr}) on batches of ${HERO.batch} windows from the corpus, computed in your browser.`}
      stepper={st}
      stepLabel="state"
      caption={heroCaption(s)}
      visual={visual}
      equation={children}
      hl={hl}
      onEquationHover={setHover}
    />
  );
}
