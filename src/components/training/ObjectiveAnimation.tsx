"use client";

/**
 * Chapter 3: next-token cross-entropy on the window "alice was", position
 * by position: the 64 logits, their softmax, the right answer and its
 * probability, −log p, and the gradient (p − y)/8 over the vocabulary.
 * States from `objectiveStates` (the trainer's own loss code), on the
 * default run's weights or Part 1's initial ones.
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
  OBJECTIVE_HL,
  objectiveCaption,
  objectiveStates,
} from "@/lib/anim/training/objective-steps";
import { fromWeights } from "@/lib/training/params";
import { PART1_MODEL } from "@/lib/training/trainer";
import { initModelWeights } from "@/lib/transformer/init";
import { ALPHABET } from "@/lib/transformer/tokenizer";
import { OKABE_ITO } from "@/lib/viz/palette";

import { ModelChoice, type ModelKind } from "./ModelChoice";
import { useTrainedWeights } from "./useTrainingData";

export default function ObjectiveAnimation({
  children,
  testId = "objective-animation",
}: {
  children?: ReactNode;
  testId?: string;
}): JSX.Element {
  const [kind, setKind] = useState<ModelKind>("trained");
  const trained = useTrainedWeights(kind === "trained");
  const params = useMemo(() => {
    if (kind === "init") return fromWeights(initModelWeights(PART1_MODEL));
    return trained && !("error" in trained) ? trained : null;
  }, [kind, trained]);
  const states = useMemo(
    () => (params ? objectiveStates(params) : []),
    [params],
  );
  const st = useStepper(Math.max(1, states.length), {
    stepMs: 800,
    resetKey: params ? kind : "",
  });
  const [hover, setHover] = useState<string | null>(null);
  const box = useContainerWidth();
  const phone = box.width > 0 && box.width < PHONE_BELOW;
  const W = phone ? 360 : 720;
  const font = useSvgFont(W);
  const fs = font.fs;

  const select = (
    <ModelChoice
      value={kind}
      onChange={setKind}
      label="Cross-entropy animation weights"
    />
  );
  if (!params || states.length === 0) {
    return (
      <div
        data-pending-widget
        className="my-8 min-h-[40rem] text-sm text-neutral-600 dark:text-neutral-400"
      >
        <p>
          {trained && "error" in trained
            ? `Could not load the weights: ${trained.error}`
            : "Loading the weights…"}
        </p>
        {select}
      </div>
    );
  }
  const s = states[st.step]!;
  const hl = OBJECTIVE_HL[s.phase];
  const on = (k: string) => hl === k || hover === k;
  const S = s.inputs.length;
  const V = s.logits.length;

  // ─── layout ──────────────────────────────────────────────
  const cw = (W - 10) / S;
  const topY = fs(12) + 10;
  const ch = fs(14) * 2 + 16;
  const lossY = topY + ch + 6;
  const lossH = 26;
  const chartY = lossY + lossH + fs(12) + 26;
  const chartH = phone ? 170 : 200;
  const bw = W / V;
  const H = chartY + chartH + fs(11) * 2 + 16;
  const showGrad = s.phase === "grad" || s.phase === "mean";
  const values = showGrad
    ? s.grad!
    : s.phase === "logits"
      ? s.logits
      : s.probs!;
  const vmax = Math.max(1e-12, ...values.map((v) => Math.abs(v)));
  const signed = showGrad || s.phase === "logits";
  const mid = signed ? chartY + chartH / 2 : chartY + chartH;
  const scale = signed ? chartH / 2 - 4 : chartH - 4;
  const lmax = Math.max(Math.log(64), ...s.losses.map((l) => l ?? 0));
  const top3 = values
    .map((v, j) => [Math.abs(v), j] as const)
    .sort((a, b) => b[0] - a[0])
    .slice(0, 3)
    .map(([, j]) => j)
    // the right answer is labelled under the chart instead
    .filter(
      (j) => !(j === s.target && s.phase !== "logits" && s.phase !== "softmax"),
    );

  const visual = (
    <div ref={box.ref} className="min-w-0">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="mx-auto w-full text-neutral-800 dark:text-neutral-200"
        role="img"
        aria-label="The window's eight characters with the character each must predict below it; a strip of the losses so far; and a bar chart over the 64 characters of the logits, probabilities or gradient at the current position, the right answer outlined"
      >
        <text x={0} y={topY - 6} fontSize={fs(12)} fill="currentColor">
          input → target (the next character)
        </text>
        {s.inputs.map((id, i) => {
          const cur = i === s.pos;
          return (
            <g
              key={i}
              data-testid={`pos-${i}`}
              data-current={cur ? "true" : undefined}
            >
              <rect
                x={i * cw + 2}
                y={topY}
                width={cw - 4}
                height={ch}
                rx={4}
                fill={cur ? OKABE_ITO.blue : "none"}
                fillOpacity={cur ? 0.15 : 0}
                stroke={cur ? OKABE_ITO.blue : "currentColor"}
                strokeOpacity={cur ? 1 : 0.25}
                strokeWidth={cur ? 2 : 1}
              />
              <text
                x={i * cw + cw / 2}
                y={topY + fs(14) + 4}
                fontSize={fs(14)}
                textAnchor="middle"
                fill="currentColor"
                fontFamily="monospace"
              >
                {showChar(ALPHABET[id]!)}
              </text>
              <text
                x={i * cw + cw / 2}
                y={topY + 2 * fs(14) + 10}
                fontSize={fs(14)}
                textAnchor="middle"
                fill="currentColor"
                fontFamily="monospace"
                fontWeight={
                  cur && s.phase !== "logits" && s.phase !== "softmax"
                    ? 700
                    : 400
                }
              >
                {`→${showChar(ALPHABET[s.targets[i]!]!)}`}
              </text>
              {s.losses[i] !== null && (
                <rect
                  x={i * cw + 2}
                  y={lossY + lossH - (lossH * s.losses[i]!) / lmax}
                  width={cw - 4}
                  height={(lossH * s.losses[i]!) / lmax}
                  fill={OKABE_ITO.vermillion}
                  fillOpacity={on("loss") && cur ? 0.95 : 0.6}
                />
              )}
            </g>
          );
        })}
        <text x={0} y={chartY - 10} fontSize={fs(12)} fill="currentColor">
          {showGrad
            ? "∂loss/∂logit for each of the 64 characters: (p − y)/8"
            : s.phase === "logits"
              ? "logits z: the model's score for each of the 64 characters"
              : "p = softmax(z): the probability of each character"}
        </text>
        <line
          x1={0}
          x2={W}
          y1={mid}
          y2={mid}
          stroke="currentColor"
          strokeOpacity={0.4}
        />
        {values.map((v, j) => {
          const isT =
            j === s.target && s.phase !== "logits" && s.phase !== "softmax";
          const h = (scale * Math.abs(v)) / vmax;
          const colour = showGrad
            ? v < 0
              ? OKABE_ITO.blue
              : OKABE_ITO.orange
            : s.phase === "logits"
              ? OKABE_ITO.purple
              : OKABE_ITO.purple;
          return (
            <rect
              key={j}
              data-testid={isT ? "target-bar" : undefined}
              x={j * bw + 0.5}
              y={v >= 0 ? mid - h : mid}
              width={Math.max(1, bw - 1)}
              height={Math.max(0.5, h)}
              fill={isT ? OKABE_ITO.blue : colour}
              fillOpacity={isT ? 1 : 0.6}
              stroke={isT ? "currentColor" : "none"}
            />
          );
        })}
        {/* the three largest bars, named */}
        {top3.map((j) => {
          const v = values[j]!;
          const h = (scale * Math.abs(v)) / vmax;
          return (
            <text
              key={`l${j}`}
              x={(j + 0.5) * bw}
              y={v >= 0 ? mid - h - 4 : mid + h + fs(11) + 2}
              fontSize={fs(11)}
              textAnchor="middle"
              fill="currentColor"
              fontFamily="monospace"
            >
              {showChar(ALPHABET[j]!)}
            </text>
          );
        })}
        {s.phase !== "logits" && s.phase !== "softmax" && (
          <text
            x={Math.min(W - 4, Math.max(4, (s.target + 0.5) * bw))}
            y={chartY + chartH + fs(11) + 6}
            fontSize={fs(11)}
            textAnchor={
              (s.target + 0.5) * bw < W * 0.2
                ? "start"
                : (s.target + 0.5) * bw > W * 0.8
                  ? "end"
                  : "middle"
            }
            fill="currentColor"
          >
            {`'${showChar(ALPHABET[s.target]!)}'${s.pTarget !== null ? ` p = ${sig(s.pTarget)}` : ""}`}
          </text>
        )}
        <text
          x={0}
          y={H - 4}
          fontSize={fs(11)}
          fill="currentColor"
          data-testid="objective-readout"
        >
          {s.mean !== null
            ? `mean loss ${sig(s.mean, 4)} · perplexity ${sig(Math.exp(s.mean))}`
            : s.loss !== null
              ? `loss at position ${s.pos}: −log p = ${sig(s.loss)}`
              : `position ${s.pos} of ${S}`}
        </text>
      </svg>
    </div>
  );

  return (
    <AnimationPanel
      testId={testId}
      title="Cross-entropy, one position at a time"
      summary={`The window "alice was": 8 inputs, 8 targets, all scored in one forward pass (teacher forcing). ${kind === "trained" ? "Weights after the default run." : "Part 1's initial weights: every probability is close to 1/64."}`}
      stepper={st}
      stepLabel="state"
      caption={objectiveCaption(s)}
      visual={visual}
      equation={children}
      hl={hl}
      onEquationHover={setHover}
      params={select}
    />
  );
}
