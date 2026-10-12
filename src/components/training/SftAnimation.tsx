"use client";

/**
 * Chapter 16's hero: the trained model's loss on a prompt and a response,
 * character by character, with the prompt masked out of the loss (SFT),
 * then the log probabilities of two continuations of one prompt (what
 * preference optimisation compares). Computed in the browser by Part 1's
 * forward pass on the default run's weights (sft-steps.ts).
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import {
  PHONE_BELOW,
  useContainerWidth,
} from "@/components/viz/useContainerWidth";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { showChar } from "@/lib/anim/format";
import {
  SFT,
  SFT_HL,
  sftCaption,
  sftData,
  sftStates,
} from "@/lib/anim/training/sft-steps";
import { OKABE_ITO } from "@/lib/viz/palette";

import { useTrainedWeights } from "./useTrainingData";

export default function SftAnimation({
  children,
  testId = "sft-animation",
}: {
  children?: ReactNode;
  testId?: string;
}): JSX.Element {
  const trained = useTrainedWeights(true);
  const data = useMemo(
    () => (trained && !("error" in trained) ? sftData(trained) : null),
    [trained],
  );
  const states = useMemo(() => (data ? sftStates(data) : []), [data]);
  const st = useStepper(Math.max(1, states.length), {
    stepMs: 900,
    resetKey: data ? "ready" : "",
  });
  const [, setHover] = useState<string | null>(null);
  const box = useContainerWidth();
  const phone = box.width > 0 && box.width < PHONE_BELOW;
  const W = phone ? 360 : 720;
  const font = useSvgFont(W);
  const fs = font.fs;
  if (!data || states.length === 0) {
    return (
      <p
        data-pending-widget
        className="my-8 min-h-[36rem] text-sm text-neutral-600 dark:text-neutral-400"
      >
        {trained && "error" in trained
          ? `Could not load the weights: ${trained.error}`
          : "Loading the weights…"}
      </p>
    );
  }
  const s = states[st.step]!;
  const toks = [...data.promptTokens, ...data.responseTokens];
  const nP = data.promptTokens.length;
  const cur =
    s.kind === "prompt" ? s.i : s.kind === "response" ? nP + s.i : toks.length;
  const maxLoss = 8;

  // ─── layout: per-character loss bars (phones: two rows) ─────────────
  const perRow = phone ? 10 : toks.length;
  const rowsN = Math.ceil(toks.length / perRow);
  const cw = (W - 4) / perRow;
  const top = fs(12) + 10;
  const barH = phone ? 80 : 110;
  const rowH = barH + fs(13) + fs(11) + 16;
  const cmpY = top + rowsN * rowH + 14;
  const cmpH = s.kind === "compare" ? 120 : 0;
  const H = cmpY + cmpH + 8;

  const visual = (
    <div ref={box.ref} className="min-w-0">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="mx-auto w-full text-neutral-800 dark:text-neutral-200"
        role="img"
        aria-label={`Loss per character on "${SFT.prompt}${SFT.response}", prompt masked`}
      >
        <defs>
          <pattern
            id="sft-mask"
            width={6}
            height={6}
            patternUnits="userSpaceOnUse"
            patternTransform="rotate(45)"
          >
            <rect width={6} height={6} fill="#7f7f7f" fillOpacity={0.25} />
            <line
              x1={0}
              y1={0}
              x2={0}
              y2={6}
              stroke="#7f7f7f"
              strokeWidth={2}
            />
          </pattern>
        </defs>
        <text x={0} y={fs(12)} fontSize={fs(12)} fill="currentColor">
          {phone
            ? "−log p per character (hatched: masked)"
            : "−log p per character (prompt hatched: masked; response counted)"}
        </text>
        {toks.map((t, i) => {
          const row = Math.floor(i / perRow);
          const col = i % perRow;
          const x = 2 + col * cw;
          const y0 = top + row * rowH;
          const shown = i <= cur;
          const masked = i < nP;
          const h = shown ? (Math.min(maxLoss, -t.logp) / maxLoss) * barH : 0;
          return (
            <g key={i} data-masked={masked ? "true" : "false"}>
              <rect
                x={x + 2}
                y={y0 + barH - h}
                width={cw - 4}
                height={h}
                fill={masked ? "url(#sft-mask)" : OKABE_ITO.vermillion}
                stroke={i === cur ? "currentColor" : "none"}
                strokeWidth={1.5}
              />
              <text
                x={x + cw / 2}
                y={y0 + barH + fs(13) + 2}
                fontSize={fs(13)}
                textAnchor="middle"
                fill="currentColor"
                fontFamily="ui-monospace, monospace"
              >
                {showChar(t.char)}
              </text>
              {shown && !phone && (
                <text
                  x={x + cw / 2}
                  y={y0 + barH + fs(13) + fs(11) + 6}
                  fontSize={fs(11)}
                  textAnchor="middle"
                  fill="currentColor"
                >
                  {(-t.logp).toFixed(1)}
                </text>
              )}
            </g>
          );
        })}
        {s.kind === "compare" &&
          (["chosen", "rejected"] as const).map((k, j) => {
            const v = k === "chosen" ? s.chosen : s.rejected;
            const text = k === "chosen" ? SFT.chosen : SFT.rejected;
            const max = Math.max(-s.chosen, -s.rejected);
            const w = ((W - 170) * -v) / max;
            const y = cmpY + 10 + j * 50;
            return (
              <g key={k} data-testid={`pref-${k}`}>
                <text
                  x={0}
                  y={y + 18}
                  fontSize={fs(12)}
                  fill="currentColor"
                  fontFamily="ui-monospace, monospace"
                >
                  {SFT.prefPrompt.trim()} {text}
                </text>
                <rect
                  x={150}
                  y={y + 4}
                  width={Math.max(1, w)}
                  height={22}
                  fill={k === "chosen" ? OKABE_ITO.green : OKABE_ITO.orange}
                />
                <text
                  x={156 + Math.max(1, w)}
                  y={y + 20}
                  fontSize={fs(11)}
                  fill="currentColor"
                >
                  {v.toFixed(1)}
                </text>
              </g>
            );
          })}
      </svg>
    </div>
  );

  return (
    <AnimationPanel
      testId={testId}
      title="From pretraining loss to fine-tuning loss"
      summary="The trained model reads a prompt and a response. Pretraining scores every character; supervised fine-tuning scores only the response's. Then two continuations of one prompt, compared the way preference optimisation compares them."
      stepper={st}
      stepLabel="step"
      caption={sftCaption(s, data)}
      visual={visual}
      equation={children}
      hl={SFT_HL[s.kind]}
      onEquationHover={setHover}
    />
  );
}
