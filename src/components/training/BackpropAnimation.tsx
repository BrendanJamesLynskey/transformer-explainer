"use client";

/**
 * Chapter 4: the gradient on its way back through Part 1's model for the
 * window "alice was", stop by stop: ∂L/∂logits, through the tied head and the
 * final LayerNorm, into each block's FFN (W₂, GELU′, W₁, LN₂), across the
 * residual, through attention (softmax backward, Q/K/V, LN₁), across the
 * residual again, and into the embedding table. Each picture is the
 * gradient itself (one row per position), recorded by the trainer's own
 * backward pass (`backpropStates`).
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
  STOPS,
  backpropCaption,
  backpropStates,
  type BackpropState,
} from "@/lib/anim/training/backprop-steps";
import { windowIds } from "@/lib/anim/training/objective-steps";
import { fromWeights } from "@/lib/training/params";
import { PART1_MODEL } from "@/lib/training/trainer";
import { initModelWeights } from "@/lib/transformer/init";
import { ALPHABET } from "@/lib/transformer/tokenizer";
import { OKABE_ITO, heat } from "@/lib/viz/palette";

import { ModelChoice, type ModelKind } from "./ModelChoice";
import { useTrainedWeights } from "./useTrainingData";

/** The model drawn top (loss) to bottom (embedding): the way back. */
const ROWS = [
  { key: "loss", label: "loss", colour: OKABE_ITO.vermillion, hl: "loss" },
  { key: "head", label: "head: x_f Eᵀ", colour: OKABE_ITO.purple, hl: "head" },
  { key: "lnf", label: "final LayerNorm", colour: OKABE_ITO.sky, hl: "ln" },
  { key: "b1f", label: "block 1 · FFN", colour: OKABE_ITO.green, hl: "ffn" },
  {
    key: "b1a",
    label: "block 1 · attention",
    colour: OKABE_ITO.vermillion,
    hl: "attn",
  },
  { key: "b0f", label: "block 0 · FFN", colour: OKABE_ITO.green, hl: "ffn" },
  {
    key: "b0a",
    label: "block 0 · attention",
    colour: OKABE_ITO.vermillion,
    hl: "attn",
  },
  { key: "emb", label: "embedding E", colour: OKABE_ITO.blue, hl: "" },
];

function rowOf(s: BackpropState): string {
  if (s.kind === "dlogits") return "loss";
  if (s.kind === "dxf") return "head";
  if (s.kind === "dx_top") return "lnf";
  if (s.kind === "embedding") return "emb";
  const b = s.block === "block 1" ? "b1" : "b0";
  return ["dact", "dpre", "dln2", "dh"].includes(s.kind) ? `${b}f` : `${b}a`;
}

export default function BackpropAnimation({
  children,
  testId = "backprop-animation",
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
    () => (params ? backpropStates(params) : []),
    [params],
  );
  const st = useStepper(Math.max(1, states.length), {
    stepMs: 1200,
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
      label="Backprop animation weights"
    />
  );
  if (!params || states.length === 0) {
    return (
      <div
        data-pending-widget
        className="my-8 min-h-[44rem] text-sm text-neutral-600 dark:text-neutral-400"
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
  const stop = STOPS[s.kind];
  const hl = stop?.hl ?? "res";
  const current = rowOf(s);
  const { inputs } = windowIds();

  // ─── layout ──────────────────────────────────────────────
  const diagW = phone ? W : 210;
  const rowH = phone ? 0 : 34;
  const chipW = (W - 6) / 4;
  const diagH = phone ? 2 * (fs(11) + 18) + 6 : ROWS.length * rowH;
  const hx = phone ? 0 : diagW + 30;
  const hy = phone ? diagH + fs(12) + 18 : fs(12) + 14;
  const cols = s.value[0]?.length ?? 0;
  const labelW = fs(12) * 1.6;
  const normW = phone ? 60 : 80;
  const cellW = Math.max(1, (W - hx - labelW - normW - 6) / cols);
  const S = s.value.length;
  const cellH = phone ? 20 : 24;
  const gridH = S * cellH;
  const maxV = Math.max(1e-30, ...s.value.flat().map((v) => Math.abs(v)));
  const maxN = Math.max(1e-30, ...s.rowNorms);
  const infoY = Math.max(phone ? 0 : diagH, hy + gridH) + fs(12) + 16;
  const H = infoY + fs(11) * 2 + 14;

  const visual = (
    <div ref={box.ref} className="min-w-0">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="mx-auto w-full text-neutral-800 dark:text-neutral-200"
        role="img"
        aria-label="The model's operations from the loss at the top to the embedding at the bottom, the current stop of the gradient highlighted; beside it the gradient at that stop as a heat map with one row per position, and each row's size as a bar"
      >
        {ROWS.map((r, k) => {
          const on = r.key === current || (hover !== null && hover === r.hl);
          const done = ROWS.findIndex((x) => x.key === current) > k;
          const x = phone ? (k % 4) * chipW : 0;
          const y = phone ? Math.floor(k / 4) * (fs(11) + 18) : k * rowH;
          const w = phone ? chipW - 4 : diagW;
          const h = phone ? fs(11) + 12 : rowH - 6;
          return (
            <g
              key={r.key}
              data-testid={`op-${r.key}`}
              data-current={on ? "true" : undefined}
            >
              <rect
                x={x}
                y={y}
                width={w}
                height={h}
                rx={4}
                fill={r.colour}
                fillOpacity={on ? 0.35 : done ? 0.06 : 0.14}
                stroke={on ? OKABE_ITO.orange : r.colour}
                strokeWidth={on ? 3 : 1}
              />
              <text
                x={x + 6}
                y={y + h / 2 + fs(11) / 3}
                fontSize={fs(phone ? 10 : 11)}
                fill="currentColor"
              >
                {phone
                  ? r.label
                      .replace("block ", "b")
                      .replace(" · ", " ")
                      .replace("attention", "attn")
                      .replace("final LayerNorm", "LN_f")
                      .replace("head: x_f Eᵀ", "head")
                      .replace("embedding E", "E")
                  : r.label}
              </text>
            </g>
          );
        })}
        {!phone && (
          <text
            x={diagW + 6}
            y={(ROWS.findIndex((x) => x.key === current) + 0.5) * rowH + 6}
            fontSize={fs(16)}
            fill={OKABE_ITO.orange}
          >
            ◀
          </text>
        )}

        <text x={hx} y={hy - 8} fontSize={fs(12)} fill="currentColor">
          {`${s.kind === "dscores_h0" ? "∂L/∂scores (head 0): row = query, column = key" : `the gradient here: ${S} positions × ${cols} numbers`}`}
        </text>
        {s.value.map((row, i) => (
          <g key={i}>
            <text
              x={hx}
              y={hy + i * cellH + cellH / 2 + fs(11) / 3}
              fontSize={fs(11)}
              fill="currentColor"
              fontFamily="monospace"
            >
              {showChar(ALPHABET[inputs[i]!]!)}
            </text>
            {row.map((v, j) => {
              const c = heat(v, maxV);
              return (
                <rect
                  key={j}
                  x={hx + labelW + j * cellW}
                  y={hy + i * cellH + 1}
                  width={Math.max(0.6, cellW - (cellW > 4 ? 1 : 0))}
                  height={cellH - 2}
                  fill={c.fill}
                  fillOpacity={c.fillOpacity}
                />
              );
            })}
            <rect
              x={W - normW + 4}
              y={hy + i * cellH + cellH / 2 - 4}
              width={Math.max(1, ((normW - 8) * s.rowNorms[i]!) / maxN)}
              height={8}
              fill={OKABE_ITO.orange}
            />
          </g>
        ))}
        <text
          x={W - normW + 4}
          y={hy + gridH + fs(11) + 4}
          fontSize={fs(10)}
          fill="currentColor"
        >
          ‖row‖
        </text>
        <text
          x={0}
          y={infoY}
          fontSize={fs(11)}
          fill="currentColor"
          data-testid="backprop-op"
        >
          {s.kind === "embedding" ? "op: x₀ = E[ids] + P" : `op: ${stop!.op}`}
        </text>
        <text
          x={0}
          y={infoY + fs(11) + 8}
          fontSize={fs(11)}
          fill="currentColor"
        >
          {s.kind === "embedding"
            ? "needs: the token ids"
            : `needs, saved from the forward pass: ${stop!.saved}`}
        </text>
      </svg>
    </div>
  );

  return (
    <AnimationPanel
      testId={testId}
      title="The gradient's way back"
      summary={`One window, "alice was"; loss ${sig(s.loss, 4)}. Each picture is the gradient at that stop, blue positive, orange negative, scaled to its own largest value.`}
      stepper={st}
      stepLabel="stop"
      caption={backpropCaption(s, st.step, states.length)}
      visual={visual}
      equation={children}
      hl={hl}
      onEquationHover={setHover}
      params={select}
      stats={
        s.params.length > 0 ? (
          <p
            className="text-xs text-neutral-700 dark:text-neutral-300"
            data-testid="backprop-params"
          >
            Finished here:{" "}
            {s.params
              .map((p) => `∂L/∂${p.name} (‖·‖ ${sig(p.norm)})`)
              .join(", ")}
          </p>
        ) : undefined
      }
    />
  );
}
