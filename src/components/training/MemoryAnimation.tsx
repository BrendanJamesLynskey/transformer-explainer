"use client";

/**
 * Chapter 10's hero and calculator: the memory one device holds through
 * one training step, event by event (memory.ts `timeline`). Left, the
 * stacked bar of what is in memory now; right, the total over the step,
 * with its peak. The parameters re-run the calculator and restart the
 * animation.
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import {
  PHONE_BELOW,
  useContainerWidth,
} from "@/components/viz/useContainerWidth";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { memoryCaption, memoryStates } from "@/lib/anim/training/memory-steps";
import {
  FP32_ADAM,
  LLAMA_7B,
  MIXED_ADAM,
  PART1_MEM,
  breakdown,
  fmtBytes,
  type Arch,
  type MemEvent,
  type Recompute,
} from "@/lib/training/memory";
import { OKABE_ITO } from "@/lib/viz/palette";

const SELECT =
  "focus-ring h-11 rounded border border-neutral-300 bg-white px-2 text-sm dark:border-neutral-700 dark:bg-neutral-950";

const PARTS: {
  key: keyof MemEvent;
  label: string;
  colour: string;
  hatch?: boolean;
}[] = [
  { key: "weights", label: "weights", colour: OKABE_ITO.blue },
  { key: "master", label: "FP32 master", colour: OKABE_ITO.sky },
  { key: "optimizer", label: "Adam m, v", colour: OKABE_ITO.green },
  { key: "grads", label: "gradients", colour: OKABE_ITO.orange },
  { key: "activations", label: "activations", colour: OKABE_ITO.purple },
  {
    key: "transient",
    label: "recomputed layer",
    colour: OKABE_ITO.purple,
    hatch: true,
  },
  { key: "logits", label: "logits", colour: OKABE_ITO.vermillion },
];

export default function MemoryAnimation({
  children,
  testId = "memory-animation",
}: {
  children?: ReactNode;
  testId?: string;
}): JSX.Element {
  const [model, setModel] = useState<"llama" | "part1">("llama");
  const [recompute, setRecompute] = useState<Recompute>("none");
  const [recipe, setRecipe] = useState<"mixed" | "fp32">("mixed");
  const [seq, setSeq] = useState(4096);
  const [batch, setBatch] = useState(1);
  const arch: Arch = useMemo(
    () => (model === "llama" ? { ...LLAMA_7B, s: seq, b: batch } : PART1_MEM),
    [model, seq, batch],
  );
  const rec = recipe === "mixed" ? MIXED_ADAM : FP32_ADAM;
  const states = useMemo(
    () => memoryStates(arch, rec, recompute),
    [arch, rec, recompute],
  );
  const key = `${model}-${recompute}-${recipe}-${seq}-${batch}`;
  const st = useStepper(states.length, { stepMs: 500, resetKey: key });
  const [, setHover] = useState<string | null>(null);
  const box = useContainerWidth();
  const phone = box.width > 0 && box.width < PHONE_BELOW;
  const W = phone ? 360 : 720;
  const font = useSvgFont(W);
  const fs = font.fs;
  const s = states[st.step]!;
  const peak = Math.max(...states.map((e) => e.total));
  const bd = breakdown(arch, rec, recompute);
  const hl =
    s.event === "forward" || s.event === "recompute"
      ? "memact"
      : s.event === "start" || s.event === "step"
        ? "memopt"
        : "memw";

  // ─── layout: bar (left) and timeline (right); stacked on phones ─────
  const barW = phone ? 90 : 120;
  const top = fs(12) * 1.8;
  const plotH = phone ? 200 : 240;
  const tlX = barW + (phone ? 34 : 70);
  const tlW = W - tlX - 10;
  const legendY = top + plotH + fs(11) * 2 + 10;
  const perRow = phone ? 2 : 4;
  const H = legendY + Math.ceil(PARTS.length / perRow) * (fs(12) + 10);
  const y = (v: number) => top + plotH * (1 - v / peak);
  let acc = 0;
  const segs = PARTS.map((p) => {
    const v = s[p.key] as number;
    const seg = { ...p, y0: y(acc + v), h: (plotH * v) / peak };
    acc += v;
    return seg;
  });
  const tx = (i: number) => tlX + (tlW * i) / Math.max(1, states.length - 1);

  const visual = (
    <div ref={box.ref} className="min-w-0">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="mx-auto w-full text-neutral-800 dark:text-neutral-200"
        role="img"
        aria-label={`Memory in use during one training step of ${arch.name}: now ${fmtBytes(s.total)}, peak ${fmtBytes(peak)}`}
      >
        <defs>
          <pattern
            id="mem-hatch"
            width={6}
            height={6}
            patternUnits="userSpaceOnUse"
            patternTransform="rotate(45)"
          >
            <rect
              width={6}
              height={6}
              fill={OKABE_ITO.purple}
              fillOpacity={0.3}
            />
            <line
              x1={0}
              y1={0}
              x2={0}
              y2={6}
              stroke={OKABE_ITO.purple}
              strokeWidth={2}
            />
          </pattern>
        </defs>
        <text x={0} y={fs(12)} fontSize={fs(12)} fill="currentColor">
          now {fmtBytes(s.total)}
        </text>
        {segs.map((g) => (
          <rect
            key={g.key}
            x={0}
            y={g.y0}
            width={barW}
            height={Math.max(0, g.h)}
            fill={g.hatch ? "url(#mem-hatch)" : g.colour}
            data-testid={`mem-${g.key}`}
          />
        ))}
        <rect
          x={0}
          y={top}
          width={barW}
          height={plotH}
          fill="none"
          stroke="currentColor"
          strokeOpacity={0.3}
        />
        <text x={tlX} y={fs(12)} fontSize={fs(12)} fill="currentColor">
          total over the step · peak {fmtBytes(peak)}
        </text>
        <line
          x1={tlX}
          x2={tlX + tlW}
          y1={y(peak)}
          y2={y(peak)}
          stroke="currentColor"
          strokeOpacity={0.3}
          strokeDasharray="4 3"
        />
        <path
          d={states
            .map(
              (e, i) =>
                `${i ? "L" : "M"}${tx(i).toFixed(1)},${y(e.total).toFixed(1)}`,
            )
            .join("")}
          fill="none"
          stroke={OKABE_ITO.purple}
          strokeWidth={2.5}
        />
        <circle
          cx={tx(st.step)}
          cy={y(s.total)}
          r={5}
          fill={OKABE_ITO.purple}
          stroke="currentColor"
        />
        <line
          x1={tlX}
          x2={tlX + tlW}
          y1={top + plotH}
          y2={top + plotH}
          stroke="currentColor"
          strokeOpacity={0.4}
        />
        <text
          x={tlX}
          y={top + plotH + fs(11) + 2}
          fontSize={fs(11)}
          fill="currentColor"
        >
          start
        </text>
        <text
          x={tx(arch.L + 1)}
          y={top + plotH + fs(11) + 2}
          fontSize={fs(11)}
          textAnchor="middle"
          fill="currentColor"
        >
          loss
        </text>
        <text
          x={tlX + tlW}
          y={top + plotH + fs(11) + 2}
          fontSize={fs(11)}
          textAnchor="end"
          fill="currentColor"
        >
          step
        </text>
        {PARTS.map((p, k) => {
          const col = k % perRow;
          const row = Math.floor(k / perRow);
          const lx = col * (W / perRow);
          const ly = legendY + row * (fs(12) + 10);
          return (
            <g key={p.key}>
              <rect
                x={lx}
                y={ly - fs(12) * 0.8}
                width={fs(12)}
                height={fs(12)}
                fill={p.hatch ? "url(#mem-hatch)" : p.colour}
              />
              <text
                x={lx + fs(12) + 4}
                y={ly}
                fontSize={fs(12)}
                fill="currentColor"
              >
                {p.label}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );

  const params = (
    <div className="grid min-w-0 grid-cols-1 gap-2 sm:grid-cols-3">
      <label className="flex min-w-0 flex-col gap-1 text-sm">
        <span className="font-medium text-neutral-700 dark:text-neutral-300">
          Model
        </span>
        <select
          value={model}
          onChange={(e) => setModel(e.target.value as "llama" | "part1")}
          className={`${SELECT} w-full`}
          aria-label="Model"
        >
          <option value="llama">LLaMA 7B (6.74B parameters)</option>
          <option value="part1">Part 1&apos;s model (5,376)</option>
        </select>
      </label>
      <label className="flex min-w-0 flex-col gap-1 text-sm">
        <span className="font-medium text-neutral-700 dark:text-neutral-300">
          Recipe
        </span>
        <select
          value={recipe}
          onChange={(e) => setRecipe(e.target.value as "mixed" | "fp32")}
          className={`${SELECT} w-full`}
          aria-label="Precision recipe"
        >
          <option value="mixed">BF16 + FP32 master, Adam</option>
          <option value="fp32">FP32, Adam</option>
        </select>
      </label>
      <label className="flex min-w-0 flex-col gap-1 text-sm">
        <span className="font-medium text-neutral-700 dark:text-neutral-300">
          Recompute
        </span>
        <select
          value={recompute}
          onChange={(e) => setRecompute(e.target.value as Recompute)}
          className={`${SELECT} w-full`}
          aria-label="Activation recomputation"
        >
          <option value="none">none</option>
          <option value="selective">selective (attention core)</option>
          <option value="full">full (keep each layer&apos;s input)</option>
        </select>
      </label>
      {model === "llama" && (
        <>
          <label className="flex min-w-0 flex-col gap-1 text-sm">
            <span className="font-medium text-neutral-700 dark:text-neutral-300">
              Sequence length
            </span>
            <select
              value={seq}
              onChange={(e) => setSeq(Number(e.target.value))}
              className={`${SELECT} w-full`}
              aria-label="Sequence length"
            >
              {[1024, 2048, 4096, 8192].map((v) => (
                <option key={v} value={v}>
                  {v.toLocaleString("en-GB")}
                </option>
              ))}
            </select>
          </label>
          <label className="flex min-w-0 flex-col gap-1 text-sm">
            <span className="font-medium text-neutral-700 dark:text-neutral-300">
              Micro-batch
            </span>
            <select
              value={batch}
              onChange={(e) => setBatch(Number(e.target.value))}
              className={`${SELECT} w-full`}
              aria-label="Micro-batch (sequences)"
            >
              {[1, 2, 4, 8].map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
        </>
      )}
    </div>
  );

  const stats = (
    <p
      className="font-mono text-xs text-neutral-700 dark:text-neutral-300"
      data-testid="memory-stats"
    >
      {bd.params.toLocaleString("en-GB")} parameters × {bd.bytesPerParam} bytes
      = {fmtBytes(bd.static)} before any activation · activations{" "}
      {fmtBytes(bd.activations)} · peak {fmtBytes(bd.peak)}
    </p>
  );

  return (
    <AnimationPanel
      testId={testId}
      title="What one training step holds in memory"
      summary={`${arch.name}${model === "llama" ? `, sequence ${seq.toLocaleString("en-GB")}, micro-batch ${batch}` : ", 16 windows of 8"}, on one device. The weights, the optimiser's state and the master copy stay all the time; the forward pass saves every layer's activations; the backward pass writes the gradients and frees the activations layer by layer.`}
      stepper={st}
      stepLabel="event"
      caption={memoryCaption(s, arch)}
      visual={visual}
      stats={stats}
      equation={children}
      hl={hl}
      onEquationHover={setHover}
      params={params}
    />
  );
}
