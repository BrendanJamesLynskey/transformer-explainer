"use client";

/**
 * Chapter 11's calculator: data parallelism and ZeRO for LLaMA 7B on H100s
 * (systems.ts): the model states each GPU holds at each ZeRO stage, the
 * bytes it sends per step, and the step time with the gradient sync either
 * waiting for the backward pass to finish or overlapping it. Four
 * sequences of 4,096 tokens per GPU per step; η is the systems model's
 * fitted efficiency (results.md §11a).
 */
import { useState } from "react";

import {
  PHONE_BELOW,
  useContainerWidth,
} from "@/components/viz/useContainerWidth";
import { fmtBytes } from "@/lib/training/memory";
import {
  HARDWARE,
  collective,
  modelFlopsPerToken,
  nParams,
  zeroMemory,
  type SysArch,
} from "@/lib/training/systems";
import { OKABE_ITO } from "@/lib/viz/palette";

const SELECT =
  "focus-ring h-11 w-full rounded border border-neutral-300 bg-white px-2 text-sm dark:border-neutral-700 dark:bg-neutral-950";

export default function DataParallelCalculator({
  arch,
  eta,
}: {
  arch: SysArch;
  eta: number;
}): JSX.Element {
  const [d, setD] = useState(64);
  const [stage, setStage] = useState(0);
  const [overlap, setOverlap] = useState(false);
  const hw = HARDWARE.h100;
  const N = nParams(arch);
  const z = zeroMemory(N, d, stage);
  const tokens = 4 * arch.s;
  const compute = (modelFlopsPerToken(arch) * tokens) / (hw.peak * eta);
  const inServer = d <= hw.perNode;
  const bw = inServer ? hw.nvlink : hw.net;
  const alpha = inServer ? hw.nvlinkAlpha : hw.netAlpha;
  const comm =
    d > 1
      ? collective("allreduce", 2 * N, d, bw, alpha).time *
        (stage >= 3 ? 1.5 : 1)
      : 0;
  const exposed = overlap ? Math.max(0, comm - (2 / 3) * compute) : comm;
  const step = compute + exposed;
  const busy = compute / step;
  const memMax = 16 * N;
  const parts = [
    { k: "weights", v: z.weights, c: OKABE_ITO.blue },
    { k: "gradients", v: z.grads, c: OKABE_ITO.orange },
    {
      k: "optimiser (Adam m, v, FP32 master)",
      v: z.optimizer,
      c: OKABE_ITO.green,
    },
  ];
  const box = useContainerWidth();
  const W = box.width > 0 && box.width < PHONE_BELOW ? 360 : 640;
  // labels at least ~12 px on a phone (the drawing is narrower than 360 there)
  const FS = W === 360 ? 15 : 13;
  const barX = 0;
  return (
    <figure
      className="my-6 rounded-lg border border-neutral-200 bg-neutral-50 p-4 dark:border-neutral-800 dark:bg-neutral-900"
      data-testid="dp-calculator"
    >
      <figcaption className="font-semibold text-neutral-900 dark:text-neutral-100">
        Data parallelism and ZeRO: {arch.name} on H100s
      </figcaption>
      <div className="mt-3 grid min-w-0 gap-3 sm:grid-cols-3">
        <label className="flex min-w-0 flex-col gap-1 text-sm">
          <span className="font-medium text-neutral-700 dark:text-neutral-300">
            Data-parallel GPUs
          </span>
          <select
            value={d}
            onChange={(e) => setD(Number(e.target.value))}
            className={SELECT}
            aria-label="Data-parallel GPUs"
          >
            {[1, 8, 64, 512].map((v) => (
              <option key={v} value={v}>
                {v} {v <= 8 ? "(one server, NVLink)" : "(network)"}
              </option>
            ))}
          </select>
        </label>
        <label className="flex min-w-0 flex-col gap-1 text-sm">
          <span className="font-medium text-neutral-700 dark:text-neutral-300">
            ZeRO stage
          </span>
          <select
            value={stage}
            onChange={(e) => setStage(Number(e.target.value))}
            className={SELECT}
            aria-label="ZeRO stage"
          >
            <option value={0}>0: every GPU holds everything</option>
            <option value={1}>1: shard the optimiser states</option>
            <option value={2}>2: + the gradients</option>
            <option value={3}>3 / FSDP: + the weights</option>
          </select>
        </label>
        <label className="flex min-w-0 flex-col gap-1 text-sm">
          <span className="font-medium text-neutral-700 dark:text-neutral-300">
            Gradient sync
          </span>
          <select
            value={overlap ? "overlap" : "after"}
            onChange={(e) => setOverlap(e.target.value === "overlap")}
            className={SELECT}
            aria-label="Gradient sync timing"
          >
            <option value="after">after the backward pass</option>
            <option value="overlap">overlapped with the backward pass</option>
          </select>
        </label>
      </div>
      <div ref={box.ref} className="min-w-0">
        <svg
          viewBox={`0 0 ${W} 96`}
          className="mt-4 w-full text-neutral-800 dark:text-neutral-200"
          role="img"
          aria-label={`Model states per GPU ${fmtBytes(z.total)}; step ${step.toFixed(2)} s, MFU ${(100 * busy * eta).toFixed(0)}%`}
        >
          <text x={0} y={14} fontSize={FS} fill="currentColor">
            model states per GPU: {fmtBytes(z.total)} (80 GB of HBM)
          </text>
          {(() => {
            let acc = 0;
            return parts.map((p) => {
              const w = (W * p.v) / memMax;
              const x = barX + (W * acc) / memMax;
              acc += p.v;
              return (
                <rect key={p.k} x={x} y={22} width={w} height={22} fill={p.c} />
              );
            });
          })()}
          <line
            x1={(W * hw.hbm) / memMax}
            x2={(W * hw.hbm) / memMax}
            y1={18}
            y2={48}
            stroke="currentColor"
            strokeWidth={2}
            strokeDasharray="4 3"
          />
          <text x={0} y={68} fontSize={FS} fill="currentColor">
            step: compute {compute.toFixed(2)} s + exposed sync{" "}
            {exposed.toFixed(2)} s
          </text>
          <rect
            x={0}
            y={74}
            width={(W * compute) / (compute + comm)}
            height={16}
            fill={OKABE_ITO.purple}
          />
          <rect
            x={(W * compute) / (compute + comm)}
            y={74}
            width={(W * exposed) / (compute + comm)}
            height={16}
            fill={OKABE_ITO.vermillion}
          />
        </svg>
      </div>
      <p className="mt-2 text-xs text-neutral-600 dark:text-neutral-400">
        {parts.map((p, i) => (
          <span key={p.k}>
            {i ? " · " : ""}
            <span style={{ color: p.c }} aria-hidden>
              ■
            </span>{" "}
            {p.k} {fmtBytes(p.v)}
          </span>
        ))}
        {" · "}dashed line: 80 GB
      </p>
      <p
        className="mt-1 font-mono text-xs text-neutral-700 dark:text-neutral-300"
        data-testid="dp-stats"
      >
        each GPU sends {fmtBytes(z.wire)} per step over{" "}
        {inServer ? "NVLink" : "the network"} ({fmtBytes(bw)}/s) · all-reduce{" "}
        {comm.toFixed(2)} s · step {step.toFixed(2)} s · MFU{" "}
        {(100 * busy * eta).toFixed(0)}% ({(tokens * d).toLocaleString("en-GB")}{" "}
        tokens per step)
      </p>
    </figure>
  );
}
