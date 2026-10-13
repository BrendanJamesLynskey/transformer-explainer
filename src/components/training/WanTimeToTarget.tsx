"use client";

/**
 * Chapter 13's step-time model: how long each way of syncing takes to reach
 * a validation loss when the workers share a slow link. Steps to the target
 * are the toy's measurements (results.md §13, mean of three seeds); the
 * bytes are its bytes per parameter; the compute is assumed (an H100 at 40%
 * MFU, the results site's assumption). Everything about other model sizes
 * is extrapolation, and labelled so.
 */
import { useState } from "react";

import {
  PHONE_BELOW,
  useContainerWidth,
} from "@/components/viz/useContainerWidth";

import { fmtBytes } from "@/lib/training/memory";
import { HARDWARE, md1Wait } from "@/lib/training/systems";
import { OKABE_ITO } from "@/lib/viz/palette";

const SELECT =
  "focus-ring h-11 w-full rounded border border-neutral-300 bg-white px-2 text-sm dark:border-neutral-700 dark:bg-neutral-950";

export type WanRow = {
  name: string;
  method: string;
  H: number;
  evalSteps: number[];
  meanEvals: (number | null)[];
  wirePerParam: number;
};

const SIZES = [
  {
    key: "toy",
    label: "The Decoder's model (5,376), 32 tokens per worker per step",
    N: 5376,
    tokens: 32,
  },
  {
    key: "cit",
    label:
      "15.8M parameters, 2,048 tokens per worker per step (the results site's experiment)",
    N: 15833472,
    tokens: 2048,
  },
  {
    key: "1b",
    label: "1B parameters, 2,048 tokens per worker per step",
    N: 1e9,
    tokens: 2048,
  },
] as const;
const MFU = 0.4;
const COLOUR: Record<string, string> = {
  dense: OKABE_ITO.purple,
  topk: OKABE_ITO.orange,
  demo: OKABE_ITO.vermillion,
  diloco: OKABE_ITO.blue,
};
const LABEL: Record<string, string> = {
  dense: "dense AdamW (all-reduce)",
  topk: "top-k + error feedback",
  demo: "DeMo",
  diloco: "DiLoCo",
};

export function stepsTo(row: WanRow, target: number): number | null {
  for (let i = 0; i < row.meanEvals.length; i++) {
    const v = row.meanEvals[i];
    if (v !== null && v !== undefined && v <= target) return row.evalSteps[i]!;
  }
  return null;
}

function fmtTime(s: number): string {
  if (s < 0.01) return `${(s * 1000).toFixed(2)} ms`;
  if (s < 1) return `${(s * 1000).toFixed(0)} ms`;
  if (s < 120) return `${s.toFixed(1)} s`;
  if (s < 7200) return `${(s / 60).toFixed(1)} min`;
  return `${(s / 3600).toFixed(1)} h`;
}

export default function WanTimeToTarget({
  rows,
}: {
  rows: WanRow[];
}): JSX.Element {
  const [logGbps, setLogGbps] = useState(1); // 10 Gb/s
  const [target, setTarget] = useState(2.6);
  const [size, setSize] = useState<(typeof SIZES)[number]["key"]>("cit");
  const [load, setLoad] = useState(0);
  const gbps = Math.pow(10, logGbps);
  const bw = (gbps * 1e9) / 8;
  const sz = SIZES.find((s) => s.key === size)!;
  const compute = (6 * sz.N * sz.tokens) / (HARDWARE.h100.peak * MFU);
  const results = rows.map((r) => {
    const steps = stepsTo(r, target);
    const bytes = r.wirePerParam * sz.N;
    const wait =
      load > 0
        ? md1Wait(load, 1e6 / bw) / (r.method === "diloco" ? r.H : 1)
        : 0;
    const comm = bytes / bw + wait;
    const step = compute + comm;
    return {
      r,
      steps,
      bytes,
      comm,
      step,
      total: steps === null ? null : steps * step,
    };
  });
  const reached = results.filter((x) => x.total !== null);
  const maxT = Math.max(1e-9, ...reached.map((x) => x.total!));
  const best = reached.reduce<(typeof reached)[number] | null>(
    (a, b) => (a === null || b.total! < a.total! ? b : a),
    null,
  );
  const box = useContainerWidth();
  const phone = box.width > 0 && box.width < PHONE_BELOW;
  const W = phone ? 360 : 640;
  const rowH = phone ? 46 : 30;
  const labelW = phone ? 0 : 190;
  const fsz = phone ? 14 : 13;
  return (
    <figure
      className="my-6 rounded-lg border border-neutral-200 bg-neutral-50 p-4 dark:border-neutral-800 dark:bg-neutral-900"
      data-testid="wan-widget"
    >
      <figcaption className="font-semibold text-neutral-900 dark:text-neutral-100">
        Time to a validation loss of {target} over a{" "}
        {gbps < 1 ? gbps.toFixed(2) : gbps.toFixed(gbps < 10 ? 1 : 0)} Gb/s link
      </figcaption>
      <div className="mt-3 grid min-w-0 gap-3 sm:grid-cols-2">
        <label className="flex min-w-0 flex-col gap-1 text-sm sm:col-span-2">
          <span className="font-medium text-neutral-700 dark:text-neutral-300">
            Link bandwidth between the workers:{" "}
            {gbps < 1 ? gbps.toFixed(2) : gbps.toFixed(gbps < 10 ? 1 : 0)} Gb/s
          </span>
          <input
            type="range"
            min={-1}
            max={2.6}
            step={0.05}
            value={logGbps}
            onChange={(e) => setLogGbps(Number(e.target.value))}
            className="focus-ring h-11 w-full accent-indigo-600"
            aria-label="Link bandwidth (log scale)"
            data-testid="wan-slider"
          />
        </label>
        <label className="flex min-w-0 flex-col gap-1 text-sm sm:col-span-2">
          <span className="font-medium text-neutral-700 dark:text-neutral-300">
            Model (extrapolated beyond the toy)
          </span>
          <select
            value={size}
            onChange={(e) =>
              setSize(e.target.value as (typeof SIZES)[number]["key"])
            }
            className={SELECT}
            aria-label="Model size"
          >
            {SIZES.map((s) => (
              <option key={s.key} value={s.key}>
                {s.label}
              </option>
            ))}
          </select>
        </label>
        <label className="flex min-w-0 flex-col gap-1 text-sm">
          <span className="font-medium text-neutral-700 dark:text-neutral-300">
            Target validation loss
          </span>
          <select
            value={target}
            onChange={(e) => setTarget(Number(e.target.value))}
            className={SELECT}
            aria-label="Target validation loss"
          >
            {[2.8, 2.6, 2.4].map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </label>
        <label className="flex min-w-0 flex-col gap-1 text-sm">
          <span className="font-medium text-neutral-700 dark:text-neutral-300">
            Other traffic on the link
          </span>
          <select
            value={load}
            onChange={(e) => setLoad(Number(e.target.value))}
            className={SELECT}
            aria-label="Background load"
          >
            <option value={0}>none</option>
            <option value={0.5}>50% busy (1 MB transfers)</option>
            <option value={0.9}>90% busy (1 MB transfers)</option>
          </select>
        </label>
      </div>
      <div ref={box.ref} className="min-w-0">
        <svg
          viewBox={`0 0 ${W} ${results.length * rowH + 10}`}
          className="mt-4 w-full text-neutral-800 dark:text-neutral-200"
          role="img"
          aria-label={`Time to validation loss ${target}: ${results
            .map(
              (x) =>
                `${x.r.name} ${x.total === null ? "not reached" : fmtTime(x.total)}`,
            )
            .join(", ")}`}
        >
          {results.map((x, i) => {
            const y = 4 + i * rowH;
            const barY = phone ? y + fsz + 4 : y + 4;
            const w =
              x.total === null
                ? 0
                : ((W - labelW - (phone ? 90 : 70)) * x.total) / maxT;
            return (
              <g key={x.r.name} data-testid={`wan-row-${x.r.method}`}>
                <text
                  x={phone ? 0 : labelW - 6}
                  y={phone ? y + fsz : y + 17}
                  fontSize={fsz}
                  textAnchor={phone ? "start" : "end"}
                  fill="currentColor"
                >
                  {LABEL[x.r.method]}
                  {x.r.method === "diloco" ? `, H = ${x.r.H}` : ""}
                </text>
                <rect
                  x={labelW}
                  y={barY}
                  width={Math.max(1, w)}
                  height={18}
                  fill={COLOUR[x.r.method]}
                />
                <text
                  x={labelW + w + 6}
                  y={barY + 14}
                  fontSize={fsz}
                  fill="currentColor"
                >
                  {x.total === null
                    ? phone
                      ? "not reached"
                      : "not reached in 2,000 steps"
                    : fmtTime(x.total)}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
      <p
        className="mt-2 font-mono text-xs text-neutral-700 dark:text-neutral-300"
        data-testid="wan-stats"
      >
        compute per step {fmtTime(compute)} (H100 at {MFU * 100}% MFU, assumed)
        ·{" "}
        {results
          .map(
            (x) =>
              `${x.r.name}: ${fmtBytes(x.bytes)}/step on the wire, ${x.steps ?? "–"} steps`,
          )
          .join(" · ")}
        {best ? ` · fastest: ${best.r.name}` : ""}
      </p>
      <p className="mt-1 text-xs text-neutral-600 dark:text-neutral-400">
        Steps to the target: the toy&apos;s measured mean over three seeds
        (first evaluation, every 50 steps, at or below it). Bytes: the
        toy&apos;s bytes per parameter per step, times the model&apos;s
        parameters; 4 workers. Bandwidth only (link latency ignored); compute
        and communication do not overlap. For other model sizes this is an
        extrapolation: it assumes the toy&apos;s steps to the target hold, which
        nothing here tested.
      </p>
    </figure>
  );
}
