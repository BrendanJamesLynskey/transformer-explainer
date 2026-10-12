"use client";

/**
 * Chapter 12's explorer: choose a 3-D layout (tensor × pipeline × data
 * parallel) for a large model and read off, from the systems model
 * (systems.ts, validated against published runs in results.md §11a), the
 * memory on the busiest GPU and where one step's time goes. η is the
 * model's single fitted constant.
 */
import { useState } from "react";

import {
  PHONE_BELOW,
  useContainerWidth,
} from "@/components/viz/useContainerWidth";
import { fmtBytes } from "@/lib/training/memory";
import {
  HARDWARE,
  deviceMemory,
  gptArch,
  stepTime,
  type Recompute,
  type SysArch,
} from "@/lib/training/systems";
import { OKABE_ITO } from "@/lib/viz/palette";

const SELECT =
  "focus-ring h-11 w-full rounded border border-neutral-300 bg-white px-2 text-sm dark:border-neutral-700 dark:bg-neutral-950";

type Preset = { arch: SysArch; B: number; label: string };

const PRESETS: Record<string, Preset> = {
  gpt3: {
    arch: gptArch("GPT-3 175B", 96, 12288, 96, 2048),
    B: 1536,
    label: "GPT-3 175B (96 layers, width 12,288), 1,536 sequences of 2,048",
  },
  llama: {
    arch: gptArch("Llama 3 405B", 126, 16384, 128, 8192, {
      V: 128000,
      f: 53248,
      kv: 8,
      swiglu: true,
      tied: false,
    }),
    B: 2048,
    label: "Llama 3 405B (126 layers), 2,048 sequences of 8,192",
  },
  t1: {
    arch: gptArch("GPT 1T", 128, 25600, 160, 2048),
    B: 3072,
    label: "Megatron's 1T GPT (128 layers, width 25,600), 3,072 of 2,048",
  },
};

const T_OPTS = [1, 2, 4, 8];
const P_OPTS = [1, 2, 4, 8, 16, 32, 64];
const D_OPTS = [1, 2, 4, 8, 16, 32, 64, 128];

function Select({
  label,
  value,
  options,
  onChange,
  fmt = (v: number) => String(v),
}: {
  label: string;
  value: number;
  options: number[];
  onChange: (v: number) => void;
  fmt?: (v: number) => string;
}): JSX.Element {
  return (
    <label className="flex min-w-0 flex-col gap-1 text-sm">
      <span className="font-medium text-neutral-700 dark:text-neutral-300">
        {label}
      </span>
      <select
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className={SELECT}
        aria-label={label}
      >
        {options.map((v) => (
          <option key={v} value={v}>
            {fmt(v)}
          </option>
        ))}
      </select>
    </label>
  );
}

export default function LayoutExplorer({ eta }: { eta: number }): JSX.Element {
  const [preset, setPreset] = useState("gpt3");
  const [hwKey, setHw] = useState<"a100" | "h100">("a100");
  const [t, setT] = useState(8);
  const [p, setP] = useState(8);
  const [d, setD] = useState(16);
  const [v, setV] = useState(1);
  const [recompute, setRecompute] = useState<Recompute>("selective");
  const [sp, setSp] = useState(true);
  const [zero, setZero] = useState(1);
  const { arch, B } = PRESETS[preset]!;
  const hw = HARDWARE[hwKey];
  const lay = { t, p, d, v, sp, recompute, zero, b: 1, B };
  const m = Math.floor(B / d);
  const mem = deviceMemory(arch, { ...lay, m });
  const st = stepTime(arch, lay, hw, eta);
  const layersOk = arch.L % (p * v) === 0;
  const fits = mem.total <= hw.hbm;
  const box = useContainerWidth();
  const W = box.width > 0 && box.width < PHONE_BELOW ? 360 : 640;
  // labels at least ~12 px on a phone (the drawing is narrower than 360 there)
  const FS = W === 360 ? 15 : 13;
  const memMax = Math.max(hw.hbm * 1.5, mem.total);
  const memParts = [
    { k: "weights", v: mem.weights, c: OKABE_ITO.blue },
    { k: "gradients", v: mem.grads, c: OKABE_ITO.orange },
    { k: "optimiser", v: mem.optimizer, c: OKABE_ITO.green },
    { k: "activations", v: mem.activations, c: OKABE_ITO.purple },
  ];
  const timeParts = [
    { k: "compute", v: st.compute, c: OKABE_ITO.green },
    { k: "tensor all-reduces", v: st.tensor, c: OKABE_ITO.sky },
    { k: "bubble", v: st.bubble, c: "#7f7f7f" },
    { k: "pipeline sends", v: st.p2p, c: OKABE_ITO.yellow },
    { k: "gradient sync", v: st.data, c: OKABE_ITO.vermillion },
  ];
  const bar = (
    parts: { k: string; v: number; c: string }[],
    max: number,
    y: number,
  ) => {
    let acc = 0;
    return parts.map((q) => {
      const x = (W * acc) / max;
      acc += q.v;
      return (
        <rect
          key={q.k}
          x={x}
          y={y}
          width={(W * q.v) / max}
          height={22}
          fill={q.c}
        />
      );
    });
  };
  return (
    <figure
      className="my-6 rounded-lg border border-neutral-200 bg-neutral-50 p-4 dark:border-neutral-800 dark:bg-neutral-900"
      data-testid="layout-explorer"
    >
      <figcaption className="font-semibold text-neutral-900 dark:text-neutral-100">
        Choose a layout: {st.gpus.toLocaleString("en-GB")} GPUs = {t} tensor ×{" "}
        {p} pipeline × {d} data
      </figcaption>
      <div className="mt-3 grid min-w-0 gap-3 sm:grid-cols-2">
        <label className="flex min-w-0 flex-col gap-1 text-sm sm:col-span-2">
          <span className="font-medium text-neutral-700 dark:text-neutral-300">
            Model and batch
          </span>
          <select
            value={preset}
            onChange={(e) => setPreset(e.target.value)}
            className={SELECT}
            aria-label="Model and batch"
          >
            {Object.entries(PRESETS).map(([k, x]) => (
              <option key={k} value={k}>
                {x.label}
              </option>
            ))}
          </select>
        </label>
        <label className="flex min-w-0 flex-col gap-1 text-sm">
          <span className="font-medium text-neutral-700 dark:text-neutral-300">
            GPU
          </span>
          <select
            value={hwKey}
            onChange={(e) => setHw(e.target.value as "a100" | "h100")}
            className={SELECT}
            aria-label="GPU"
          >
            <option value="a100">A100 80GB</option>
            <option value="h100">H100 80GB</option>
          </select>
        </label>
        <Select
          label="Tensor parallel t (in a server)"
          value={t}
          options={T_OPTS}
          onChange={setT}
        />
        <Select
          label="Pipeline stages p"
          value={p}
          options={P_OPTS}
          onChange={setP}
        />
        <Select
          label="Data parallel d"
          value={d}
          options={D_OPTS}
          onChange={setD}
        />
        <Select
          label="Chunks per stage v (interleaving)"
          value={v}
          options={[1, 2, 4]}
          onChange={setV}
        />
        <label className="flex min-w-0 flex-col gap-1 text-sm">
          <span className="font-medium text-neutral-700 dark:text-neutral-300">
            Activation recomputation
          </span>
          <select
            value={recompute}
            onChange={(e) => setRecompute(e.target.value as Recompute)}
            className={SELECT}
            aria-label="Activation recomputation"
          >
            <option value="none">none</option>
            <option value="selective">selective</option>
            <option value="full">full</option>
          </select>
        </label>
        <Select
          label="Sequence parallel"
          value={sp ? 1 : 0}
          options={[1, 0]}
          onChange={(x) => setSp(x === 1)}
          fmt={(x) => (x ? "on" : "off")}
        />
        <Select
          label="ZeRO stage over d"
          value={zero}
          options={[0, 1, 2, 3]}
          onChange={setZero}
        />
      </div>
      <div ref={box.ref} className="min-w-0">
        <svg
          viewBox={`0 0 ${W} 110`}
          className="mt-4 w-full text-neutral-800 dark:text-neutral-200"
          role="img"
          aria-label={`Busiest GPU holds ${fmtBytes(mem.total)}; step ${st.time.toFixed(1)} s; MFU ${(100 * st.mfu).toFixed(1)}%`}
        >
          <text x={0} y={14} fontSize={FS} fill="currentColor">
            busiest GPU: {fmtBytes(mem.total)}{" "}
            {fits ? "(fits in 80 GB)" : "(over 80 GB: does not fit)"}
          </text>
          {bar(memParts, memMax, 20)}
          <line
            x1={(W * hw.hbm) / memMax}
            x2={(W * hw.hbm) / memMax}
            y1={16}
            y2={46}
            stroke="currentColor"
            strokeWidth={2}
            strokeDasharray="4 3"
          />
          <text x={0} y={72} fontSize={FS} fill="currentColor">
            one step: {st.time.toFixed(2)} s · MFU {(100 * st.mfu).toFixed(1)}%
          </text>
          {bar(timeParts, st.time, 80)}
        </svg>
      </div>
      <p className="mt-2 text-xs text-neutral-600 dark:text-neutral-400">
        {[...memParts, ...timeParts].map((q, i) => (
          <span key={q.k}>
            {i ? " · " : ""}
            <span style={{ color: q.c }} aria-hidden>
              ■
            </span>{" "}
            {q.k} {i < memParts.length ? fmtBytes(q.v) : `${q.v.toFixed(2)} s`}
          </span>
        ))}
      </p>
      <p
        className="mt-1 font-mono text-xs text-neutral-700 dark:text-neutral-300"
        data-testid="layout-stats"
      >
        {m} micro-batches of 1 sequence per pipeline · bubble (p − 1)/(v·m) ={" "}
        {((100 * (p - 1)) / (v * m)).toFixed(1)}% ·{" "}
        {st.tokensPerSecond.toLocaleString("en-GB", {
          maximumFractionDigits: 0,
        })}{" "}
        tokens/s
        {!layersOk &&
          ` · ${arch.L} layers do not split evenly into ${p * v} chunks (the model rounds)`}
        {m < 1 && " · fewer sequences than data-parallel groups"}
      </p>
    </figure>
  );
}
