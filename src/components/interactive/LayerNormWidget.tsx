"use client";

/**
 * LayerNorm widget — a self-contained sandbox for the explainer page. The
 * input is a fixed seven-element vector with a deliberately broad spread
 * so the effect of γ/β sliders is visible. Everything runs in the
 * browser; no API call.
 */
import { useMemo, useState } from "react";

import { BarChart } from "@/components/viz/BarChart";
import { layernorm } from "@/lib/transformer/layernorm";

const INPUT = [-3, -1, 0, 1, 2, 5, 7];

export function LayerNormWidget(): JSX.Element {
  const [gamma, setGamma] = useState(1);
  const [beta, setBeta] = useState(0);

  const output = useMemo(() => {
    const g = new Array<number>(INPUT.length).fill(gamma);
    const b = new Array<number>(INPUT.length).fill(beta);
    return layernorm(INPUT, g, b);
  }, [gamma, beta]);

  return (
    <div className="my-8 rounded-lg border border-neutral-200 bg-neutral-50 p-4 dark:border-neutral-800 dark:bg-neutral-900">
      <div className="grid gap-4 sm:grid-cols-2">
        <Slider
          label="γ (scale)"
          value={gamma}
          onChange={setGamma}
          min={0}
          max={3}
          step={0.1}
        />
        <Slider
          label="β (shift)"
          value={beta}
          onChange={setBeta}
          min={-3}
          max={3}
          step={0.1}
        />
      </div>

      <div className="mt-6 grid gap-6 sm:grid-cols-2">
        <Panel title="Input" subtitle="A fixed vector with a wide spread.">
          <BarChart values={INPUT} width={260} rowHeight={14} />
        </Panel>
        <Panel
          title="γ · (x − μ)/σ + β"
          subtitle={`After LayerNorm with γ=${gamma.toFixed(1)}, β=${beta.toFixed(1)}.`}
        >
          <BarChart values={output} width={260} rowHeight={14} />
        </Panel>
      </div>

      <p className="mt-4 text-xs text-neutral-500">
        Notice how γ scales the spread and β shifts the centre. With γ=1 and β=0
        the output has zero mean and (very nearly) unit variance.
      </p>
    </div>
  );
}

function Slider({
  label,
  value,
  onChange,
  min,
  max,
  step,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
}): JSX.Element {
  return (
    <label className="flex flex-col gap-1 text-sm">
      <span className="flex justify-between font-medium text-neutral-700 dark:text-neutral-300">
        {label}
        <span className="font-mono text-xs text-neutral-500">
          {value.toFixed(1)}
        </span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        aria-label={label}
        className="focus-ring accent-accent"
      />
    </label>
  );
}

function Panel({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}): JSX.Element {
  return (
    <div>
      <h4 className="text-sm font-semibold text-neutral-800 dark:text-neutral-200">
        {title}
      </h4>
      {subtitle && <p className="text-xs text-neutral-500">{subtitle}</p>}
      <div className="mt-2">{children}</div>
    </div>
  );
}
