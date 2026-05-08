"use client";

/**
 * Sampling widget — turns the final-position logits into a sampled token.
 * The user picks a mode (greedy, temperature, top-k, top-p), watches the
 * top-of-vocab bar chart, hits "Sample", and the chosen character is
 * appended to the input. Iterative sampling = press "Sample" repeatedly.
 */
import { useEffect, useMemo, useState } from "react";

import { BarChart } from "@/components/viz/BarChart";
import { usePreset } from "./usePreset";

type ForwardResponse = {
  ok: true;
  data: {
    tokens: string[];
    tokenIds: number[];
    logits: number[][]; // [seq_len, vocab_size]
  };
};
type ErrorResponse = { ok: false; error: string };

type SampleResponse = {
  ok: true;
  data: { id: number; char: string };
};

type Mode = "greedy" | "temperature" | "top-k" | "top-p";

const SEQ_LEN = 8;
const VOCAB = 64;

async function fetchForward(text: string, seed: number) {
  const res = await fetch("/api/compute/forward", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      text,
      seed,
      seqLen: SEQ_LEN,
      dModel: 16,
      dFf: 32,
      nHeads: 2,
      nBlocks: 2,
    }),
  });
  return (await res.json()) as ForwardResponse | ErrorResponse;
}

async function fetchSample(
  logits: number[],
  mode: Mode,
  temperature: number,
  k: number,
  p: number,
) {
  const res = await fetch("/api/compute/sample", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ logits, mode, temperature, k, p }),
  });
  return (await res.json()) as SampleResponse | ErrorResponse;
}

function lastUsedPosition(tokenIds: number[]): number {
  // Pad token id is 0; treat the last *non-zero* position as the prompt's end
  // so iterative sampling appends one position at a time. Falls back to 0
  // when the prompt is empty.
  for (let i = tokenIds.length - 1; i >= 0; i--) {
    if ((tokenIds[i] ?? 0) !== 0) return i;
  }
  return 0;
}

export function SamplingWidget(): JSX.Element {
  const [text, setText] = useState("the");
  const [seed, setSeed] = useState(42);
  const [mode, setMode] = useState<Mode>("temperature");
  const [temperature, setTemperature] = useState(0.7);
  const [k, setK] = useState(8);
  const [p, setP] = useState(0.9);
  const [data, setData] = useState<ForwardResponse["data"] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  usePreset(setText, setSeed);

  useEffect(() => {
    let cancelled = false;
    fetchForward(text, seed)
      .then((r) => {
        if (cancelled) return;
        if (r.ok) setData(r.data);
        else setError(r.error);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : "Failed");
      });
    return () => {
      cancelled = true;
    };
  }, [text, seed]);

  const lastIdx = data ? lastUsedPosition(data.tokenIds) : 0;
  const lastLogits = useMemo(() => {
    if (!data) return [] as number[];
    const row = data.logits[lastIdx];
    return row ? row.slice(0, VOCAB) : [];
  }, [data, lastIdx]);

  // Sort indices by logit descending — render the top-12 as a bar chart
  // labelled by token. The full V=64 chart would be visually noisy.
  const top = useMemo(() => {
    if (lastLogits.length === 0) return [];
    return lastLogits
      .map((v, i) => ({ id: i, value: v }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 12);
  }, [lastLogits]);

  async function handleSample() {
    if (!data || lastLogits.length === 0 || busy) return;
    setBusy(true);
    setError(null);
    try {
      const r = await fetchSample(lastLogits, mode, temperature, k, p);
      if (r.ok) {
        setText((prev) => (prev + r.data.char).slice(0, SEQ_LEN * 2));
      } else {
        setError(r.error);
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="my-8 rounded-lg border border-neutral-200 bg-neutral-50 p-4 dark:border-neutral-800 dark:bg-neutral-900">
      <div className="grid gap-3 sm:grid-cols-3">
        <label className="flex flex-col gap-1 text-sm sm:col-span-2">
          <span className="font-medium text-neutral-700 dark:text-neutral-300">
            Prompt:
          </span>
          <input
            type="text"
            value={text}
            maxLength={SEQ_LEN * 2}
            onChange={(e) => setText(e.target.value)}
            aria-label="Sampling widget prompt"
            className="focus-ring rounded border border-neutral-300 bg-white px-2 py-1 font-mono text-sm dark:border-neutral-700 dark:bg-neutral-950"
          />
        </label>
        <label className="flex flex-col gap-1 text-sm">
          <span className="font-medium text-neutral-700 dark:text-neutral-300">
            Seed:
          </span>
          <input
            type="number"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value || 0))}
            aria-label="Sampling widget seed"
            className="focus-ring rounded border border-neutral-300 bg-white px-2 py-1 font-mono text-sm dark:border-neutral-700 dark:bg-neutral-950"
          />
        </label>
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <div>
          <span className="text-xs font-medium uppercase tracking-widest text-neutral-500">
            Mode
          </span>
          <div
            role="radiogroup"
            aria-label="Sampling mode"
            className="mt-1 inline-flex flex-wrap gap-1 rounded border border-neutral-300 bg-white p-1 text-xs dark:border-neutral-700 dark:bg-neutral-950"
          >
            {(["greedy", "temperature", "top-k", "top-p"] as Mode[]).map(
              (m) => (
                <button
                  key={m}
                  type="button"
                  role="radio"
                  aria-checked={mode === m}
                  onClick={() => setMode(m)}
                  className={`focus-ring rounded px-2 py-1 font-mono ${
                    mode === m
                      ? "bg-accent text-accent-fg"
                      : "text-neutral-600 hover:bg-neutral-100 dark:text-neutral-300 dark:hover:bg-neutral-800"
                  }`}
                >
                  {m}
                </button>
              ),
            )}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3 text-xs">
          {mode !== "greedy" && (
            <NumberInput
              label="τ"
              value={temperature}
              step={0.1}
              min={0.1}
              max={3}
              onChange={setTemperature}
            />
          )}
          {mode === "top-k" && (
            <NumberInput
              label="k"
              value={k}
              step={1}
              min={1}
              max={32}
              onChange={(v) => setK(Math.round(v))}
            />
          )}
          {mode === "top-p" && (
            <NumberInput
              label="p"
              value={p}
              step={0.05}
              min={0.05}
              max={1}
              onChange={setP}
            />
          )}
        </div>
      </div>

      <div className="mt-4 flex items-center gap-3">
        <button
          type="button"
          onClick={handleSample}
          disabled={!data || busy}
          aria-label="Sample the next token"
          className="focus-ring rounded bg-accent px-4 py-1.5 text-sm font-medium text-accent-fg hover:opacity-90 disabled:opacity-40"
        >
          Sample next token
        </button>
        <button
          type="button"
          onClick={() => setText("the")}
          className="focus-ring rounded border border-neutral-300 px-3 py-1.5 text-xs text-neutral-600 hover:bg-neutral-100 dark:border-neutral-700 dark:text-neutral-300 dark:hover:bg-neutral-800"
        >
          Reset
        </button>
      </div>

      {error && (
        <p
          role="alert"
          className="mt-3 rounded border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-800 dark:border-red-800 dark:bg-red-950/40 dark:text-red-200"
        >
          {error}
        </p>
      )}

      {data && top.length > 0 && (
        <div className="mt-6">
          <h4 className="text-sm font-semibold text-neutral-800 dark:text-neutral-200">
            Top-12 logits at position {lastIdx}
          </h4>
          <p className="text-xs text-neutral-500">
            Higher = more likely next token under temperature 1.
          </p>
          <ol className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-3">
            {top.map((row) => (
              <li
                key={row.id}
                className="flex items-center gap-2 rounded bg-white p-2 text-xs ring-1 ring-neutral-200 dark:bg-neutral-950 dark:ring-neutral-800"
              >
                <span className="w-7 shrink-0 font-mono text-neutral-800 dark:text-neutral-200">
                  {row.id === 36
                    ? "␣"
                    : row.id === 47
                      ? "↵"
                      : (data.tokens[row.id] ??
                        // Show the actual character for this id from the alphabet.
                        // For ids beyond the printable subset, fall back to id.
                        (row.id >= 0 ? String(row.id) : "?"))}
                </span>
                <span className="flex-1">
                  <BarChart values={[row.value]} width={120} rowHeight={12} />
                </span>
                <span className="font-mono text-[10px] text-neutral-500">
                  {row.value.toFixed(2)}
                </span>
              </li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}

function NumberInput({
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
    <label className="flex items-center gap-1">
      <span className="font-medium text-neutral-700 dark:text-neutral-300">
        {label}
      </span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value || 0))}
        aria-label={`Sampling ${label}`}
        className="focus-ring w-16 rounded border border-neutral-300 bg-white px-1.5 py-0.5 font-mono dark:border-neutral-700 dark:bg-neutral-950"
      />
    </label>
  );
}
