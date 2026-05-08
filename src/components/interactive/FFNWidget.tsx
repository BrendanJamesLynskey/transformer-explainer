"use client";

/**
 * FFN widget — runs the position-wise feed-forward network and visualises
 * the up-projection, the GELU activation, and the down-projection back to
 * `d_model`. The user picks which sequence position to inspect.
 */
import { useEffect, useState } from "react";

import { BarChart } from "@/components/viz/BarChart";
import { usePreset } from "./usePreset";

type FFNResponse = {
  ok: true;
  data: {
    tokens: string[];
    input: number[][]; // LN2(h)        [S, d_model]
    pre: number[][]; //   x · W1 + b1   [S, d_ff]
    act: number[][]; //   GELU(pre)     [S, d_ff]
    output: number[][]; //  act · W2 + b2  [S, d_model]
    dModel: number;
    dFf: number;
  };
};

type ErrorResponse = { ok: false; error: string };

const SEQ_LEN = 8;
const D_MODEL = 16;
const D_FF = 32;

async function fetchFFN(text: string, seed: number) {
  const res = await fetch("/api/compute/ffn", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      text,
      seed,
      seqLen: SEQ_LEN,
      dModel: D_MODEL,
      dFf: D_FF,
      nHeads: 2,
    }),
  });
  return (await res.json()) as FFNResponse | ErrorResponse;
}

function visibleToken(t: string): string {
  return t === "\n" ? "↵" : t === " " ? "␣" : t;
}

export function FFNWidget(): JSX.Element {
  const [text, setText] = useState("hello");
  const [seed, setSeed] = useState(42);
  const [position, setPosition] = useState(0);
  const [data, setData] = useState<FFNResponse["data"] | null>(null);
  const [error, setError] = useState<string | null>(null);

  usePreset(setText, setSeed);

  useEffect(() => {
    let cancelled = false;
    fetchFFN(text, seed)
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

  const inputRow = data?.input[position] ?? [];
  const preRow = data?.pre[position] ?? [];
  const actRow = data?.act[position] ?? [];
  const outRow = data?.output[position] ?? [];

  return (
    <div className="my-8 rounded-lg border border-neutral-200 bg-neutral-50 p-4 dark:border-neutral-800 dark:bg-neutral-900">
      <div className="grid gap-3 sm:grid-cols-3">
        <label className="flex flex-col gap-1 text-sm sm:col-span-2">
          <span className="font-medium text-neutral-700 dark:text-neutral-300">
            Input text:
          </span>
          <input
            type="text"
            value={text}
            maxLength={SEQ_LEN * 2}
            onChange={(e) => setText(e.target.value)}
            aria-label="FFN widget input text"
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
            aria-label="FFN widget seed"
            className="focus-ring rounded border border-neutral-300 bg-white px-2 py-1 font-mono text-sm dark:border-neutral-700 dark:bg-neutral-950"
          />
        </label>
      </div>

      {data && (
        <>
          <div className="mt-4 flex items-center gap-2 text-sm">
            <span className="font-medium text-neutral-700 dark:text-neutral-300">
              Position:
            </span>
            <div
              role="group"
              aria-label="Pick a sequence position"
              className="inline-flex flex-wrap gap-1 rounded border border-neutral-300 bg-white p-1 text-xs dark:border-neutral-700 dark:bg-neutral-950"
            >
              {data.tokens.map((tok, i) => (
                <button
                  key={i}
                  type="button"
                  aria-pressed={position === i}
                  onClick={() => setPosition(i)}
                  className={`focus-ring rounded px-2 py-1 font-mono ${
                    position === i
                      ? "bg-accent text-accent-fg"
                      : "text-neutral-600 hover:bg-neutral-100 dark:text-neutral-300 dark:hover:bg-neutral-800"
                  }`}
                >
                  {i}·{visibleToken(tok)}
                </button>
              ))}
            </div>
          </div>

          <div className="mt-6 grid gap-6 lg:grid-cols-2">
            <Panel title={`Input vector (${D_MODEL}-d, post-LN)`}>
              <BarChart values={inputRow} width={300} rowHeight={10} />
            </Panel>
            <Panel title={`Pre-activation (${D_FF}-d, x · W1 + b1)`}>
              <BarChart values={preRow} width={300} rowHeight={10} />
            </Panel>
            <Panel title={`After GELU (${D_FF}-d)`}>
              <BarChart values={actRow} width={300} rowHeight={10} />
            </Panel>
            <Panel
              title={`Output vector (${D_MODEL}-d, act · W2 + b2)`}
              subtitle="Same dimensionality as the input — added back to the residual stream."
            >
              <BarChart values={outRow} width={300} rowHeight={10} />
            </Panel>
          </div>
        </>
      )}

      {error && (
        <p
          role="alert"
          className="mt-3 rounded border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-800 dark:border-red-800 dark:bg-red-950/40 dark:text-red-200"
        >
          {error}
        </p>
      )}
    </div>
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
      <div className="mt-2 overflow-x-auto">{children}</div>
    </div>
  );
}
