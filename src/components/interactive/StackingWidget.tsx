"use client";

/**
 * Stacking widget — visualises how the same input changes under
 * successive blocks. Renders one attention pattern grid per block (head 0)
 * so the reader can compare layer-by-layer.
 */
import { useEffect, useState } from "react";

import { AttentionMatrix } from "@/components/viz/AttentionMatrix";
import type { WeightKind } from "@/lib/transformer/trained";

import { usePreset } from "./usePreset";
import { WeightsToggle, useWeightKind } from "./WeightsToggle";

type ForwardResponse = {
  ok: true;
  data: {
    tokens: string[];
    perBlockAttention: number[][][][]; // [blocks][heads][S][S]
  };
};
type ErrorResponse = { ok: false; error: string };

const SEQ_LEN = 8;

async function fetchForward(
  text: string,
  seed: number,
  nBlocks: number,
  weights: WeightKind,
) {
  const res = await fetch("/api/compute/forward", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      text,
      seed,
      weights,
      seqLen: SEQ_LEN,
      dModel: 16,
      dFf: 32,
      nHeads: 2,
      nBlocks,
    }),
  });
  return (await res.json()) as ForwardResponse | ErrorResponse;
}

export function StackingWidget(): JSX.Element {
  const [text, setText] = useState("hello");
  const [seed, setSeed] = useState(42);
  const [weights, setWeights] = useWeightKind();
  const [nBlocks, setNBlocks] = useState(2);
  const [data, setData] = useState<ForwardResponse["data"] | null>(null);
  const [error, setError] = useState<string | null>(null);
  // the trained model has two blocks; other depths run random init
  const depth = weights === "trained" ? 2 : nBlocks;

  usePreset(setText, setSeed);

  useEffect(() => {
    let cancelled = false;
    fetchForward(text, seed, depth, weights)
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
  }, [text, seed, depth, weights]);

  return (
    <div className="my-8 rounded-lg border border-neutral-200 bg-neutral-50 p-4 dark:border-neutral-800 dark:bg-neutral-900">
      <div className="grid gap-3 sm:grid-cols-4">
        <label className="flex flex-col gap-1 text-sm sm:col-span-2">
          <span className="font-medium text-neutral-700 dark:text-neutral-300">
            Input text:
          </span>
          <input
            type="text"
            value={text}
            maxLength={SEQ_LEN * 2}
            onChange={(e) => setText(e.target.value)}
            aria-label="Stacking widget input text"
            className="focus-ring rounded border border-neutral-300 bg-white px-2 py-1 font-mono text-sm dark:border-neutral-700 dark:bg-neutral-950"
          />
        </label>
        <label className="flex flex-col gap-1 text-sm">
          <span className="font-medium text-neutral-700 dark:text-neutral-300">
            {weights === "trained" ? "Seed (random init only):" : "Seed:"}
          </span>
          <input
            type="number"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value || 0))}
            aria-label="Stacking widget seed"
            disabled={weights === "trained"}
            className="focus-ring rounded border border-neutral-300 bg-white px-2 py-1 font-mono text-sm dark:border-neutral-700 dark:bg-neutral-950"
          />
        </label>
        <WeightsToggle
          value={weights}
          onChange={setWeights}
          label="Stacking widget weights"
        />
      </div>

      <div className="mt-4 flex items-center gap-2 text-sm">
        <span className="font-medium text-neutral-700 dark:text-neutral-300">
          Blocks:
        </span>
        <div
          role="group"
          aria-label="Number of blocks"
          className="inline-flex rounded border border-neutral-300 bg-white text-xs dark:border-neutral-700 dark:bg-neutral-950"
        >
          {[1, 2, 3, 4].map((n) => (
            <button
              key={n}
              type="button"
              aria-pressed={depth === n}
              disabled={weights === "trained" && n !== 2}
              onClick={() => setNBlocks(n)}
              className={`focus-ring px-3 py-1 first:rounded-l last:rounded-r disabled:opacity-40 ${
                depth === n
                  ? "bg-accent text-accent-fg"
                  : "text-neutral-600 hover:bg-neutral-100 dark:text-neutral-300 dark:hover:bg-neutral-800"
              }`}
            >
              {n}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <p
          role="alert"
          className="mt-3 rounded border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-800 dark:border-red-800 dark:bg-red-950/40 dark:text-red-200"
        >
          {error}
        </p>
      )}

      {data && (
        <div className="mt-6 grid gap-6 lg:grid-cols-2">
          {data.perBlockAttention.map((heads, blockIdx) => (
            <div key={blockIdx}>
              <h4 className="text-sm font-semibold text-neutral-800 dark:text-neutral-200">
                Block {blockIdx} · head 0 weights
              </h4>
              <p className="text-xs text-neutral-500 dark:text-neutral-400">
                {weights === "trained"
                  ? "Trained: block 0's head 0 puts most of each row on the current letter and the one before it; block 1's head 0 spreads its weight further back."
                  : "Random init: rows spread their weight almost evenly; they still differ from block to block because each layer sees a different residual stream."}
              </p>
              <div className="mt-2 overflow-x-auto">
                <AttentionMatrix
                  matrix={heads[0] ?? []}
                  tokens={data.tokens}
                  fixedRange={{ min: 0, max: 1 }}
                />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
