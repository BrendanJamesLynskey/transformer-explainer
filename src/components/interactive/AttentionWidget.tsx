"use client";

/**
 * Attention widget — the headline interactive in /learn/03-attention.
 *
 * The user types a string, picks a head, and watches the four panels
 * (raw scores, causal mask overlay, softmax-normalised weights, output
 * heatmap) update from the live trace returned by /api/compute/attention.
 *
 * Hovering a query row in any panel highlights that row across the panels
 * and renders the keys it attends to in a side strip.
 */
import { useEffect, useMemo, useState } from "react";

import { AttentionMatrix } from "@/components/viz/AttentionMatrix";
import { EmbeddingHeatmap } from "@/components/viz/EmbeddingHeatmap";
import { usePreset } from "./usePreset";

type AttnResponse = {
  ok: true;
  data: {
    alphabet: string;
    tokenIds: number[];
    tokens: string[];
    Q: number[][];
    K: number[][];
    V: number[][];
    mask: number[][];
    scores: number[][][]; // [n_heads][S][S]
    weights: number[][][]; // [n_heads][S][S]
    output: number[][];
    nHeads: number;
  };
};

type ErrorResponse = { ok: false; error: string };

const SEED_DEFAULT = 42;
const SEQ_LEN = 8;
const D_MODEL = 16;
const N_HEADS = 2;

async function fetchAttention(text: string, seed: number) {
  const res = await fetch("/api/compute/attention", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      text,
      seed,
      seqLen: SEQ_LEN,
      dModel: D_MODEL,
      nHeads: N_HEADS,
    }),
  });
  return (await res.json()) as AttnResponse | ErrorResponse;
}

function sliceHead(m: number[][], head: number, dHead: number): number[][] {
  return m.map((row) => row.slice(head * dHead, (head + 1) * dHead));
}

function visibleToken(t: string): string {
  return t === "\n" ? "↵" : t === " " ? "␣" : t;
}

export function AttentionWidget(): JSX.Element {
  const [text, setText] = useState("the cat");
  const [seed, setSeed] = useState(SEED_DEFAULT);
  const [head, setHead] = useState(0);
  const [hoverRow, setHoverRow] = useState<number | null>(null);
  const [data, setData] = useState<AttnResponse["data"] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  usePreset(setText, setSeed);

  useEffect(() => {
    let cancelled = false;
    setPending(true);
    setError(null);
    fetchAttention(text, seed)
      .then((r) => {
        if (cancelled) return;
        if (r.ok) setData(r.data);
        else setError(r.error);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : "Failed");
      })
      .finally(() => {
        if (!cancelled) setPending(false);
      });
    return () => {
      cancelled = true;
    };
  }, [text, seed]);

  const dHead = D_MODEL / N_HEADS;
  const headData = useMemo(() => {
    if (!data) return null;
    return {
      Q: sliceHead(data.Q, head, dHead),
      K: sliceHead(data.K, head, dHead),
      V: sliceHead(data.V, head, dHead),
      scores: data.scores[head] ?? [],
      weights: data.weights[head] ?? [],
    };
  }, [data, head, dHead]);

  return (
    <div className="my-8 rounded-lg border border-neutral-200 bg-neutral-50 p-4 dark:border-neutral-800 dark:bg-neutral-900">
      <div className="grid gap-3 sm:grid-cols-3">
        <label className="flex flex-col gap-1 text-sm sm:col-span-2">
          <span className="font-medium text-neutral-700 dark:text-neutral-300">
            Type up to {SEQ_LEN} characters:
          </span>
          <input
            type="text"
            value={text}
            maxLength={SEQ_LEN * 2}
            onChange={(e) => setText(e.target.value)}
            aria-label="Attention widget input text"
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
            aria-label="Attention widget seed"
            className="focus-ring rounded border border-neutral-300 bg-white px-2 py-1 font-mono text-sm dark:border-neutral-700 dark:bg-neutral-950"
          />
        </label>
      </div>

      <div className="mt-4 flex items-center gap-2 text-sm">
        <span className="font-medium text-neutral-700 dark:text-neutral-300">
          Head:
        </span>
        <div
          role="group"
          aria-label="Select an attention head"
          className="inline-flex rounded border border-neutral-300 bg-white text-xs dark:border-neutral-700 dark:bg-neutral-950"
        >
          {Array.from({ length: N_HEADS }, (_, h) => (
            <button
              key={h}
              type="button"
              onClick={() => setHead(h)}
              aria-pressed={head === h}
              className={`focus-ring px-3 py-1 first:rounded-l last:rounded-r ${
                head === h
                  ? "bg-accent text-accent-fg"
                  : "text-neutral-600 hover:bg-neutral-100 dark:text-neutral-300 dark:hover:bg-neutral-800"
              }`}
            >
              {h}
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

      {data && headData && (
        <div className="mt-4 grid gap-6 lg:grid-cols-2">
          <Panel
            title="Raw scores (Q · Kᵀ / √d_head)"
            subtitle="Before mask + softmax. Hover a row to highlight a query."
          >
            <AttentionMatrix
              matrix={headData.scores}
              tokens={data.tokens}
              activeRow={hoverRow}
              onHoverRow={setHoverRow}
            />
          </Panel>

          <Panel
            title="Causal mask"
            subtitle="−∞ above the diagonal — futures stay zero after softmax."
          >
            <AttentionMatrix
              matrix={data.mask}
              tokens={data.tokens}
              activeRow={hoverRow}
              onHoverRow={setHoverRow}
            />
          </Panel>

          <Panel
            title="Softmax weights"
            subtitle="Each row sums to 1 — the query's attention distribution."
          >
            <AttentionMatrix
              matrix={headData.weights}
              tokens={data.tokens}
              activeRow={hoverRow}
              onHoverRow={setHoverRow}
              fixedRange={{ min: 0, max: 1 }}
            />
          </Panel>

          <Panel
            title="Output rows = Σ weights · V"
            subtitle="One [d_head] vector per query position."
          >
            <EmbeddingHeatmap
              matrix={headData.V}
              rowLabels={data.tokens.map(visibleToken)}
            />
          </Panel>

          {hoverRow !== null && headData.weights[hoverRow] && (
            <div className="rounded border border-neutral-200 bg-white p-3 text-sm dark:border-neutral-800 dark:bg-neutral-950 lg:col-span-2">
              <p className="text-xs uppercase tracking-widest text-neutral-500">
                Query row {hoverRow}
                {data.tokens[hoverRow] && (
                  <>
                    {" "}
                    (token{" "}
                    <span className="font-mono">
                      {visibleToken(data.tokens[hoverRow])}
                    </span>
                    )
                  </>
                )}{" "}
                attends to:
              </p>
              <ol className="mt-2 flex flex-wrap gap-2">
                {[...(headData.weights[hoverRow] ?? [])]
                  .map((w, idx) => ({ key: idx, weight: w }))
                  .filter((x) => x.weight > 1e-3)
                  .sort((a, b) => b.weight - a.weight)
                  .slice(0, 6)
                  .map(({ key, weight }) => (
                    <li
                      key={key}
                      className="rounded bg-neutral-100 px-2 py-1 font-mono text-xs dark:bg-neutral-900"
                    >
                      <span className="text-neutral-500">k{key}</span>
                      <span className="mx-1 text-neutral-400">·</span>
                      <span>{visibleToken(data.tokens[key] ?? "?")}</span>
                      <span className="mx-1 text-neutral-400">·</span>
                      <span>{(weight * 100).toFixed(1)}%</span>
                    </li>
                  ))}
              </ol>
            </div>
          )}
        </div>
      )}

      {pending && !data && (
        <p className="mt-4 text-sm text-neutral-500">Computing…</p>
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
  subtitle: string;
  children: React.ReactNode;
}): JSX.Element {
  return (
    <div>
      <h4 className="text-sm font-semibold text-neutral-800 dark:text-neutral-200">
        {title}
      </h4>
      <p className="text-xs text-neutral-500">{subtitle}</p>
      <div className="mt-2 overflow-x-auto">{children}</div>
    </div>
  );
}
