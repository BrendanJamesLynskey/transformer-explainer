"use client";

/**
 * Embedding widget — the first interactive widget in /learn/02-embeddings.
 *
 * The user types a string. We POST it to `/api/compute/embed`, which runs
 * the same maths the server-side decoder uses, and we render the resulting
 * trace as three heatmaps: token embedding, positional encoding, and
 * their sum (the input to block 0).
 */
import { useEffect, useState } from "react";

import { EmbeddingHeatmap } from "@/components/viz/EmbeddingHeatmap";

type EmbedResponse = {
  ok: true;
  data: {
    alphabet: string;
    tokenIds: number[];
    tokens: string[];
    tokEmb: number[][];
    posEmb: number[][];
    xAfterEmb: number[][];
  };
};

type ErrorResponse = { ok: false; error: string };

const SEED_DEFAULT = 42;
const SEQ_LEN = 8;
const D_MODEL = 16;

async function fetchEmbed(text: string, seed: number) {
  const res = await fetch("/api/compute/embed", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ text, seed, seqLen: SEQ_LEN, dModel: D_MODEL }),
  });
  return (await res.json()) as EmbedResponse | ErrorResponse;
}

export function EmbeddingWidget(): JSX.Element {
  const [text, setText] = useState("hello!");
  const [seed, setSeed] = useState(SEED_DEFAULT);
  const [data, setData] = useState<EmbedResponse["data"] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setPending(true);
    setError(null);
    fetchEmbed(text, seed)
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
            aria-label="Embedding widget input text"
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
            aria-label="Embedding widget seed"
            className="focus-ring rounded border border-neutral-300 bg-white px-2 py-1 font-mono text-sm dark:border-neutral-700 dark:bg-neutral-950"
          />
        </label>
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
        <div className="mt-4 grid gap-6">
          <Section
            title="Token ids"
            subtitle={`${data.tokenIds.length} positions, padded with id 0.`}
          >
            <ol className="flex flex-wrap gap-1.5 font-mono text-xs">
              {data.tokens.map((tok, i) => (
                <li
                  key={i}
                  className="rounded bg-white px-2 py-1 ring-1 ring-neutral-200 dark:bg-neutral-950 dark:ring-neutral-800"
                >
                  <span className="text-neutral-500">{i}</span>
                  <span className="mx-1.5 text-neutral-400">→</span>
                  <span>{data.tokenIds[i]}</span>
                  <span className="mx-1.5 text-neutral-400">·</span>
                  <span>{tok === "\n" ? "↵" : tok === " " ? "␣" : tok}</span>
                </li>
              ))}
            </ol>
          </Section>

          <Section
            title="Token embeddings"
            subtitle={`${data.tokEmb.length} × ${D_MODEL}. Each row is one token's vector.`}
          >
            <EmbeddingHeatmap
              matrix={data.tokEmb}
              rowLabels={data.tokens.map((t) =>
                t === "\n" ? "↵" : t === " " ? "␣" : t,
              )}
            />
          </Section>

          <Section
            title="Positional encoding"
            subtitle={`Sinusoidal pattern. Same for any input — depends only on position.`}
          >
            <EmbeddingHeatmap matrix={data.posEmb} />
          </Section>

          <Section
            title="Sum: input to block 0"
            subtitle="Token + positional. This is what the first attention block sees."
          >
            <EmbeddingHeatmap matrix={data.xAfterEmb} />
          </Section>
        </div>
      )}

      {pending && !data && (
        <p className="mt-4 text-sm text-neutral-500">Computing…</p>
      )}
    </div>
  );
}

function Section({
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
