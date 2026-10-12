"use client";

/**
 * The training data the Part 2 widgets share, fetched once per page from
 * /public/training: the corpus (public domain, see CORPUS.md) and the
 * weights the default run ends with (results.md §4; chapter 1's live run
 * reaches the same weights to ~3e-9).
 */
import { useEffect, useState } from "react";

import { encodeChars } from "@/lib/training/data";
import { paramNames, type ParamMap } from "@/lib/training/params";
import { PART1_MODEL } from "@/lib/training/trainer";

let corpusPromise: Promise<string> | null = null;
let weightsPromise: Promise<ParamMap> | null = null;

/** The corpus text (cached per page). */
export function loadCorpus(): Promise<string> {
  corpusPromise ??= fetch("/training/corpus.txt").then((r) => {
    if (!r.ok) throw new Error(`corpus: HTTP ${r.status}`);
    return r.text();
  });
  return corpusPromise;
}

/** The default run's final weights, in the trainer's parameter order. */
export function loadTrainedWeights(): Promise<ParamMap> {
  weightsPromise ??= fetch("/training/default-run-weights.json")
    .then((r) => {
      if (!r.ok) throw new Error(`weights: HTTP ${r.status}`);
      return r.json() as Promise<{ params: ParamMap }>;
    })
    .then((j) =>
      Object.fromEntries(paramNames(PART1_MODEL).map((k) => [k, j.params[k]!])),
    );
  return weightsPromise;
}

/** The corpus and its ids, or null while loading; an error message if it failed. */
export function useCorpus():
  | { text: string; ids: number[] }
  | { error: string }
  | null {
  const [v, setV] = useState<
    { text: string; ids: number[] } | { error: string } | null
  >(null);
  useEffect(() => {
    let live = true;
    loadCorpus()
      .then((text) => live && setV({ text, ids: encodeChars(text) }))
      .catch((e: Error) => live && setV({ error: e.message }));
    return () => {
      live = false;
    };
  }, []);
  return v;
}

/** The trained weights when `want` is true (null while loading). */
export function useTrainedWeights(
  want: boolean,
): ParamMap | { error: string } | null {
  const [v, setV] = useState<ParamMap | { error: string } | null>(null);
  useEffect(() => {
    if (!want) return;
    let live = true;
    loadTrainedWeights()
      .then((p) => live && setV(p))
      .catch((e: Error) => live && setV({ error: e.message }));
    return () => {
      live = false;
    };
  }, [want]);
  return want ? v : null;
}
