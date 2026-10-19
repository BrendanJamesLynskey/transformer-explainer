/**
 * The MDX side of the learning layer (brief 44): `<Check>` resolves a
 * self-check's numeric answer and hands it to the shared `SelfCheck`
 * (src/components/learn/, byte-identical on every *-explained site), and
 * `<V>` prints one of the learn values inline, as `<R>` prints a result.
 *
 *   <Check id="t01-q2" n={2} input="number" k="derived.predictionsPerStep"
 *          q="How many next-character predictions does one step score?"
 *          href="#t-training-step" more="The training step">
 *     Why, in a sentence or two (Markdown; `<R>` and `<V>` work here).
 *   </Check>
 *
 * A numeric answer comes from `k` (a path into results.json, as `<R k>`) or
 * `v` (a value computed by the site's model, src/lib/learn/values.ts), and is
 * printed with `<R>`'s own props (`d`, `pct`, `sep`, `div`, `sig`), so the
 * answer reads exactly as the chapter's prose does. Nothing is typed in.
 * Server Components.
 */
import type { ReactNode } from "react";

import { SelfCheck } from "@/components/learn/SelfCheck";
import {
  formatValue,
  lookup,
  scaled,
  type Format,
} from "@/components/training/R";
import { learnValue } from "@/lib/learn/values";
import { getSiteContext } from "@/lib/site/context";
import { siteHref } from "@/lib/site/hosts";

/** The raw value behind `k` (results.json) or `v` (learn values). */
export function resolveAnswer(k?: string, v?: string): unknown {
  if (k !== undefined) return lookup(k);
  if (v !== undefined) return learnValue(v);
  return undefined;
}

export function Check({
  id,
  n,
  q,
  input = "choice",
  choices,
  answer,
  k,
  v,
  unit,
  tol,
  abs,
  min,
  max,
  step,
  href,
  more,
  children,
  ...format
}: {
  id: string;
  n: number;
  /** the question */
  q: string;
  input?: "choice" | "number" | "slider";
  choices?: string[];
  /** choice: index of the right one */
  answer?: number;
  /** number or slider: results.json path, or learn-value name */
  k?: string;
  v?: string;
  unit?: string;
  tol?: number;
  abs?: number;
  min?: number;
  max?: number;
  step?: number;
  /** the section that explains it, and its name */
  href?: string;
  more?: string;
  /** why the answer is what it is */
  children?: ReactNode;
} & Format): JSX.Element {
  let value: number | undefined;
  let display: string | undefined;
  let shownUnit = unit;
  if (input !== "choice") {
    const raw = resolveAnswer(k, v);
    if (typeof raw !== "number")
      throw new Error(`self-check ${id}: ${k ?? v} is not a number`);
    value = scaled(raw, format);
    display = formatValue(raw, format);
    if (format.pct) {
      display = display.replace(/%$/, "");
      shownUnit = unit ?? "%";
    }
  } else if (
    choices === undefined ||
    answer === undefined ||
    answer < 0 ||
    answer >= choices.length
  ) {
    throw new Error(`self-check ${id}: a choice needs choices and an answer`);
  }
  return (
    <SelfCheck
      id={id}
      n={n}
      question={q}
      input={input}
      choices={choices}
      answer={answer}
      value={value}
      display={display}
      unit={shownUnit}
      tol={tol}
      abs={abs}
      min={min}
      max={max}
      step={step}
      why={children}
      href={href ? siteHref(href, getSiteContext()) : undefined}
      hrefLabel={more}
    />
  );
}

/** A learn value printed inline: `<V v="decoder.params" sep />`. */
export function V({ v, ...format }: { v: string } & Format): JSX.Element {
  return <span data-learn-value={v}>{formatValue(learnValue(v), format)}</span>;
}
