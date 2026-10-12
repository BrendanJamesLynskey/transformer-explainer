"use client";

/**
 * Chapter 2: BPE learning its first 24 merges on the corpus, live. Each
 * step shows the five most frequent adjacent pairs (the winner is merged),
 * the new token, the corpus's token count shrinking, and the book's first
 * line re-tokenised with the merges so far. States from `bpeStates`, which
 * runs the tokeniser trainer (bpe.ts) in the browser.
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import {
  PHONE_BELOW,
  useContainerWidth,
} from "@/components/viz/useContainerWidth";
import { useSvgFont } from "@/components/viz/useSvgFont";
import {
  BPE_ANIM_MERGES,
  bpeCaption,
  bpeStates,
  showToken,
} from "@/lib/anim/training/bpe-steps";
import { ALPHABET } from "@/lib/transformer/tokenizer";
import { OKABE_ITO } from "@/lib/viz/palette";

import { useCorpus } from "./useTrainingData";

export default function BpeAnimation({
  children,
  testId = "bpe-animation",
}: {
  children?: ReactNode;
  testId?: string;
}): JSX.Element {
  const corpus = useCorpus();
  const text = corpus && "text" in corpus ? corpus.text : null;
  const states = useMemo(() => (text ? bpeStates(text) : []), [text]);
  const st = useStepper(Math.max(1, states.length), {
    stepMs: 1100,
    resetKey: text ? "ready" : "",
  });
  const [hover, setHover] = useState<string | null>(null);
  const box = useContainerWidth();
  const phone = box.width > 0 && box.width < PHONE_BELOW;
  const W = phone ? 360 : 720;
  const font = useSvgFont(W);
  const fs = font.fs;

  if (!text || states.length === 0) {
    return (
      <p
        data-pending-widget
        className="my-8 min-h-[36rem] text-sm text-neutral-600 dark:text-neutral-400"
      >
        {corpus && "error" in corpus
          ? `Could not load the corpus: ${corpus.error}`
          : "Loading the corpus…"}
      </p>
    );
  }
  const s = states[st.step]!;
  const vocab = (id: number) =>
    id < ALPHABET.length
      ? ALPHABET[id]!
      : states[id - ALPHABET.length + 1]!.token!;
  const hl = s.step < 0 ? "pair" : "merge";
  const on = (k: string) => hl === k || hover === k;

  // ─── layout: the pair counts, then the excerpt as token chips ─────────
  const barTop = fs(12) + 14;
  const rowH = fs(12) + 12;
  const barsH = 5 * rowH;
  const labelW = phone ? 110 : 150;
  const maxCount = Math.max(1, ...(s.top ?? []).map((p) => p.count));
  // token chips, wrapped to the width
  const chipH = fs(13) + 12;
  const charW = fs(13) * 0.62;
  const chips: {
    x: number;
    y: number;
    w: number;
    t: string;
    isNew: boolean;
  }[] = [];
  let x = 0;
  let y = barTop + barsH + fs(12) + 34;
  for (const t of s.excerpt) {
    const w = Math.max(14, showToken(t).length * charW + 8);
    if (x + w > W) {
      x = 0;
      y += chipH + 6;
    }
    chips.push({ x, y, w, t, isNew: s.token !== null && t === s.token });
    x += w + 3;
  }
  const H = y + chipH + 32;

  const visual = (
    <div ref={box.ref} className="min-w-0">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="mx-auto w-full text-neutral-800 dark:text-neutral-200"
        role="img"
        aria-label="The five most frequent pairs of adjacent tokens as bars, the merged pair first; below, the start of the book split into tokens, the newest token highlighted"
      >
        <text x={0} y={barTop - 10} fontSize={fs(12)} fill="currentColor">
          {s.step < 0
            ? "before any merge: one token per character"
            : `merge ${s.step + 1} of ${BPE_ANIM_MERGES}: the most frequent adjacent pairs`}
        </text>
        {(s.top ?? []).map((p, i) => {
          const yy = barTop + i * rowH;
          const win = i === 0;
          return (
            <g key={`${p.pair[0]}-${p.pair[1]}`} data-testid={`pair-${i}`}>
              <text
                x={0}
                y={yy + fs(12)}
                fontSize={fs(12)}
                fill="currentColor"
                fontFamily="monospace"
              >
                {`'${showToken(vocab(p.pair[0]))}' + '${showToken(vocab(p.pair[1]))}'`}
              </text>
              <rect
                x={labelW}
                y={yy + 2}
                width={((W - labelW - 70) * p.count) / maxCount}
                height={rowH - 6}
                fill={win ? OKABE_ITO.blue : OKABE_ITO.sky}
                fillOpacity={win ? (on("merge") ? 0.95 : 0.8) : 0.35}
              />
              <text
                x={W - 4}
                y={yy + fs(12)}
                fontSize={fs(11)}
                textAnchor="end"
                fill="currentColor"
              >
                {p.count.toLocaleString("en-GB")}
              </text>
            </g>
          );
        })}
        <text
          x={0}
          y={barTop + barsH + fs(12) + 10}
          fontSize={fs(12)}
          fill="currentColor"
          data-testid="bpe-count"
        >
          {`the book: ${s.tokensAfter.toLocaleString("en-GB")} tokens (${(states[0]!.tokensBefore / s.tokensAfter).toFixed(2)} characters each); the first line: ${s.excerpt.length}`}
        </text>
        {chips.map((c, i) => (
          <g key={i}>
            <rect
              x={c.x}
              y={c.y}
              width={c.w}
              height={chipH}
              rx={4}
              fill={
                c.isNew
                  ? OKABE_ITO.blue
                  : i % 2
                    ? OKABE_ITO.sky
                    : OKABE_ITO.green
              }
              fillOpacity={c.isNew ? 0.45 : 0.16}
              stroke={c.isNew ? OKABE_ITO.blue : "none"}
              strokeWidth={2}
            />
            <text
              x={c.x + c.w / 2}
              y={c.y + chipH / 2 + fs(13) / 3}
              fontSize={fs(13)}
              textAnchor="middle"
              fill="currentColor"
              fontFamily="monospace"
            >
              {showToken(c.t)}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );

  return (
    <AnimationPanel
      testId={testId}
      title="BPE learns its first merges"
      summary={`The tokeniser trainer runs on the whole book in your browser: count every adjacent pair of tokens inside words, merge the most frequent, repeat. ${BPE_ANIM_MERGES} merges shown.`}
      stepper={st}
      stepLabel="merge"
      countFrom={0}
      caption={bpeCaption(s, vocab)}
      visual={visual}
      equation={children}
      hl={hl}
      onEquationHover={setHover}
    />
  );
}
