"use client";

/**
 * Chapter 07's animation (and the playground's "step through generation"
 * mode): the generation loop. Each round the model runs, the new token's
 * keys and values join the KV cache, the logits become a distribution,
 * temperature reshapes it, top-k or top-p truncates it, a seeded draw picks
 * a token, and the token is appended. Every state comes from `genStates`
 * (the sampler's own functions on a traced forward pass, in the browser).
 *
 * Moving the temperature, k or p re-runs the model at the current step
 * rather than restarting, so the distribution can be watched reshaping as
 * the slider moves; the temperature step itself also animates from τ = 1
 * to the chosen τ.
 */
import { useId, useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import {
  PHONE_BELOW,
  useContainerWidth,
} from "@/components/viz/useContainerWidth";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { maxAbs, pct, showChar, sig } from "@/lib/anim/format";
import {
  GEN_HL,
  displayOrder,
  genCaption,
  genStates,
  temperatureFrame,
  type GenMode,
} from "@/lib/anim/gen-steps";
import { DEFAULT_PARAMS, configFor } from "@/lib/compute/traces";
import { initModelWeights } from "@/lib/transformer/init";
import { ALPHABET } from "@/lib/transformer/tokenizer";
import { LEVEL_COLOUR, MUTED, PART_COLOUR, heat } from "@/lib/viz/palette";

import { usePreset } from "./usePreset";

const KS = [1, 3, 5, 10, 20] as const;
const PS = [0.5, 0.75, 0.9, 0.95] as const;
const MODES: { value: GenMode; label: string }[] = [
  { value: "temperature", label: "temperature only" },
  { value: "top-k", label: "temperature + top-k" },
  { value: "top-p", label: "temperature + top-p" },
];

const chr = (id: number) => showChar(ALPHABET[id] ?? "?");

export default function GenerationAnimation({
  testId = "generation-animation",
  title = "The generation loop, one token at a time",
  children,
}: {
  testId?: string;
  title?: string;
  children?: ReactNode;
}): JSX.Element {
  const [text, setText] = useState("hello");
  const [seed, setSeed] = useState(DEFAULT_PARAMS.seed);
  const [mode, setMode] = useState<GenMode>("top-k");
  const [tau, setTau] = useState(0.1);
  const [k, setK] = useState(5);
  const [p, setP] = useState(0.9);
  const [hover, setHover] = useState<string | null>(null);
  // the playground's presets set the prompt and seed
  usePreset(setText, setSeed);
  const hatch = `gen-hatch-${useId().replace(/:/g, "")}`;
  const config = useMemo(() => configFor({ ...DEFAULT_PARAMS, seed }), [seed]);
  const weights = useMemo(() => initModelWeights(config), [config]);
  const states = useMemo(
    () =>
      genStates(text, config, weights, {
        mode,
        temperature: tau,
        k,
        p,
        rounds: config.seq_len,
      }),
    [text, config, weights, mode, tau, k, p],
  );
  // τ, k and p are left out of the key: moving them keeps the step
  const st = useStepper(states.length, {
    stepMs: 1000,
    smooth: true,
    resetKey: `${text}|${seed}|${mode}`,
  });
  const box = useContainerWidth();
  const phone = box.width > 0 && box.width < PHONE_BELOW;
  const W = phone ? 360 : 720;
  const font = useSvgFont(W);
  const fs = font.fs;
  const s = states[st.step]!;
  const hl = GEN_HL[s.phase];
  const lit = (key: string) => hl === key || hover === key;
  const t = st.step === 0 ? 1 : Math.min(1, st.frac * 2 + (st.playing ? 0 : 1));
  const S = config.seq_len;

  // the vector the bars show now
  const order = s.logits ? displayOrder(s.logits) : [];
  let bars: number[] | null = null;
  let tauNow = 1;
  if (s.phase === "logits") bars = s.logits;
  else if (s.phase === "softmax") bars = s.probs1;
  else if (s.phase === "temperature") {
    const f = temperatureFrame(s, t);
    bars = f.probs;
    tauNow = f.tau;
  } else if (s.final) bars = s.final;
  const isLogits = s.phase === "logits";
  const kept = s.kept ? new Set(s.kept) : null;
  // a fixed scale per round, so temperature and truncation visibly grow bars
  const pScale = Math.max(
    1e-9,
    maxAbs(s.probs1 ?? []),
    maxAbs(s.probsT ?? []),
    maxAbs(s.final ?? []),
  );
  const lScale = Math.max(1e-9, maxAbs(s.logits ?? []));

  // ─── layout ──────────────────────────────────────────────
  const slot = phone ? 40 : 44;
  const ctxX = (W - S * slot) / 2;
  const ctxY = fs(11) + 8;
  const barsY = ctxY + slot + fs(11) * 2 + 18;
  const barsW = phone ? W : 470;
  const barsH = phone ? 110 : 130;
  const bw = barsW / 64;
  const legX = phone ? 0 : barsW + 30;
  const legY = phone ? barsY + barsH + fs(11) + 10 : barsY;
  const legRow = fs(11) + 7;
  const drawY = (phone ? legY + 2 * legRow : barsY + barsH) + fs(11) * 2 + 18;
  const drawH = 22;
  const kvY = drawY + drawH + fs(11) * 2 + 26;
  const kc = phone ? 9 : 12;
  const kr = phone ? 4 : 5;
  const D = config.d_model;
  const gridW = S * kc;
  const gridH = D * kr;
  const nGrid = config.n_blocks * 2;
  const perRow = phone ? 2 : nGrid;
  const gapX = phone ? (W - 2 * gridW) / 3 : (W - nGrid * gridW) / (nGrid + 1);
  const gridXY = (gi: number) => ({
    x: gapX + (gi % perRow) * (gridW + gapX),
    y: kvY + Math.floor(gi / perRow) * (gridH + fs(11) + 16),
  });
  const kvRows = Math.ceil(nGrid / perRow);
  const H = kvY + kvRows * (gridH + fs(11) + 16) + 6;
  const zero = barsY + barsH / 2;
  const base = barsY + barsH;

  // columns written this step appear one by one during the forward step
  const shownCols = (i: number) => {
    if (s.phase !== "forward" || !s.fresh.includes(i)) return true;
    const k0 = s.fresh.indexOf(i);
    return t >= (k0 + 1) / (s.fresh.length + 1);
  };
  const kvMax = Math.max(
    1e-9,
    ...s.cache.flatMap((c) => [...c.K.flat(), ...c.V.flat()].map(Math.abs)),
  );

  // the cumulative strip of the draw (id order, as sampleFromProbs walks it)
  let acc = 0;
  const segs =
    (s.phase === "draw" || s.phase === "append") && s.final
      ? s.final.flatMap((q, id) => {
          const x0 = acc;
          acc += q;
          return q > 0 ? [{ id, x0, q }] : [];
        })
      : [];
  const top5 = bars ? order.slice(0, 5) : [];

  const visual = (
    <div ref={box.ref} className="min-w-0">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="mx-auto w-full text-neutral-800 dark:text-neutral-200"
        role="img"
        aria-label="The context tokens; a bar for each of the 64 characters, most likely first, showing the logits or probabilities; a list of the five most likely; the cumulative probabilities with the random draw marked; and the KV cache as grids of keys and values, one column per cached token"
      >
        <defs>
          <pattern
            id={hatch}
            width={5}
            height={5}
            patternUnits="userSpaceOnUse"
            patternTransform="rotate(45)"
          >
            <rect width={5} height={5} fill={MUTED.light} fillOpacity={0.35} />
            <line
              x1={0}
              y1={0}
              x2={0}
              y2={5}
              stroke={MUTED.dark}
              strokeWidth={1.5}
            />
          </pattern>
        </defs>

        {/* context */}
        <text x={ctxX} y={ctxY - 6} fontSize={fs(11)} fill="currentColor">
          context (round {s.round + 1})
        </text>
        {Array.from({ length: S }, (_, i) => {
          const id = s.context[i];
          const isNew = s.phase === "append" && i === s.context.length - 1;
          const cur = i === s.pos && s.phase !== "append";
          return (
            <g key={i} data-testid={`gctx-${i}`}>
              <rect
                x={ctxX + i * slot + 2}
                y={ctxY}
                width={slot - 4}
                height={slot - 4}
                rx={5}
                fill={isNew ? PART_COLOUR.sample : PART_COLOUR.tok}
                fillOpacity={id === undefined ? 0 : cur || isNew ? 0.25 : 0.06}
                stroke={cur ? PART_COLOUR.tok : "currentColor"}
                strokeOpacity={cur ? 1 : id === undefined ? 0.25 : 0.4}
                strokeWidth={cur ? 2.5 : 1}
                strokeDasharray={id === undefined ? "3 3" : undefined}
              />
              {id !== undefined && (
                <text
                  x={ctxX + i * slot + slot / 2}
                  y={ctxY + slot / 2 + fs(13) / 3}
                  textAnchor="middle"
                  fontSize={fs(13)}
                  fontFamily="monospace"
                  fill="currentColor"
                  opacity={isNew ? Math.max(0.2, t) : 1}
                >
                  {chr(id)}
                </text>
              )}
            </g>
          );
        })}

        {/* the distribution */}
        <text
          x={0}
          y={barsY - fs(11) - 10}
          fontSize={fs(11)}
          fill="currentColor"
        >
          {!bars
            ? "the model runs on the context"
            : isLogits
              ? `logits ℓ at position ${s.pos}, most likely first`
              : s.phase === "softmax"
                ? "p = softmax(ℓ), τ = 1"
                : s.phase === "temperature"
                  ? `p = softmax(ℓ / τ), τ = ${sig(tauNow)}`
                  : s.kept
                    ? `kept ${s.kept.length} of 64, renormalised (hatched: dropped)`
                    : `p = softmax(ℓ / τ), τ = ${sig(s.tau)}`}
        </text>
        <text x={0} y={barsY - 6} fontSize={fs(11)} fill="currentColor">
          {bars
            ? isLogits
              ? `scale ±${sig(lScale)}`
              : `bar height: probability (top = ${pct(pScale)})`
            : ""}
        </text>
        <g data-testid="gen-bars">
          <line
            x1={0}
            x2={barsW}
            y1={isLogits ? zero : base}
            y2={isLogits ? zero : base}
            stroke="currentColor"
            strokeOpacity={0.4}
          />
          {bars &&
            order.map((id, r) => {
              const v = bars![id]!;
              const x = r * bw + 0.5;
              const w = Math.max(0.8, bw - 1);
              const dropped = kept !== null && !kept.has(id);
              const picked =
                (s.phase === "draw" || s.phase === "append") &&
                id === s.sampled;
              if (isLogits) {
                const h = (Math.abs(v) / lScale) * (barsH / 2 - 2);
                return (
                  <rect
                    key={id}
                    x={x}
                    y={v >= 0 ? zero - h : zero}
                    width={w}
                    height={Math.max(0.5, h)}
                    fill={PART_COLOUR.logits}
                    fillOpacity={0.75}
                  />
                );
              }
              // a dropped token keeps its p_τ height, hatched
              const hv = dropped ? (s.probsT?.[id] ?? 0) : v;
              const h = (hv / pScale) * (barsH - 4);
              return (
                <rect
                  key={id}
                  data-testid={`bar-${id}`}
                  data-dropped={dropped || undefined}
                  x={x}
                  y={base - h}
                  width={w}
                  height={Math.max(0.5, h)}
                  fill={dropped ? `url(#${hatch})` : PART_COLOUR.sample}
                  fillOpacity={dropped ? 1 : picked ? 1 : 0.6}
                  stroke={picked ? "currentColor" : "none"}
                  strokeWidth={picked ? 1.5 : 0}
                />
              );
            })}
        </g>

        {/* the five most likely */}
        {bars && (
          <g data-testid="gen-top5">
            {!phone && (
              <text x={legX} y={legY - 6} fontSize={fs(11)} fill="currentColor">
                most likely
              </text>
            )}
            {top5.map((id, r) => (
              <text
                key={id}
                x={legX + (phone ? (r % 3) * 120 : 0)}
                y={legY + (phone ? Math.floor(r / 3) : r) * legRow + fs(11)}
                fontSize={fs(11)}
                fontFamily="monospace"
                fill="currentColor"
                fontWeight={
                  (s.phase === "draw" || s.phase === "append") &&
                  id === s.sampled
                    ? 700
                    : 400
                }
              >
                {`${r + 1}. '${chr(id)}' ${isLogits ? sig(bars![id]!) : pct(bars![id]!)}${kept && !kept.has(id) ? " ✕" : ""}`}
              </text>
            ))}
          </g>
        )}

        {/* the draw */}
        <g data-testid="gen-draw">
          <text x={0} y={drawY - 12} fontSize={fs(11)} fill="currentColor">
            {segs.length > 0
              ? `cumulative probability in id order; u = ${sig(s.u!)} lands on '${chr(s.sampled!)}'`
              : "the draw: u ~ U[0, 1) against the cumulative probabilities"}
          </text>
          <rect
            x={0}
            y={drawY}
            width={W}
            height={drawH}
            fill="none"
            stroke="currentColor"
            strokeOpacity={0.3}
          />
          {segs.map((g) => (
            <rect
              key={g.id}
              x={g.x0 * W}
              y={drawY}
              width={Math.max(0.5, g.q * W)}
              height={drawH}
              fill={PART_COLOUR.sample}
              fillOpacity={g.id === s.sampled ? 0.95 : g.id % 2 ? 0.25 : 0.4}
              stroke={g.id === s.sampled ? "currentColor" : "none"}
              strokeWidth={1.5}
            />
          ))}
          {segs.length > 0 && (
            <path
              data-testid="gen-u"
              d={`M${s.u! * W},${drawY - 2} l-5,-7 h10 z`}
              fill="currentColor"
              opacity={s.phase === "draw" ? Math.max(0.2, t) : 1}
            />
          )}
        </g>

        {/* the KV cache */}
        <text x={0} y={kvY - fs(11) - 12} fontSize={fs(11)} fill="currentColor">
          KV cache: one column per token (HBM on a real accelerator)
        </text>
        {s.cache.flatMap((c, b) =>
          (["K", "V"] as const).map((kind, j) => {
            const gi = b * 2 + j;
            const { x, y } = gridXY(gi);
            const m = c[kind];
            return (
              <g key={gi} data-testid={`kv-${b}-${kind}`}>
                <text x={x} y={y - 6} fontSize={fs(11)} fill="currentColor">
                  block {b + 1} {kind}
                </text>
                <rect
                  x={x - 1}
                  y={y - 1}
                  width={gridW + 2}
                  height={gridH + 2}
                  fill="none"
                  stroke={LEVEL_COLOUR.hbm}
                  strokeWidth={lit("kv") ? 2.5 : 1.2}
                  strokeDasharray={undefined}
                />
                {m.map((row, i) =>
                  shownCols(i) ? (
                    <g
                      key={i}
                      data-testid={`kv-${b}-${kind}-col-${i}`}
                      data-fresh={
                        (s.phase === "forward" && s.fresh.includes(i)) ||
                        undefined
                      }
                    >
                      {row.map((v, d) => {
                        const hcol = heat(v, kvMax);
                        return (
                          <rect
                            key={d}
                            x={x + i * kc}
                            y={y + d * kr}
                            width={kc - 1}
                            height={kr - 0.5}
                            fill={hcol.fill}
                            fillOpacity={hcol.fillOpacity}
                          />
                        );
                      })}
                      {s.phase === "forward" && s.fresh.includes(i) && (
                        <rect
                          x={x + i * kc - 0.5}
                          y={y - 0.5}
                          width={kc}
                          height={gridH + 1}
                          fill="none"
                          stroke="currentColor"
                          strokeWidth={1.5}
                        />
                      )}
                    </g>
                  ) : null,
                )}
              </g>
            );
          }),
        )}
      </svg>
    </div>
  );

  const select =
    "focus-ring h-11 rounded border border-neutral-300 bg-white px-2 font-mono text-sm dark:border-neutral-700 dark:bg-neutral-950";
  const labelText = "font-medium text-neutral-700 dark:text-neutral-300";
  return (
    <AnimationPanel
      testId={testId}
      title={title}
      summary="Logits become a distribution, temperature reshapes it, top-k or top-p trims it, a seeded draw picks a token, and the token is appended while the KV cache grows by a column. Every number is the model's, on your prompt."
      stepper={st}
      stepLabel="step"
      caption={genCaption(s)}
      visual={visual}
      equation={children}
      hl={hl}
      onEquationHover={setHover}
      params={
        <>
          <label className="flex min-w-0 flex-col gap-1 text-sm">
            <span className={labelText}>
              Your prompt (up to {S - 1} characters)
            </span>
            <input
              type="text"
              value={text}
              maxLength={S - 1}
              onChange={(e) => setText(e.target.value.toLowerCase())}
              aria-label="Generation prompt"
              className={select}
            />
          </label>
          <label className="flex min-w-0 flex-col gap-1 text-sm">
            <span className={labelText}>Sampling rule</span>
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value as GenMode)}
              aria-label="Sampling rule"
              className={select}
            >
              {MODES.map((m) => (
                <option key={m.value} value={m.value}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
          <label className="flex min-w-0 flex-col gap-1 text-sm">
            <span className={labelText}>Temperature τ = {sig(tau)}</span>
            <input
              type="range"
              min={0.05}
              max={2}
              step={0.05}
              value={tau}
              onChange={(e) => setTau(Number(e.target.value))}
              aria-label="Temperature"
              className="focus-ring h-11 w-full accent-indigo-600"
            />
          </label>
          {mode === "top-k" && (
            <label className="flex min-w-0 flex-col gap-1 text-sm">
              <span className={labelText}>k (tokens kept)</span>
              <select
                value={k}
                onChange={(e) => setK(Number(e.target.value))}
                aria-label="Top-k"
                className={select}
              >
                {KS.map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            </label>
          )}
          {mode === "top-p" && (
            <label className="flex min-w-0 flex-col gap-1 text-sm">
              <span className={labelText}>p (probability kept)</span>
              <select
                value={p}
                onChange={(e) => setP(Number(e.target.value))}
                aria-label="Top-p"
                className={select}
              >
                {PS.map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            </label>
          )}
          <label className="flex min-w-0 flex-col gap-1 text-sm">
            <span className={labelText}>Seed (weights and the draw)</span>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Math.trunc(Number(e.target.value || 0)))}
              aria-label="Generation seed"
              className={select}
            />
          </label>
        </>
      }
    />
  );
}
