"use client";

/**
 * Chapter 1's live run: Part 1's model trained in a Web Worker on the
 * corpus, with its loss curve, its validation loss and a sample every few
 * hundred steps. With the default settings it is the run results.md §4
 * records (the recorded validation losses are drawn as rings, and the
 * browser's run lands on them: the trainer matches the Python reference to
 * about 1e-9 after 2,000 steps).
 */
import { useEffect, useRef, useState } from "react";

import {
  PHONE_BELOW,
  useContainerWidth,
} from "@/components/viz/useContainerWidth";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { sig } from "@/lib/anim/format";
import type { OptimizerKind } from "@/lib/training/optimizers";
import {
  DEFAULT_TRAIN,
  type StepLog,
  type TrainConfig,
} from "@/lib/training/trainer";
import { OKABE_ITO } from "@/lib/viz/palette";

import { loadCorpus } from "./useTrainingData";

export type OptimizerChoice = {
  kind: OptimizerKind;
  lr: number;
  label: string;
};

type Eval = { step: number; val: number; sample?: string };

const BTN =
  "focus-ring inline-flex h-11 min-w-11 items-center justify-center rounded border border-neutral-300 bg-white px-3 text-sm text-neutral-800 hover:bg-neutral-100 disabled:opacity-40 dark:border-neutral-700 dark:bg-neutral-950 dark:text-neutral-200 dark:hover:bg-neutral-800";
const SELECT =
  "focus-ring h-11 rounded border border-neutral-300 bg-white px-2 text-sm dark:border-neutral-700 dark:bg-neutral-950";

export default function LiveTrainer({
  recorded,
  choices,
  prompt,
}: {
  /** the recorded default run's validation losses (results.md §4) */
  recorded: Eval[];
  /** optimiser choices: AdamW (the default run) and each optimiser's best LR (results.md §5) */
  choices: OptimizerChoice[];
  prompt: string;
}): JSX.Element {
  const [choice, setChoice] = useState(0);
  const [steps, setSteps] = useState(DEFAULT_TRAIN.steps);
  const [state, setState] = useState<"idle" | "running" | "paused" | "done">(
    "idle",
  );
  const [logs, setLogs] = useState<StepLog[]>([]);
  const [evals, setEvals] = useState<Eval[]>([]);
  const [rate, setRate] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const worker = useRef<Worker | null>(null);
  const started = useRef(0);
  const box = useContainerWidth();
  const phone = box.width > 0 && box.width < PHONE_BELOW;
  const W = phone ? 360 : 720;
  const font = useSvgFont(W);
  const fs = font.fs;

  useEffect(() => () => worker.current?.terminate(), []);

  const isDefault = choice === 0 && steps === DEFAULT_TRAIN.steps;

  async function start(): Promise<void> {
    worker.current?.terminate();
    setLogs([]);
    setEvals([]);
    setError(null);
    let text: string;
    try {
      text = await loadCorpus();
    } catch (e) {
      setError((e as Error).message);
      return;
    }
    const c = choices[choice]!;
    const cfg: TrainConfig = {
      ...DEFAULT_TRAIN,
      steps,
      optimizer:
        choice === 0
          ? DEFAULT_TRAIN.optimizer
          : {
              kind: c.kind,
              beta2: 0.99,
              weightDecay: c.kind === "adamw" ? 0.01 : 0,
            },
      schedule: { ...DEFAULT_TRAIN.schedule, lr: c.lr },
    };
    const w = new Worker(
      new URL("../../workers/train.worker.ts", import.meta.url),
    );
    worker.current = w;
    started.current = performance.now();
    w.onmessage = (e: MessageEvent) => {
      const m = e.data as
        | { type: "progress"; logs: StepLog[] }
        | ({ type: "eval" } & Eval)
        | { type: "done"; step: number };
      if (m.type === "progress") {
        setLogs((prev) => {
          const next = prev.concat(m.logs);
          const secs = (performance.now() - started.current) / 1000;
          setRate(secs > 0 ? next.length / secs : 0);
          return next;
        });
      } else if (m.type === "eval") {
        setEvals((prev) =>
          prev.concat([{ step: m.step, val: m.val, sample: m.sample }]),
        );
      } else {
        setState("done");
      }
    };
    w.onerror = (e) => setError(e.message || "the training worker failed");
    w.postMessage({
      type: "start",
      cfg,
      text,
      evalEvery: 100,
      sampleEvery: 500,
      prompt,
    });
    setState("running");
  }

  function pause(): void {
    worker.current?.postMessage({ type: "pause" });
    setState("paused");
  }
  function resume(): void {
    worker.current?.postMessage({ type: "resume" });
    setState("running");
  }
  function reset(): void {
    worker.current?.terminate();
    worker.current = null;
    setLogs([]);
    setEvals([]);
    setState("idle");
  }

  // ─── chart ───────────────────────────────────────────────
  const H = phone ? 240 : 280;
  const left = fs(11) * 3;
  const bottom = fs(11) + 18;
  const top = 10;
  const yLo = 1.8;
  const yHi = 4.4;
  const xOf = (t: number) => left + ((W - left - 8) * t) / steps;
  const yOf = (v: number) =>
    top +
    ((H - top - bottom) * (yHi - Math.min(yHi, Math.max(yLo, v)))) /
      (yHi - yLo);
  // mean of every 20 steps' training losses (one batch is noisy)
  const smooth: [number, number][] = [];
  for (let i = 0; i + 20 <= logs.length; i += 20) {
    let s = 0;
    for (let k = i; k < i + 20; k++) s += logs[k]!.loss;
    smooth.push([i + 10, s / 20]);
  }
  const path = smooth
    .map(
      ([t, v], k) =>
        `${k ? "L" : "M"}${xOf(t).toFixed(1)},${yOf(v).toFixed(1)}`,
    )
    .join("");
  const last = logs[logs.length - 1];
  const lastEval = evals[evals.length - 1];
  const samples = evals.filter((e) => e.sample !== undefined);
  const recordedFinal = recorded[recorded.length - 1];

  return (
    <figure
      data-testid="live-trainer"
      data-state={state}
      data-steps-done={logs.length}
      className="my-8 min-w-0 rounded-lg border border-neutral-200 bg-neutral-50 p-4 dark:border-neutral-800 dark:bg-neutral-900"
    >
      <figcaption>
        <p className="font-mono text-[0.65rem] uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
          Live · runs in your browser
        </p>
        <p className="mt-1 font-semibold text-neutral-900 dark:text-neutral-100">
          Train Part 1&rsquo;s model
        </p>
        <p className="mt-1 text-xs text-neutral-600 dark:text-neutral-400">
          {DEFAULT_TRAIN.batch} windows of {DEFAULT_TRAIN.model.seq_len}{" "}
          characters per step, a warmup of {DEFAULT_TRAIN.schedule.warmup} steps
          then a cosine decay, clipped at a gradient norm of{" "}
          {DEFAULT_TRAIN.clip}. It runs in a Web Worker, so the page stays
          responsive; the status line shows how many steps a second your machine
          manages.
        </p>
      </figcaption>

      <div className="mt-3 flex flex-wrap items-end gap-2">
        <label className="flex flex-col gap-1 text-xs text-neutral-700 dark:text-neutral-300">
          Optimiser
          <select
            value={choice}
            onChange={(e) => setChoice(Number(e.target.value))}
            disabled={state === "running"}
            className={SELECT}
            aria-label="Optimiser"
          >
            {choices.map((c, i) => (
              <option key={c.label} value={i}>
                {c.label}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-xs text-neutral-700 dark:text-neutral-300">
          Steps
          <select
            value={steps}
            onChange={(e) => setSteps(Number(e.target.value))}
            disabled={state === "running"}
            className={SELECT}
            aria-label="Steps"
          >
            {[200, 2000, 4000].map((n) => (
              <option key={n} value={n}>
                {n === 200 ? "200 (a quick check)" : n.toLocaleString("en-GB")}
              </option>
            ))}
          </select>
        </label>
        {state === "idle" || state === "done" ? (
          <button
            type="button"
            className={BTN}
            onClick={() => void start()}
            data-testid="train-start"
          >
            ▶ {state === "done" ? "Train again" : "Train"}
          </button>
        ) : state === "running" ? (
          <button
            type="button"
            className={BTN}
            onClick={pause}
            data-testid="train-pause"
          >
            ❚❚ Pause
          </button>
        ) : (
          <button
            type="button"
            className={BTN}
            onClick={resume}
            data-testid="train-resume"
          >
            ▶ Resume
          </button>
        )}
        <button
          type="button"
          className={BTN}
          onClick={reset}
          disabled={state === "idle"}
        >
          Reset
        </button>
      </div>

      <p
        aria-live="polite"
        data-testid="train-status"
        className="mt-3 min-h-10 rounded bg-white px-3 py-2 text-sm text-neutral-800 ring-1 ring-neutral-200 dark:bg-neutral-950 dark:text-neutral-200 dark:ring-neutral-800"
      >
        {error
          ? `The trainer failed: ${error}`
          : state === "idle"
            ? "Press Train. The weights start as Part 1's (seed 42); the loss starts near ln 64 = 4.16. Expect a long plateau before the model learns which characters follow which: a run shorter than about 1,000 steps ends on it."
            : `Step ${logs.length.toLocaleString("en-GB")} of ${steps.toLocaleString("en-GB")}${last ? ` · batch loss ${sig(last.loss, 3)} · lr ${sig(last.lr, 2)}` : ""}${lastEval ? ` · validation ${lastEval.val.toFixed(3)}` : ""}${rate ? ` · ${Math.round(rate)} steps/s` : ""}${state === "done" ? " · done" : state === "paused" ? " · paused" : ""}`}
      </p>

      <div ref={box.ref} className="mt-3 min-w-0">
        <svg
          ref={font.ref}
          viewBox={`0 0 ${W} ${H}`}
          className="w-full text-neutral-800 dark:text-neutral-200"
          role="img"
          aria-label="Loss against training step: the batch loss averaged over 20 steps as a line, the validation loss as dots, and the recorded run's validation losses as rings"
        >
          {[2, 2.5, 3, 3.5, 4].map((v) => (
            <g key={v}>
              <line
                x1={left}
                x2={W - 8}
                y1={yOf(v)}
                y2={yOf(v)}
                stroke="currentColor"
                strokeOpacity={0.12}
              />
              <text
                x={left - 4}
                y={yOf(v) + fs(10) / 3}
                fontSize={fs(10)}
                textAnchor="end"
                fill="currentColor"
              >
                {v.toFixed(1)}
              </text>
            </g>
          ))}
          <line
            x1={left}
            x2={W - 8}
            y1={yOf(Math.log(64))}
            y2={yOf(Math.log(64))}
            stroke={OKABE_ITO.vermillion}
            strokeDasharray="4 3"
            strokeOpacity={0.7}
          />
          <text
            x={W - 10}
            y={yOf(Math.log(64)) - 4}
            fontSize={fs(10)}
            textAnchor="end"
            fill="currentColor"
          >
            ln 64: guessing
          </text>
          <text
            x={(W + left) / 2}
            y={H - 4}
            fontSize={fs(11)}
            textAnchor="middle"
            fill="currentColor"
          >
            training step (0 to {steps.toLocaleString("en-GB")})
          </text>
          {isDefault &&
            recorded.map((e) => (
              <circle
                key={`r${e.step}`}
                cx={xOf(e.step)}
                cy={yOf(e.val)}
                r={5}
                fill="none"
                stroke={OKABE_ITO.blue}
                strokeOpacity={0.7}
              />
            ))}
          <path
            d={path}
            fill="none"
            stroke={OKABE_ITO.orange}
            strokeWidth={2}
          />
          {evals.map((e) => (
            <circle
              key={`v${e.step}`}
              cx={xOf(e.step)}
              cy={yOf(e.val)}
              r={3}
              fill={OKABE_ITO.blue}
              data-testid="val-dot"
            />
          ))}
        </svg>
        <p className="mt-1 text-xs text-neutral-600 dark:text-neutral-400">
          <span style={{ color: OKABE_ITO.orange }}>━</span> batch loss (mean of
          20 steps) · <span style={{ color: OKABE_ITO.blue }}>●</span>{" "}
          validation loss (held-out text) ·{" "}
          {isDefault ? (
            <>
              <span style={{ color: OKABE_ITO.blue }}>○</span> the recorded run
              (results.md §4), ending at {recordedFinal?.val.toFixed(3)}
            </>
          ) : (
            "the recorded run is drawn for the default settings (AdamW, 2,000 steps)"
          )}
        </p>
      </div>

      {samples.length > 0 && (
        <div className="mt-3 space-y-1" data-testid="samples">
          <p className="text-xs font-medium text-neutral-700 dark:text-neutral-300">
            Samples: &ldquo;{prompt}&rdquo; + 80 characters (temperature 0.8,
            seed 7)
          </p>
          {samples.map((e) => (
            <p
              key={e.step}
              className="break-words font-mono text-xs text-neutral-800 dark:text-neutral-200"
            >
              <span className="text-neutral-500 dark:text-neutral-400">
                step {e.step}:{" "}
              </span>
              {prompt}
              {e.sample}
            </p>
          ))}
        </div>
      )}
    </figure>
  );
}
