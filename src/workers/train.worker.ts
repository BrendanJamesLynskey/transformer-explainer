/*
 * The in-browser trainer (chapter 1's live run), off the main thread.
 *
 * Messages in:  { type: "start", cfg: TrainConfig, text, evalEvery, sampleEvery }
 *               { type: "pause" } | { type: "resume" }
 * Messages out: { type: "progress", logs: StepLog[] }       (about every 50 ms)
 *               { type: "eval", step, val, sample? }       (every evalEvery steps)
 *               { type: "done", step }
 *
 * It runs src/lib/training/trainer.ts unchanged: the same code the unit
 * tests check step for step against the Python reference.
 */
import { encodeChars } from "@/lib/training/data";
import {
  createTrainer,
  evaluate,
  sampleText,
  trainStep,
  type StepLog,
  type TrainConfig,
  type Trainer,
} from "@/lib/training/trainer";

type Start = {
  type: "start";
  cfg: TrainConfig;
  text: string;
  evalEvery: number;
  sampleEvery: number;
  prompt: string;
};
type Msg = Start | { type: "pause" } | { type: "resume" };

let tr: Trainer | null = null;
let running = false;
let opts: Start | null = null;

const post = (m: unknown) => (self as unknown as Worker).postMessage(m);

function evalNow(): void {
  if (!tr || !opts) return;
  const step = tr.t;
  const sample =
    step % opts.sampleEvery === 0 || step === tr.cfg.steps
      ? sampleText(
          tr.params,
          tr.cfg.model,
          encodeChars(opts.prompt),
          80,
          0.8,
          7,
        )
      : undefined;
  post({ type: "eval", step, val: evaluate(tr), sample });
}

function loop(): void {
  if (!running || !tr || !opts) return;
  const t0 = performance.now();
  const logs: StepLog[] = [];
  while (performance.now() - t0 < 50 && tr.t < tr.cfg.steps) {
    if (tr.t % opts.evalEvery === 0) evalNow();
    logs.push(trainStep(tr));
  }
  post({ type: "progress", logs });
  if (tr.t >= tr.cfg.steps) {
    evalNow();
    running = false;
    post({ type: "done", step: tr.t });
    return;
  }
  setTimeout(loop, 0);
}

self.onmessage = (e: MessageEvent<Msg>) => {
  const m = e.data;
  if (m.type === "start") {
    opts = m;
    tr = createTrainer(m.cfg, encodeChars(m.text));
    running = true;
    loop();
  } else if (m.type === "pause") {
    running = false;
  } else if (m.type === "resume" && tr && !running) {
    running = true;
    loop();
  }
};
