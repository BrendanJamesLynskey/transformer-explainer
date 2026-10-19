/**
 * The numbers the self-checks (brief 44) quote that are not already in
 * src/data/training/results.json: each one computed here, at build time,
 * by the site's own tested model code (src/lib/transformer, which
 * verify-maths checks against the PyTorch fixtures; src/lib/training and
 * src/lib/anim/training, whose unit tests replay the NumPy reference).
 * Nothing is typed in: change the model and these change with it.
 *
 * MDX reaches them through `<Check v="…">` (src/components/mdx/Check.tsx),
 * as `<R k="…">` reaches results.json; tests/unit/learn-values.test.ts
 * checks each against the fixture or recorded value it can be checked on.
 */
import results from "@/data/training/results.json";
import trainedHeads from "@/data/decoder/trained-heads.json";
import { OPTIM2D, VALLEY } from "@/lib/anim/training/optim2d-steps";
import { MIXED_ADAM, bytesPerParam } from "@/lib/training/memory";
import { causalMask } from "@/lib/transformer/attention";
import { positionalEncoding } from "@/lib/transformer/embeddings";
import { gelu } from "@/lib/transformer/gelu";
import { initModelWeights } from "@/lib/transformer/init";
import { layernorm } from "@/lib/transformer/layernorm";
import { forwardTyped, type ModelConfig } from "@/lib/transformer/model";
import { softmax } from "@/lib/transformer/softmax";
import { applyTemperature, argmax, topPMask } from "@/lib/transformer/sampling";
import { ALPHABET, VOCAB_SIZE, encode } from "@/lib/transformer/tokenizer";
import { TRAINED_CONFIG, trainedWeights } from "@/lib/transformer/trained";

/** How many numbers a nested array of numbers holds. */
export function countNumbers(x: unknown): number {
  if (typeof x === "number") return 1;
  if (Array.isArray(x))
    return x.reduce((n: number, v) => n + countNumbers(v), 0);
  if (x !== null && typeof x === "object")
    return Object.values(x).reduce((n: number, v) => n + countNumbers(v), 0);
  return 0;
}

/** Parameters of a freshly initialised model of this shape. */
export function paramsOf(config: ModelConfig): number {
  return countNumbers(initModelWeights(config));
}

const C = TRAINED_CONFIG;
const dHead = C.d_model / C.n_heads;
/** The sampling chapter's example logits, three tokens. */
export const EXAMPLE_LOGITS = [2, 1, 0];
/** The stacking chapter's prompt (its widget's and animation's default). */
export const STACK_PROMPT = "hello!";

function headStat(block: number, head: number) {
  const h = trainedHeads.trained.find(
    (r) => r.block === block && r.head === head,
  );
  if (!h) throw new Error(`no trained head b${block}h${head}`);
  return h;
}

/** Index of an optimiser in chapter 5's race. */
function race(kind: string): number {
  const i = OPTIM2D.findIndex((o) => o.kind === kind);
  if (i < 0) throw new Error(`no ${kind} in the race`);
  return i;
}

/** Each value, computed on first use. */
const VALUES: Record<string, () => number | string> = {
  // ── Decoder ────────────────────────────────────────────────
  "decoder.vocab": () => VOCAB_SIZE,
  "decoder.seqLen": () => C.seq_len,
  "decoder.dModel": () => C.d_model,
  "decoder.nBlocks": () => C.n_blocks,
  "decoder.params": () => countNumbers(trainedWeights()),
  /** Σ of the positional vector at position 0: sin 0 = 0, cos 0 = 1. */
  "decoder.pe0Sum": () =>
    positionalEncoding(1, C.d_model)[0]!.reduce((a, b) => a + b, 0),
  /** Entries of one head's S × S pattern the causal mask sets to −∞. */
  "decoder.maskedPerHead": () =>
    causalMask(C.seq_len)
      .flat()
      .filter((v) => v === -Infinity).length,
  "decoder.headDim": () => dHead,
  "decoder.scoreScale": () => 1 / Math.sqrt(dHead),
  /** Block 1's second head: the share of attention on the current and the previous character. */
  "decoder.b1h1CurrentPrev": () => {
    const h = headStat(0, 1);
    return h.self + h.prev;
  },
  /** Block 2's second head: share of a space's attention on the window's first two positions. */
  "decoder.b2h1SpaceStart": () => headStat(1, 1).spaceStart,
  /** How many windows of the book those head statistics average over. */
  "decoder.headWindows": () => headStat(0, 1).windows,
  "decoder.ffnWeights": () => 2 * C.d_model * C.d_ff,
  "decoder.attnWeights": () => 4 * C.d_model * C.d_model,
  "decoder.geluMinus1": () => gelu(-1),
  "decoder.ln1234Max": () =>
    Math.max(...layernorm([1, 2, 3, 4], [1, 1, 1, 1], [0, 0, 0, 0])),
  "decoder.lnConstantMaxAbs": () =>
    Math.max(
      ...layernorm([5, 5, 5, 5], [1, 1, 1, 1], [0, 0, 0, 0]).map(Math.abs),
    ),
  /** Parameters with 4 blocks over parameters with 2 (same width). */
  "decoder.params4over2": () =>
    paramsOf({ ...C, n_blocks: 4 }) / paramsOf({ ...C, n_blocks: 2 }),
  /** The trained model's greedy next character after "hello!", shown as text. */
  "decoder.helloNext": () => {
    const ids = encode(STACK_PROMPT, C.seq_len);
    const { logits } = forwardTyped(ids, C, trainedWeights());
    const ch = ALPHABET[argmax(logits[STACK_PROMPT.length - 1]!)]!;
    return ch === " " ? "a space" : `"${ch}"`;
  },
  "decoder.top1AtT1": () => softmax(applyTemperature(EXAMPLE_LOGITS, 1))[0]!,
  "decoder.top1AtT05": () => softmax(applyTemperature(EXAMPLE_LOGITS, 0.5))[0]!,
  "decoder.topP09Kept": () =>
    topPMask(EXAMPLE_LOGITS, 0.9).filter((v) => v !== -Infinity).length,

  // ── Training ───────────────────────────────────────────────
  /** Chapter 5's valley: plain SGD is stable for η < 2 / (steepest curvature). */
  "training.valleySgdLimit": () => 2 / Math.max(VALLEY.a, VALLEY.b),
  "training.raceSgdLr": () => OPTIM2D[race("sgd")]!.lr,
  "training.bytesPerParamMixedAdam": () => {
    const b = bytesPerParam(MIXED_ADAM);
    return b.weights + b.grads + b.optimizer + b.master;
  },
  /** Bytes a worker sends in a ring all-reduce, over the vector's, among 64. */
  "training.ringWireOverPayload64": () =>
    results.systems.collectives.rows[4]!.wire /
    results.systems.collectives.bytes,
  /** Top-k (1 in 16, value + position) payload over the dense gradient's. */
  "training.topkPayloadRatio": () =>
    results.slowLinks.rows[1]!.bytes.payload /
    results.slowLinks.rows[0]!.bytes.payload,
  /** GPipe, 4 devices, 8 micro-batches: idle time as a share of the whole step. */
  "training.gpipeIdleOfStep": () => {
    const b = results.pipeline.rows[0]!.bubble;
    return b / (1 + b);
  },
};

/** The value named `name`; throws at build time if there is none. */
export function learnValue(name: string): number | string {
  const f = VALUES[name];
  if (!f) throw new Error(`no learn value ${name}`);
  return f();
}

/** Every value's name (for the tests). */
export const LEARN_VALUE_NAMES: readonly string[] = Object.keys(VALUES);
