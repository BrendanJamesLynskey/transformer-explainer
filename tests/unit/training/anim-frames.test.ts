/**
 * Frame tests for Part 2's animations (visual standard §4): every state the
 * site builds equals the state scripts/training/anim.py builds from the
 * NumPy reference, to 1e-11 (1e-12 where no training step is involved), and
 * caption(TS state) = caption(Python state) for every state. Some captions
 * are pinned literally.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import bpx from "../fixtures/training/anim_backprop.json";
import bx from "../fixtures/training/anim_bpe.json";
import hx from "../fixtures/training/anim_hero.json";
import ox from "../fixtures/training/anim_objective.json";
import qx from "../fixtures/training/anim_optim2d.json";
import { mismatch } from "../helpers/close";

import { CORPUS_IDS, CORPUS_TEXT } from "./corpus";

import { backpropCaption, backpropStates } from "@/lib/anim/training/backprop-steps";
import { bpeCaption, bpeStates, type BpeState } from "@/lib/anim/training/bpe-steps";
import { heroCaption, heroStates, type HeroState } from "@/lib/anim/training/hero-steps";
import {
  objectiveCaption,
  objectiveStates,
  type ObjectiveState,
} from "@/lib/anim/training/objective-steps";
import {
  OPTIM2D,
  VALLEY,
  optim2dCaption,
  optim2dStates,
  type Optim2dState,
} from "@/lib/anim/training/optim2d-steps";
import { fromWeights, paramNames, type ParamMap } from "@/lib/training/params";
import { PART1_MODEL } from "@/lib/training/trainer";
import { initModelWeights } from "@/lib/transformer/init";
import { ALPHABET } from "@/lib/transformer/tokenizer";

const checkpoint = JSON.parse(
  readFileSync(join(__dirname, "../../../public/training/default-run-weights.json"), "utf8"),
) as { params: ParamMap; steps: number };
const trained = (): ParamMap =>
  Object.fromEntries(paramNames(PART1_MODEL).map((k) => [k, checkpoint.params[k]!]));
const init = () => fromWeights(initModelWeights(PART1_MODEL));

describe("chapter 1 hero: three real training steps", () => {
  const states = heroStates(CORPUS_IDS);
  it("every state = Python's", () => {
    expect(states).toHaveLength(hx.states.length);
    expect(mismatch(states, hx.states, 1e-11)).toBeNull();
  });
  it("every caption = the caption of Python's state", () => {
    states.forEach((s, i) =>
      expect(heroCaption(s)).toBe(heroCaption(hx.states[i] as HeroState)),
    );
  });
  it("each step lowers the batch's loss; the first starts near ln 64", () => {
    for (const s of states.filter((x) => x.phase === "update"))
      expect(s.lossAfter!).toBeLessThan(s.loss!);
    expect(Math.abs(states[2]!.loss! - Math.log(64))).toBeLessThan(0.05);
    expect(heroCaption(states[0]!)).toBe(
      "Step 1: draw 4 windows of 8 characters; each position's target is the character after it (32 predictions).",
    );
  });
});

describe("chapter 2: BPE merges", () => {
  const states = bpeStates(CORPUS_TEXT);
  const vocab = (id: number) =>
    id < 64 ? ALPHABET[id]! : (states[id - 64 + 1]!.token as string);
  it("every state = Python's", () => {
    expect(mismatch(states, bx.states)).toBeNull();
  });
  it("every caption = the caption of Python's state; the first merge is pinned", () => {
    states.forEach((s, i) =>
      expect(bpeCaption(s, vocab)).toBe(bpeCaption(bx.states[i] as BpeState, vocab)),
    );
    expect(bpeCaption(states[1]!, vocab)).toMatch(/^Merge 1: /);
    expect(states.at(-1)!.excerpt.length).toBeLessThan(states[0]!.excerpt.length);
  });
});

describe("chapter 3: cross-entropy position by position", () => {
  for (const [name, params] of [
    ["init", init],
    ["trained", trained],
  ] as const) {
    it(`${name}: every state and caption = Python's`, () => {
      const states = objectiveStates(params());
      const want = ox[name] as ObjectiveState[];
      expect(mismatch(states, want, 1e-12)).toBeNull();
      states.forEach((s, i) => expect(objectiveCaption(s)).toBe(objectiveCaption(want[i]!)));
    });
  }
  it("the gradient over the vocabulary sums to 0 at each position", () => {
    for (const s of objectiveStates(trained()).filter((x) => x.phase === "grad")) {
      expect(Math.abs(s.grad!.reduce((a, b) => a + b, 0))).toBeLessThan(1e-15);
    }
  });
  it("the trained model is better on the window than the initial one", () => {
    const m = (p: ParamMap) => objectiveStates(p).at(-1)!.mean!;
    expect(m(trained())).toBeLessThan(m(init()));
  });
});

describe("chapter 4: the gradient on its way back", () => {
  for (const [name, params] of [
    ["init", init],
    ["trained", trained],
  ] as const) {
    it(`${name}: every recorded gradient and every weight's ‖g‖ = Python's`, () => {
      const states = backpropStates(params());
      const want = bpx[name];
      expect(states.slice(0, -1).map((s) => s.name)).toEqual(want.record.map((r) => r.name));
      states.slice(0, -1).forEach((s, i) => {
        expect(mismatch(s.value, want.record[i]!.value, 1e-12)).toBeNull();
      });
      for (const s of states) {
        for (const p of s.params) {
          expect(
            mismatch(p.norm, (want.gradNorms as Record<string, number>)[p.name], 1e-12),
          ).toBeNull();
        }
        expect(mismatch(s.loss, want.loss, 1e-12)).toBeNull();
      }
      // every parameter's gradient is finished at exactly one stop
      const done = states.flatMap((s) => s.params.map((p) => p.name)).sort();
      expect(done).toEqual(paramNames(PART1_MODEL).slice().sort());
      states.forEach((s, i) => expect(backpropCaption(s, i, states.length)).toMatch(/^Stop /));
    });
  }
});

describe("chapter 5: four optimisers on a 2-D valley", () => {
  const states = optim2dStates();
  it("every trajectory and Adam's moments = Python's", () => {
    expect(qx.valley.steps).toBe(VALLEY.steps);
    expect(qx.optimisers.map((o) => o.kind)).toEqual(OPTIM2D.map((o) => o.kind));
    expect(mismatch(states, qx.states, 1e-12)).toBeNull();
    states.forEach((s, i) =>
      expect(optim2dCaption(s)).toBe(optim2dCaption(qx.states[i] as Optim2dState)),
    );
  });
  it("momentum beats SGD in 60 steps; AdamW's decay pulls it towards the origin", () => {
    const last = states.at(-1)!;
    expect(last.losses[1]!).toBeLessThan(last.losses[0]!);
    const r = (p: [number, number]) => Math.hypot(p[0], p[1]);
    expect(r(last.points[3]!)).toBeLessThan(r(last.points[2]!));
  });
});
