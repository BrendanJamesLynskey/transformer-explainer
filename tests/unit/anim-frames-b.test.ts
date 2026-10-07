/**
 * Frame tests for the brief 27B animations (visual standard §4): every
 * state of the LayerNorm, stacking and generation-loop animations equals
 * the state `scripts/reference.py` builds independently (PyTorch ops in
 * float64 on the site's weights; its own ports of the sampler's masks and
 * the seeded generator) to 1e-12, and the caption built from the
 * TypeScript state equals the caption built from the Python state. Key
 * frames are also pinned literally.
 */
import { describe, expect, it } from "vitest";

import fx from "./fixtures/animations_b.json";
import { mismatch } from "./helpers/close";

import { genCaption, genStates, type GenState } from "@/lib/anim/gen-steps";
import {
  lnCaption,
  lnStates,
  residualTrace,
  type LnState,
} from "@/lib/anim/ln-steps";
import {
  stackCaption,
  stackStates,
  type StackState,
} from "@/lib/anim/stack-steps";
import { DEFAULT_PARAMS, configFor } from "@/lib/compute/traces";
import { initModelWeights } from "@/lib/transformer/init";
import type { GenOptions } from "@/lib/anim/gen-steps";

const TOKENS = ["h", "e", "l", "l", "o", "!", "a", "a"];

describe("the 27B fixture describes the site's model", () => {
  it("uses the widgets' default config", () => {
    expect(configFor(DEFAULT_PARAMS)).toEqual(fx.config);
  });
});

describe("LayerNorm and residuals animation", () => {
  for (const run of fx.layernorm) {
    const want = run.states as unknown as LnState[];
    const tr = residualTrace(
      { ...DEFAULT_PARAMS, text: run.text },
      run.gamma,
      run.beta,
    );
    const got = lnStates(tr, run.pos);
    const name = `pos ${run.pos}, γ ${run.gamma}, β ${run.beta}`;
    it(`${name}: the reference's states, frame by frame`, () => {
      expect(got.length).toBe(16);
      expect(got.length).toBe(want.length);
      got.forEach((s, k) =>
        expect(mismatch(s, want[k]), `frame ${k}`).toBe(null),
      );
    });
    it(`${name}: captions agree`, () => {
      got.forEach((s, k) =>
        expect(lnCaption(s, tr.tokens)).toBe(lnCaption(want[k]!, TOKENS)),
      );
    });
  }
  it("pins key frames", () => {
    const want = fx.layernorm[0]!.states as unknown as LnState[];
    expect(lnCaption(want[0]!, TOKENS)).toBe(
      "Sub-layer 1, position 5 '!': the residual stream x arrives, 16 numbers; LN₁ normalises a copy before attention reads it.",
    );
    expect(lnCaption(want[1]!, TOKENS)).toMatch(
      /^LN₁: the mean of the 16 numbers is μ = −?\d\.\d+(e−\d+)?\.$/,
    );
    expect(lnCaption(want[5]!, TOKENS)).toMatch(
      /^LN₁: γ ⊙ x̂ \+ β with the model's γ = 1, β = 0 changes nothing at initialisation; training learns them\. Mean .+, spread 1\.$/,
    );
    expect(lnCaption(want[15]!, TOKENS)).toMatch(
      /^Residual add: the stream carries on as h \+ Δ \(‖y‖ = 2\.\d+\); the normalised copy is not kept\.$/,
    );
    const g2 = fx.layernorm[1]!.states as unknown as LnState[];
    expect(lnCaption(g2[5]!, TOKENS)).toMatch(
      /^LN₁: γ ⊙ x̂ \+ β with γ = 2, β = 0\.5: the spread becomes 2 and the mean 0\.5\.$/,
    );
  });
});

describe("stacking animation", () => {
  for (const run of fx.stacking) {
    const want = run.states as unknown as StackState[];
    const config = configFor({ ...DEFAULT_PARAMS, nBlocks: run.nBlocks });
    const got = stackStates(
      run.text,
      config,
      initModelWeights(config),
      run.pos,
    );
    const name = `${run.nBlocks} blocks, pos ${run.pos}`;
    it(`${name}: the reference's states, frame by frame`, () => {
      expect(got.length).toBe(2 * run.nBlocks + 2);
      expect(got.length).toBe(want.length);
      got.forEach((s, k) =>
        expect(mismatch(s, want[k]), `frame ${k}`).toBe(null),
      );
    });
    it(`${name}: captions agree`, () => {
      got.forEach((s, k) =>
        expect(stackCaption(s, TOKENS)).toBe(stackCaption(want[k]!, TOKENS)),
      );
    });
  }
  it("pins key frames", () => {
    const want = fx.stacking[0]!.states as unknown as StackState[];
    expect(stackCaption(want[0]!, TOKENS)).toMatch(
      /^Position 5 '!', layer 0: the stream starts as E\[t\] \+ P\[p\] \(‖x‖ = 2\.\d+\)\. Read out now, the logit lens's top guess is '.' at \d+\.\d%\.$/,
    );
    expect(stackCaption(want[1]!, TOKENS)).toMatch(
      /^Block 1 of 4, attention: an update of size 0\.0\d+ is added/,
    );
    expect(stackCaption(want[want.length - 1]!, TOKENS)).toMatch(
      /^After block 4: the final LayerNorm and the output head give the model's logits \(the lens at layer 4, exactly\)\. Top next token: '.' at \d+\.\d%\.$/,
    );
    expect(want[want.length - 1]!.rows).toHaveLength(5);
  });
});

describe("generation loop", () => {
  for (const run of fx.generation) {
    const want = run.states as unknown as GenState[];
    const config = configFor({ ...DEFAULT_PARAMS, seed: run.seed });
    const opts = run.opts as GenOptions;
    const got = genStates(run.text, config, initModelWeights(config), opts);
    const name = `${opts.mode}, τ ${opts.temperature}, seed ${run.seed}`;
    it(`${name}: the reference's states, frame by frame`, () => {
      expect(got.length).toBe(want.length);
      got.forEach((s, k) =>
        expect(mismatch(s, want[k]), `frame ${k}`).toBe(null),
      );
    });
    it(`${name}: captions agree`, () => {
      got.forEach((s, k) => expect(genCaption(s)).toBe(genCaption(want[k]!)));
    });
  }
  it("pins key frames", () => {
    const want = fx.generation[0]!.states as unknown as GenState[];
    expect(want).toHaveLength(21);
    expect(genCaption(want[0]!)).toBe(
      "Round 1 (prefill): the model reads the 5 prompt tokens and writes their keys and values into the KV cache, 5 columns per block.",
    );
    expect(genCaption(want[4]!)).toMatch(
      /^Round 1: top-k \(k = 5\) keeps the 5 most likely, drops the other 59 and renormalises; the top one now has \d+\.\d%\.$/,
    );
    expect(genCaption(want[7]!)).toMatch(
      /^Round 2: the new token '.' at position 5 adds one key and one value column per block \(6 now\); the 5 earlier columns are unchanged, so a server reuses them\.$/,
    );
    expect(genCaption(want[20]!)).toMatch(
      /^Round 3: append '.' at position 7; the context is full \(8 tokens\), so generation stops\.$/,
    );
  });
});
