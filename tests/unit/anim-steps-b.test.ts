/**
 * The brief 27B animations show the model's own numbers: the LayerNorm
 * animation's normalised copy and stream sums equal block 0's trace
 * exactly, the stacking animation's last logit lens is the model's logits
 * exactly, every token the generation loop samples is the one the site's
 * sampler (`sample`) draws with the same seeded generator, and a cached
 * key or value never changes from one round to the next.
 */
import { describe, expect, it } from "vitest";

import {
  GEN_HL,
  displayOrder,
  genCaption,
  genStates,
  sampleMode,
  temperatureFrame,
  type GenOptions,
} from "@/lib/anim/gen-steps";
import {
  LN_HL,
  LN_PHASES,
  lnCaption,
  lnHl,
  lnStates,
  meanStd,
  residualTrace,
} from "@/lib/anim/ln-steps";
import {
  STACK_HL,
  logitLens,
  stackCaption,
  stackStates,
} from "@/lib/anim/stack-steps";
import {
  DEFAULT_PARAMS,
  computeBlock,
  configFor,
  tracedForward,
} from "@/lib/compute/traces";
import { initModelWeights } from "@/lib/transformer/init";
import { mulberry32 } from "@/lib/transformer/random";
import { sample } from "@/lib/transformer/sampling";
import { softmax } from "@/lib/transformer/softmax";
import { encode } from "@/lib/transformer/tokenizer";

const P = { ...DEFAULT_PARAMS, text: "hello!" };

describe("LayerNorm and residuals", () => {
  const tr = residualTrace(P);
  it("with the model's γ = 1, β = 0 it is block 0 exactly (computeBlock)", () => {
    const b = computeBlock(P);
    expect(tr.x).toEqual(b.input);
    expect(tr.ln1).toEqual(b.ln1);
    expect(tr.h).toEqual(b.attnOut);
    expect(tr.ln2).toEqual(b.ln2);
    expect(tr.y).toEqual(b.output);
  });
  it("adds exactly the sub-layer outputs the block added", () => {
    tr.x.forEach((row, p) => {
      expect(row.map((v, k) => v + tr.attn[p]![k]!)).toEqual(tr.h[p]);
      expect(tr.h[p]!.map((v, k) => v + tr.ffn[p]![k]!)).toEqual(tr.y[p]);
    });
  });
  for (const [g, b] of [
    [1, 0],
    [2, 0.5],
    [0.5, -0.5],
  ] as const) {
    it(`γ = ${g}, β = ${b}: the affine step is the block's LN output exactly`, () => {
      const t = residualTrace(P, g, b);
      for (const pos of [0, 3, 7]) {
        const st = lnStates(t, pos);
        expect(st).toHaveLength(16);
        expect(st.map((s) => s.phase)).toEqual([...LN_PHASES, ...LN_PHASES]);
        expect(st[5]!.ln).toEqual(t.ln1[pos]);
        expect(st[13]!.ln).toEqual(t.ln2[pos]);
        expect(st[7]!.out).toEqual(t.h[pos]);
        expect(st[15]!.out).toEqual(t.y[pos]);
        // the centred and scaled vectors have mean 0 and spread 1
        const c = meanStd(st[3]!.values);
        const z = meanStd(st[4]!.values);
        expect(Math.abs(c.mean)).toBeLessThan(1e-12);
        expect(Math.abs(z.mean)).toBeLessThan(1e-12);
        expect(z.std).toBeCloseTo(1, 4);
        const a = meanStd(st[5]!.values);
        expect(a.std).toBeCloseTo(g, 4);
        expect(a.mean).toBeCloseTo(b, 10);
      }
    });
  }
  it("clamps the position and captions every phase", () => {
    const st = lnStates(tr, 99);
    expect(st[0]!.pos).toBe(7);
    expect(lnStates(tr, -3)[0]!.pos).toBe(0);
    const caps = lnStates(tr, 5).map((s) => lnCaption(s, tr.tokens));
    expect(new Set(caps).size).toBe(16);
    expect(caps[8]).toMatch(
      /^Sub-layer 2, position 5 '!': the residual stream h arrives/,
    );
    expect(caps[14]).toMatch(
      /^The FFN reads LN₂\(h\) and returns its update Δ/,
    );
    expect(lnCaption(lnStates(tr, 5)[0]!, [])).toContain("'?'");
  });
  it("highlights the right term for each sub-layer", () => {
    const st = lnStates(tr, 5);
    expect(st.map(lnHl)).toEqual([
      "x",
      "mu",
      "sig",
      "hat",
      "hat",
      "gb",
      "sub",
      "res",
      "x2",
      "mu",
      "sig",
      "hat",
      "hat",
      "gb",
      "sub2",
      "res2",
    ]);
    expect(Object.keys(LN_HL)).toEqual([...LN_PHASES]);
  });
});

describe("stacking", () => {
  for (const nBlocks of [1, 2, 3, 4]) {
    it(`${nBlocks} block(s): the last lens is the model's logits exactly`, () => {
      const config = configFor({ ...P, nBlocks });
      const w = initModelWeights(config);
      const st = stackStates(P.text, config, w, 5);
      expect(st).toHaveLength(2 * nBlocks + 2);
      const tr = tracedForward(encode(P.text, 8), config, w);
      const last = tr.blocks[nBlocks - 1]!.ffnOut[5]!;
      expect(logitLens(last, w)).toEqual(tr.logits[5]);
      const fin = st[st.length - 1]!;
      expect(fin.kind).toBe("final");
      expect(fin.rows).toHaveLength(nBlocks + 1);
      expect(fin.top).toEqual(fin.rows[nBlocks]!.lens);
      expect(fin.vec).toEqual(tr.xFinal[5]);
      // every block's FFN step ends on that block's output
      st.filter((s) => s.kind === "ffn").forEach((s) =>
        expect(s.vec).toEqual(tr.blocks[s.block]!.ffnOut[5]),
      );
      expect(st.map((s) => STACK_HL[s.kind])[0]).toBe("x0");
    });
  }
  it("lens probabilities are a softmax and captions cover every kind", () => {
    const config = configFor(P);
    const w = initModelWeights(config);
    const st = stackStates(P.text, config, w, 42);
    expect(st[0]!.pos).toBe(7);
    const lens = st[0]!.rows[0]!.lens;
    const probs = softmax(logitLens(st[0]!.vec, w));
    expect(lens[0]!.prob).toBe(Math.max(...probs));
    const tokens = ["h", "e", "l", "l", "o", "!", "a", "a"];
    const caps = st.map((s) => stackCaption(s, tokens));
    expect(caps[0]).toMatch(/^Position 7 'a', layer 0/);
    expect(caps[1]).toMatch(/^Block 1 of 2, attention/);
    expect(caps[2]).toMatch(/^Block 1 of 2, FFN/);
    expect(caps[5]).toMatch(/^After block 2/);
    expect(stackCaption(st[0]!, [])).toContain("'?'");
  });
});

describe("generation loop", () => {
  const config = configFor(P);
  const w = initModelWeights(config);
  const base: GenOptions = {
    mode: "top-k",
    temperature: 0.1,
    k: 5,
    p: 0.9,
    rounds: 8,
  };
  const variants: GenOptions[] = [
    base,
    { ...base, mode: "top-p", temperature: 0.5 },
    { ...base, mode: "top-p", temperature: 0.05, p: 0.5 },
    { ...base, mode: "temperature", temperature: 1 },
    { ...base, mode: "temperature", temperature: 2 },
    { ...base, mode: "top-k", k: 1 },
  ];
  for (const o of variants) {
    it(`${o.mode} τ ${o.temperature}: every draw is the site sampler's`, () => {
      for (const text of ["hello", "the", "", "abcdefgh"]) {
        const st = genStates(text, config, w, o);
        const rng = mulberry32(config.seed);
        const draws = st.filter((s) => s.phase === "draw");
        for (const s of draws)
          expect(s.sampled).toBe(sample(s.logits!, sampleMode(o), rng));
        // the loop stops when the context is full
        const last = st[st.length - 1]!;
        expect(last.phase).toBe("append");
        expect(last.context).toHaveLength(8);
        expect(genCaption(last)).toMatch(/so generation stops\.$/);
      }
    });
  }
  it("a cached key or value never changes from round to round", () => {
    const st = genStates("hello", config, w, base);
    const fwd = st.filter((s) => s.phase === "forward");
    expect(fwd.map((s) => s.cache[0]!.K.length)).toEqual([5, 6, 7]);
    expect(fwd.map((s) => s.fresh)).toEqual([[0, 1, 2, 3, 4], [5], [6]]);
    for (let r = 1; r < fwd.length; r++)
      fwd[r - 1]!.cache.forEach((c, b) => {
        expect(fwd[r]!.cache[b]!.K.slice(0, c.K.length)).toEqual(c.K);
        expect(fwd[r]!.cache[b]!.V.slice(0, c.V.length)).toEqual(c.V);
      });
    // and the cache is the traced forward pass's K and V
    const ids = encode("hello", 5).concat([0, 0, 0]);
    const tr = tracedForward(ids, config, w);
    expect(fwd[0]!.cache[1]!.K).toEqual(tr.blocks[1]!.attn.K.slice(0, 5));
  });
  it("truncation keeps what the masks keep and renormalises", () => {
    const st = genStates("hello", config, w, base);
    const tr = st.find((s) => s.phase === "truncate")!;
    expect(tr.kept).toHaveLength(5);
    expect(tr.kept).toEqual(
      displayOrder(tr.logits!)
        .slice(0, 5)
        .sort((a, b) => a - b),
    );
    const sum = tr.final!.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 12);
    const tp = genStates("hello", config, w, { ...base, mode: "top-p" });
    const t2 = tp.find((s) => s.phase === "truncate")!;
    const mass = t2.kept!.reduce((a, i) => a + t2.probsT![i]!, 0);
    expect(mass).toBeGreaterThanOrEqual(0.9);
    // temperature-only mode has no truncation step
    const tm = genStates("hello", config, w, { ...base, mode: "temperature" });
    expect(tm.some((s) => s.phase === "truncate")).toBe(false);
    expect(tm.length).toBe(18);
  });
  it("the temperature frame runs from τ = 1 to τ", () => {
    const st = genStates("hello", config, w, base);
    const s = st.find((x) => x.phase === "temperature")!;
    expect(temperatureFrame(s, 0)).toEqual({ tau: 1, probs: s.probs1 });
    expect(temperatureFrame(s, 1)).toEqual({ tau: 0.1, probs: s.probsT });
    const mid = temperatureFrame(s, 0.5);
    expect(mid.tau).toBeCloseTo(0.55, 12);
    expect(Math.max(...mid.probs)).toBeGreaterThan(Math.max(...s.probs1!));
    expect(Math.max(...mid.probs)).toBeLessThan(Math.max(...s.probsT!));
    const fwd = st[0]!;
    expect(temperatureFrame(fwd, 0.5)).toEqual({ tau: 0.1, probs: [] });
  });
  it("captions and highlights cover every phase; rounds can be capped", () => {
    const st = genStates("hello", config, w, { ...base, rounds: 1 });
    expect(st.map((s) => s.phase)).toEqual([
      "forward",
      "logits",
      "softmax",
      "temperature",
      "truncate",
      "draw",
      "append",
    ]);
    expect(genCaption(st[6]!)).toMatch(/the loop runs again\.$/);
    expect(st.map((s) => GEN_HL[s.phase])).toEqual([
      "kv",
      "logits",
      "tau",
      "tau",
      "trunc",
      "draw",
      "sample",
    ]);
    const one = genStates("hello", config, w, {
      ...base,
      mode: "temperature",
      temperature: 1,
      rounds: 1,
    });
    expect(genCaption(one[3]!)).toBe(
      "Round 1: temperature τ = 1 leaves the distribution as it is.",
    );
    const tp = genStates("hello", config, w, { ...base, mode: "top-p" });
    expect(genCaption(tp[4]!)).toMatch(
      /^Round 1: top-p \(p = 0\.9\) keeps the \d+ most likely, the fewest holding at least 90\.0%/,
    );
    expect(sampleMode({ ...base, mode: "temperature" })).toEqual({
      kind: "temperature",
      temperature: 0.1,
    });
  });
  it("display order is most likely first, ties by id", () => {
    expect(displayOrder([0.1, 0.3, 0.3, -1])).toEqual([1, 2, 0, 3]);
  });
});
