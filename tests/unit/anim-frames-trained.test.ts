/**
 * Frame tests for brief 41: the Decoder's animations on Training's trained
 * weights. Every state each animation builds with `weights: "trained"`
 * equals the state `scripts/reference.py` builds independently (PyTorch
 * ops in float64 on the same weights, loaded from
 * public/training/default-run-weights.json) to 1e-12, and the captions
 * built from both agree. The landing hero's attention weights are checked
 * the same way, and so is the random/trained switch itself.
 */
import { describe, expect, it } from "vitest";

import fx from "./fixtures/animations_trained.json";
import { mismatch } from "./helpers/close";

import { attentionPreview } from "@/lib/attention-preview";
import {
  attentionCaption,
  attentionStates,
  type AttnState,
} from "@/lib/anim/attention-steps";
import {
  embedCaption,
  embedStates,
  type EmbedState,
} from "@/lib/anim/embed-steps";
import { ffnCaption, ffnStates, type FfnState } from "@/lib/anim/ffn-steps";
import {
  genCaption,
  genStates,
  type GenOptions,
  type GenState,
} from "@/lib/anim/gen-steps";
import {
  lnCaption,
  lnStates,
  residualTrace,
  type LnState,
} from "@/lib/anim/ln-steps";
import {
  overviewCaption,
  overviewStates,
  type OverviewState,
} from "@/lib/anim/overview-steps";
import {
  stackCaption,
  stackStates,
  type StackState,
} from "@/lib/anim/stack-steps";
import {
  DEFAULT_PARAMS,
  computeAttention,
  computeBlock,
  computeEmbed,
  computeFfn,
  computeForward,
  configFor,
  trainedFits,
} from "@/lib/compute/traces";
import { initModelWeights } from "@/lib/transformer/init";
import {
  TRAINED_CONFIG,
  TRAINED_STEPS,
  fitsTrained,
  trainedWeights,
  weightsFor,
} from "@/lib/transformer/trained";

const T = { ...DEFAULT_PARAMS, weights: "trained" as const };
const params = (text: string) => ({ ...T, text });
const TOKENS = ["h", "e", "l", "l", "o", "!", "a", "a"];

describe("the trained weights", () => {
  it("are Training's default run, at the widgets' shapes", () => {
    expect(TRAINED_CONFIG).toEqual(fx.config);
    expect(configFor(DEFAULT_PARAMS)).toEqual(fx.config);
    expect(TRAINED_STEPS).toBe(2000);
    const w = trainedWeights();
    expect(w).toBe(trainedWeights()); // unpacked once
    expect(mismatch(w.tok_emb[0], fx.weights_sample.tok_emb_row0, 0)).toBe(
      null,
    );
    expect(
      mismatch(
        w.blocks[1]!.ffn.W2[31],
        fx.weights_sample["block_1.ffn.W2_last_row"],
        0,
      ),
    ).toBe(null);
  });
  it("are used only where the shapes fit; random init otherwise", () => {
    const cfg = configFor(DEFAULT_PARAMS);
    expect(fitsTrained(cfg)).toBe(true);
    expect(fitsTrained({ ...cfg, seq_len: 12, seed: 7 })).toBe(true);
    expect(fitsTrained({ ...cfg, n_blocks: 4 })).toBe(false);
    expect(fitsTrained({ ...cfg, d_model: 32 })).toBe(false);
    expect(fitsTrained({ d_model: 16, n_heads: 2 }, ["n_heads"])).toBe(true);
    expect(weightsFor(cfg, "trained")).toBe(trainedWeights());
    const four = { ...cfg, n_blocks: 4 };
    expect(weightsFor(four, "trained")).toEqual(initModelWeights(four));
    expect(weightsFor(cfg, "random")).toEqual(initModelWeights(cfg));
  });
  it("the traces refuse shapes the trained weights don't have", () => {
    expect(trainedFits(T)).toBe(true);
    expect(trainedFits({ ...T, nBlocks: 3 })).toBe(false);
    expect(trainedFits({ ...T, nBlocks: 3 }, ["dModel", "nHeads"])).toBe(true);
    expect(() => computeForward({ ...T, nBlocks: 3 })).toThrow(/trained/);
    expect(() => computeAttention({ ...T, nHeads: 4 })).toThrow(/trained/);
    // a partial trace only needs the fields it reads
    expect(computeAttention({ ...T, dFf: 64 }).nHeads).toBe(2);
    expect(computeEmbed({ ...T, nHeads: 1 }).tokEmb).toHaveLength(8);
  });
  it("change what every trace computes (and random stays the default)", () => {
    const r = computeBlock({ ...DEFAULT_PARAMS });
    expect(computeBlock({ ...DEFAULT_PARAMS, weights: "random" })).toEqual(r);
    expect(computeBlock(T).output).not.toEqual(r.output);
    expect(computeFfn(T).output).not.toEqual(computeFfn(DEFAULT_PARAMS).output);
  });
});

describe("embeddings animation (trained)", () => {
  const want = fx.embed.states as unknown as EmbedState[];
  const got = embedStates(computeEmbed(params(fx.embed.text)), fx.embed.pos);
  it("has the reference's states, frame by frame", () => {
    expect(got.length).toBe(want.length);
    got.forEach((s, k) =>
      expect(mismatch(s, want[k]), `frame ${k}`).toBe(null),
    );
  });
  it("captions from both states agree", () => {
    got.forEach((s, k) => expect(embedCaption(s)).toBe(embedCaption(want[k]!)));
  });
});

describe("attention animation (trained)", () => {
  for (const run of fx.attention) {
    const want = run.states as unknown as AttnState[];
    const tr = computeAttention(params(run.text));
    const head = run.head === "all" ? ("all" as const) : Number(run.head);
    const got = attentionStates(tr, { head, row: run.row });
    it(`head ${run.head} from row ${run.row}: the reference's states`, () => {
      expect(got.length).toBe(want.length);
      got.forEach((s, k) =>
        expect(mismatch(s, want[k]), `frame ${k}`).toBe(null),
      );
    });
    it(`head ${run.head} from row ${run.row}: captions agree`, () => {
      got.forEach((s, k) =>
        expect(attentionCaption(s, tr.tokens)).toBe(
          attentionCaption(want[k]!, tr.tokens),
        ),
      );
    });
  }
  it("the trained head is far from uniform where random init is not", () => {
    const trained = computeAttention(params("hello!")).weights[0]![3]!;
    const random = computeAttention({ ...DEFAULT_PARAMS, text: "hello!" })
      .weights[0]![3]!;
    const spread = (r: number[]) => Math.max(...r.slice(0, 4)) - 0.25;
    expect(spread(random)).toBeLessThan(0.01);
    expect(spread(trained)).toBeGreaterThan(0.2);
  });
});

describe("feed-forward animation (trained)", () => {
  const want = fx.ffn.states as unknown as FfnState[];
  const got = ffnStates(computeFfn(params(fx.ffn.text)), fx.ffn.pos);
  it("has the reference's states, frame by frame", () => {
    expect(got.length).toBe(want.length);
    got.forEach((s, k) =>
      expect(mismatch(s, want[k]), `frame ${k}`).toBe(null),
    );
  });
  it("captions from both states agree", () => {
    got.forEach((s, k) => expect(ffnCaption(s)).toBe(ffnCaption(want[k]!)));
  });
});

describe("overview hero (trained)", () => {
  const want = fx.overview.states as unknown as OverviewState[];
  const config = configFor(DEFAULT_PARAMS);
  const got = overviewStates(
    fx.overview.text,
    config,
    weightsFor(config, "trained"),
    { rounds: fx.overview.rounds },
  );
  it("has the reference's states, frame by frame", () => {
    expect(got.length).toBe(want.length);
    got.forEach((s, k) =>
      expect(mismatch(s, want[k]), `frame ${k}`).toBe(null),
    );
  });
  it("captions from both states agree", () => {
    got.forEach((s, k) =>
      expect(overviewCaption(s)).toBe(overviewCaption(want[k]!)),
    );
  });
});

describe("LayerNorm animation (trained stream, γ and β from the sliders)", () => {
  for (const run of fx.layernorm) {
    const want = run.states as unknown as LnState[];
    const tr = residualTrace(params(run.text), run.gamma, run.beta);
    const got = lnStates(tr, run.pos);
    it("has the reference's states, frame by frame", () => {
      expect(got.length).toBe(want.length);
      got.forEach((s, k) =>
        expect(mismatch(s, want[k]), `frame ${k}`).toBe(null),
      );
    });
    it("captions agree", () => {
      got.forEach((s, k) =>
        expect(lnCaption(s, tr.tokens)).toBe(lnCaption(want[k]!, TOKENS)),
      );
    });
  }
});

describe("stacking animation and logit lens (trained)", () => {
  const run = fx.stacking[0]!;
  const want = run.states as unknown as StackState[];
  const config = configFor({ ...DEFAULT_PARAMS, nBlocks: run.nBlocks });
  const got = stackStates(
    run.text,
    config,
    weightsFor(config, "trained"),
    run.pos,
  );
  it("has the reference's states, frame by frame", () => {
    expect(got.length).toBe(want.length);
    got.forEach((s, k) =>
      expect(mismatch(s, want[k]), `frame ${k}`).toBe(null),
    );
  });
  it("captions agree", () => {
    got.forEach((s, k) =>
      expect(stackCaption(s, TOKENS)).toBe(stackCaption(want[k]!, TOKENS)),
    );
  });
  it("the lens moves: layer 0 reads back the input character, later layers a space", () => {
    const rows = want[want.length - 1]!.rows;
    expect(rows[0]!.lens[0]!.id).toBe(7); // 'h': the tied head reads E back
    expect(rows[2]!.lens[0]!.id).toBe(36); // ' ' after "hello!"
    expect(rows[2]!.lens[0]!.prob).toBeGreaterThan(rows[1]!.lens[0]!.prob);
  });
});

describe("generation loop (trained)", () => {
  for (const run of fx.generation) {
    const want = run.states as unknown as GenState[];
    const config = configFor({ ...DEFAULT_PARAMS, seed: run.seed });
    const opts = run.opts as GenOptions;
    const got = genStates(
      run.text,
      config,
      weightsFor(config, "trained"),
      opts,
    );
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
});

describe("the landing hero's attention", () => {
  const p = attentionPreview({ text: fx.preview.text });
  it("is the reference's, head by head", () => {
    expect(p.tokens.join("")).toBe(fx.preview.text);
    expect(p.heads).toHaveLength(4);
    p.heads.forEach((h, k) =>
      expect(mismatch(h, fx.preview.heads[k]), `head ${k}`).toBe(null),
    );
    expect(p.weights).toEqual(p.heads[0]!.weights);
  });
  it("random init is the seeded initialiser and changes with the seed", () => {
    const r = attentionPreview({ text: fx.preview.text, kind: "random" });
    expect(r.heads[0]!.weights).not.toEqual(p.heads[0]!.weights);
    expect(
      attentionPreview({ text: fx.preview.text, kind: "random", seed: 7 })
        .weights,
    ).not.toEqual(r.weights);
    expect(attentionPreview({ text: "ab", head: 5 }).weights).toEqual([]);
  });
});
