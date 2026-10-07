/**
 * Frame tests (visual standard §4): every animation state the site builds
 * equals the state `scripts/reference.py` builds independently (PyTorch
 * ops in float64 on the site's weights) to 1e-12, and the caption built
 * from the TypeScript state equals the caption built from the Python
 * state. A few captions are also pinned literally.
 */
import { describe, expect, it } from "vitest";

import fx from "./fixtures/animations.json";
import { mismatch } from "./helpers/close";

import {
  DEFAULT_PARAMS,
  computeAttention,
  computeEmbed,
  computeFfn,
  configFor,
} from "@/lib/compute/traces";
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
  overviewCaption,
  overviewStates,
  type OverviewState,
} from "@/lib/anim/overview-steps";
import { initModelWeights } from "@/lib/transformer/init";

const params = (text: string) => ({ ...DEFAULT_PARAMS, text });

describe("the fixture describes the site's model", () => {
  it("uses the widgets' default config", () => {
    expect(fx.config).toEqual({
      seq_len: 8,
      d_model: 16,
      n_heads: 2,
      d_ff: 32,
      n_blocks: 2,
      vocab_size: 64,
      seed: 42,
    });
    expect(configFor(DEFAULT_PARAMS)).toEqual(fx.config);
  });
  it("Python's port of the mulberry32 initialiser draws the site's weights", () => {
    const w = initModelWeights(configFor(DEFAULT_PARAMS));
    expect(mismatch(w.tok_emb[0], fx.weights_sample.tok_emb_row0, 1e-15)).toBe(
      null,
    );
    expect(
      mismatch(
        w.blocks[1]!.ffn.W2[31],
        fx.weights_sample["block_1.ffn.W2_last_row"],
        1e-15,
      ),
    ).toBe(null);
  });
});

describe("embeddings animation", () => {
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
  it("pins key frames", () => {
    expect(embedCaption(want[0]!)).toBe(
      "Position 2 holds 'l', token id 11. The id is just a row number.",
    );
    expect(embedCaption(want[1]!)).toMatch(
      /^Lookup: id 11 selects row 11 of the embedding table E, a vector of 16 numbers/,
    );
    expect(embedCaption(want[2]!)).toBe(
      "The positional vector P[2] depends only on the position: P[2]₀ = sin(2) = 0.909, P[2]₁ = cos(2) = −0.416.",
    );
    expect(embedCaption(want[want.length - 1]!)).toBe(
      "Row 7 ('a') of X = E + P, the same lookup and add: 8 of 8 rows done.",
    );
  });
});

describe("attention animation", () => {
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
  it("pins key frames", () => {
    const want = fx.attention[0]!.states as unknown as AttnState[];
    const tokens = computeAttention(params("hello!")).tokens;
    const at = (phase: string) => want.find((s) => s.phase === phase)!;
    expect(attentionCaption(at("mask"), tokens)).toBe(
      "Head 0, query 3 'l': the causal mask sets the 4 future scores to −∞.",
    );
    expect(attentionCaption(at("sum"), tokens)).toMatch(
      /^Head 0, query 3 'l': the 4 exponentials sum to 3\.\d\d\.$/,
    );
    expect(attentionCaption(at("concat"), tokens)).toBe(
      "Concatenate the heads: each row of [head 0 ‖ head 1 ‖ …] has 16 numbers.",
    );
  });
});

describe("feed-forward animation", () => {
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
  it("pins key frames", () => {
    expect(ffnCaption(want[0]!)).toBe(
      "Position 5: the FFN reads x = LN₂(h), 16 numbers.",
    );
    expect(ffnCaption(want[1]!)).toMatch(
      /^Expand: u = x W₁ \+ b₁ has 32 numbers \(d_ff = 2·d\)/,
    );
    expect(ffnCaption(want[want.length - 1]!)).toMatch(
      /^\d+ of 32 neurons fired \(u > 0\) for this position/,
    );
  });
});

describe("overview hero", () => {
  const want = fx.overview.states as unknown as OverviewState[];
  const config = configFor(DEFAULT_PARAMS);
  const got = overviewStates(
    fx.overview.text,
    config,
    initModelWeights(config),
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
  it("pins key frames", () => {
    expect(overviewCaption(want[0]!)).toBe(
      "Position 4: the newest character 'o' is token id 14.",
    );
    expect(want.filter((s) => s.kind === "append").length).toBe(3);
    expect(overviewCaption(want[11]!)).toMatch(
      /^Round 2: the token just added, '.+' \(id \d+\), is now at position 5\.$/,
    );
  });
});
