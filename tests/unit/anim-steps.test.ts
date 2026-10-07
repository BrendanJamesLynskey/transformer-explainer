/**
 * The animation states are the model's own numbers: the values each
 * animation ends on equal the trace exactly (not just closely), and the
 * state lists have the shape the widgets rely on.
 */
import { describe, expect, it } from "vitest";

import {
  ATTN_HL,
  attentionCaption,
  attentionStates,
  concatHeads,
} from "@/lib/anim/attention-steps";
import { embedCaption, embedStates } from "@/lib/anim/embed-steps";
import { FFN_HL, ffnCaption, ffnStates, geluCurve } from "@/lib/anim/ffn-steps";
import { maxAbs, minus, norm, pct, showChar, sig } from "@/lib/anim/format";
import {
  OVERVIEW_HL,
  overviewCaption,
  overviewStates,
  promptIds,
  topTokens,
} from "@/lib/anim/overview-steps";
import {
  DEFAULT_PARAMS,
  computeAttention,
  computeEmbed,
  computeFfn,
  configFor,
  tracedForward,
} from "@/lib/compute/traces";
import { gelu } from "@/lib/transformer/gelu";
import { initModelWeights } from "@/lib/transformer/init";
import { matmul } from "@/lib/transformer/matmul";
import { encode } from "@/lib/transformer/tokenizer";

const P = { ...DEFAULT_PARAMS, text: "hello!" };

describe("format", () => {
  it("rounds to three significant figures with a typographic minus", () => {
    expect(sig(0.0183456)).toBe("0.0183");
    expect(sig(-1.2749)).toBe("−1.27");
    expect(sig(0)).toBe("0");
    expect(sig(-0)).toBe("0");
    expect(sig(1.5e-6)).toBe("1.5e−6");
    expect(sig(-2.5e7)).toBe("−2.5e7");
    expect(minus("-3")).toBe("−3");
    expect(pct(0.01646)).toBe("1.6%");
    expect(showChar(" ")).toBe("␣");
    expect(showChar("\n")).toBe("↵");
    expect(showChar("a")).toBe("a");
    expect(norm([3, 4])).toBe(5);
    expect(maxAbs([1, null, -3])).toBe(3);
    expect(maxAbs([])).toBe(0);
  });
});

describe("embeddings", () => {
  const tr = computeEmbed(P);
  const st = embedStates(tr, 2);
  it("walks the chosen position, element by element, then the other rows", () => {
    expect(st.map((s) => s.kind).slice(0, 4)).toEqual([
      "ids",
      "lookup",
      "position",
      "add",
    ]);
    expect(st).toHaveLength(3 + 16 + 7);
    const last = st[st.length - 1]!;
    expect(last.done.slice().sort()).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    // the sum shown is the model's x = E + P exactly
    expect(st[3 + 15]!.sum).toEqual(tr.xAfterEmb[2]);
    expect(st[1]!.tok).toEqual(tr.tokEmb[2]);
  });
  it("clamps the position into the sequence", () => {
    expect(embedStates(tr, 99)[0]!.pos).toBe(7);
    expect(embedStates(tr, -3)[0]!.pos).toBe(0);
    expect(st.every((s) => embedCaption(s).length > 0)).toBe(true);
  });
});

describe("attention", () => {
  const tr = computeAttention(P);
  const w = initModelWeights(configFor(P, { d_ff: 1, n_blocks: 1 }));
  it("ends each row on the trace's weights, and the softmax steps produce them", () => {
    const st = attentionStates(tr, { head: 1, row: 0 });
    for (const s of st.filter((x) => x.phase === "softmax")) {
      const want = tr.weights[1]![s.row]!;
      s.weights!.forEach((v, j) => expect(v).toBe(j <= s.row ? want[j] : null));
      // a_j = e_j / Σ e, exactly as softmax computes it
      s.exps!.forEach((e, j) =>
        expect(e === null ? null : e / s.sum!).toBe(s.weights![j]),
      );
    }
  });
  it("accumulates exactly the head's output, and W_o gives the trace's output", () => {
    const st = attentionStates(tr, { head: "all", row: 0 });
    const concat = concatHeads(tr);
    for (const s of st.filter((x) => x.phase === "wsum" && x.j === x.row)) {
      expect(s.out).toEqual(concat[s.row]!.slice(s.head * 8, s.head * 8 + 8));
    }
    expect(matmul(concat, w.blocks[0]!.attn.W_o)).toEqual(tr.output);
    const proj = st[st.length - 1]!;
    expect(proj.phase).toBe("project");
    expect(proj.projected).toEqual(tr.output);
  });
  it("has S dots + 5 softmax steps + (i + 1) value rows per row", () => {
    const st = attentionStates(tr, { head: 0, row: 3 });
    const rows = [3, 4, 5, 6, 7];
    expect(st).toHaveLength(rows.reduce((a, i) => a + 8 + 5 + i + 1, 0) + 2);
    expect(attentionStates(tr, { head: 9, row: 99 })[0]).toMatchObject({
      head: 1,
      row: 7,
    });
    // the dot products are the scores before scaling, in the same order
    const scale = st.find((s) => s.phase === "scale")!;
    scale.scores!.forEach((v, j) => expect(v).toBe(tr.scores[0]![3]![j]));
  });
  it("captions every phase, and highlights a term for each", () => {
    const st = attentionStates(tr, { head: 0, row: 7 });
    for (const s of st) {
      expect(attentionCaption(s, tr.tokens).length).toBeGreaterThan(10);
      expect(ATTN_HL[s.phase]).toBeTruthy();
    }
    const mask = st.find((s) => s.phase === "mask")!;
    expect(attentionCaption(mask, tr.tokens)).toMatch(/hides nothing/);
    const future = attentionStates(tr, { head: 0, row: 0 })[3]!;
    expect(attentionCaption(future, tr.tokens)).toMatch(/future key/);
    const mask1 = attentionStates(tr, { head: 0, row: 6 }).find(
      (s) => s.phase === "mask",
    )!;
    expect(attentionCaption(mask1, tr.tokens)).toMatch(/1 future score to/);
  });
});

describe("feed-forward", () => {
  const tr = computeFfn(P);
  const st = ffnStates(tr, 5);
  it("expands, applies GELU neuron by neuron, contracts", () => {
    expect(st).toHaveLength(2 + 32 + 2);
    const g = st.filter((s) => s.phase === "gelu");
    g.forEach((s, n) => {
      expect(s.neuron).toBe(n);
      expect(s.act![n]).toBe(gelu(s.pre![n]!));
      expect(s.act!.filter((v) => v !== null)).toHaveLength(n + 1);
    });
    const fired = st[st.length - 1]!;
    expect(fired.out).toEqual(tr.output[5]);
    expect(fired.fired).toEqual(
      tr.pre[5]!.flatMap((u, n) => (u > 0 ? [n] : [])),
    );
    for (const s of st) {
      expect(ffnCaption(s).length).toBeGreaterThan(10);
      expect(FFN_HL[s.phase]).toBeTruthy();
    }
    expect(ffnStates(tr, -1)[0]!.pos).toBe(0);
  });
  it("captions a squashed and a passed neuron", () => {
    const g = st.filter((s) => s.phase === "gelu");
    const caps = g.map(ffnCaption).join("\n");
    expect(caps).toMatch(/squashed towards 0/);
    expect(caps).toMatch(/passed \(nearly\) unchanged/);
  });
  it("samples the GELU curve", () => {
    const c = geluCurve(-3, 3, 7);
    expect(c).toHaveLength(7);
    expect(c[3]).toEqual([0, 0]);
    expect(c[6]![1]).toBeCloseTo(gelu(3));
  });
});

describe("overview hero", () => {
  const config = configFor(DEFAULT_PARAMS);
  const w = initModelWeights(config);
  it("follows the newest token through every block, then samples and appends", () => {
    const st = overviewStates("hello", config, w, { rounds: 3 });
    expect(st).toHaveLength(3 * 11);
    expect(st.slice(0, 11).map((s) => s.kind)).toEqual([
      "token",
      "embed",
      "position",
      "attn",
      "ffn",
      "attn",
      "ffn",
      "norm",
      "logits",
      "sample",
      "append",
    ]);
    // the stream after the last block is the model's, at the followed position
    const tr = tracedForward(
      encode("hello", 8).map((v, i) => (i < 5 ? v : 0)),
      config,
      w,
    );
    expect(st[6]!.vec).toEqual(tr.blocks[1]!.ffnOut[4]);
    expect(st[8]!.top![0]!.prob).toBeGreaterThan(st[8]!.top![4]!.prob);
    // each round starts one position further on, with the token just added
    expect(st[11]!.pos).toBe(5);
    expect(st[11]!.id).toBe(st[10]!.sampled);
    for (const s of st) {
      expect(overviewCaption(s).length).toBeGreaterThan(10);
      expect(OVERVIEW_HL[s.kind]).toBeTruthy();
    }
  });
  it("stops when the sequence is full, and keeps a prompt that leaves room", () => {
    expect(promptIds("abcdefghij", 8)).toHaveLength(7);
    expect(promptIds("", 8)).toEqual([36]);
    const st = overviewStates("abcdef", config, w, { rounds: 5 });
    expect(st.filter((s) => s.kind === "append")).toHaveLength(2);
  });
  it("ranks ties by id", () => {
    expect(topTokens([1, 1, 0], [0.4, 0.4, 0.2], 2).map((t) => t.id)).toEqual([
      0, 1,
    ]);
  });
});
