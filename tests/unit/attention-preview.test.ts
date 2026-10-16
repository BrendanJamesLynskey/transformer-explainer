import { describe, expect, it } from "vitest";

import { attentionPreview } from "@/lib/attention-preview";

describe("attentionPreview", () => {
  const text = "the cat sat";
  const p = attentionPreview({ text });

  it("labels one token per character", () => {
    expect(p.tokens.join("")).toBe(text);
    expect(p.weights).toHaveLength(text.length);
  });

  it("returns causal softmax weights: rows sum to 1, nothing above the diagonal", () => {
    p.weights.forEach((row, i) => {
      expect(row).toHaveLength(text.length);
      expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
      row.forEach((w, j) => {
        if (j > i) expect(w).toBe(0);
        else expect(w).toBeGreaterThan(0);
      });
    });
  });

  it("is deterministic, and changes with the weights, the seed and the head", () => {
    expect(attentionPreview({ text })).toEqual(p);
    const r = attentionPreview({ text, kind: "random" });
    expect(r.weights).not.toEqual(p.weights);
    // the seed only seeds random init
    expect(attentionPreview({ text, seed: 7 })).toEqual(p);
    expect(
      attentionPreview({ text, kind: "random", seed: 7 }).weights,
    ).not.toEqual(r.weights);
    expect(attentionPreview({ text, head: 1 }).weights).not.toEqual(p.weights);
  });

  it("returns an empty matrix for a head that doesn't exist", () => {
    expect(attentionPreview({ text: "ab", head: 5 }).weights).toEqual([]);
  });
});
