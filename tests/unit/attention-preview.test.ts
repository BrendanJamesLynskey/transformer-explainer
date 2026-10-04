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

  it("is deterministic for a seed and changes with the seed and head", () => {
    expect(attentionPreview({ text })).toEqual(p);
    expect(attentionPreview({ text, seed: 7 }).weights).not.toEqual(p.weights);
    expect(attentionPreview({ text, head: 1 }).weights).not.toEqual(p.weights);
  });

  it("returns an empty matrix for a head that doesn't exist", () => {
    expect(attentionPreview({ text: "ab", head: 5 }).weights).toEqual([]);
  });
});
