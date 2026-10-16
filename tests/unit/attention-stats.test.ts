/**
 * The trained heads' statistics the landing page and chapter 3 quote
 * (src/data/decoder/trained-heads.json, written by scripts/reference.py)
 * equal the TypeScript model's own measurement on the same corpus windows.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import heads from "@/data/decoder/trained-heads.json";
import { mismatch } from "./helpers/close";

import { headStats } from "@/lib/attention-stats";
import { configFor, DEFAULT_PARAMS } from "@/lib/compute/traces";
import { initModelWeights } from "@/lib/transformer/init";
import { trainedWeights } from "@/lib/transformer/trained";

const corpus = readFileSync(
  join(__dirname, "../../public/training/corpus.txt"),
  "utf8",
);

describe("trained-heads.json", () => {
  const trained = headStats(trainedWeights(), corpus, heads.stride);
  it("matches the TypeScript model on the trained weights", () => {
    expect(mismatch(trained, heads.trained, 1e-12)).toBe(null);
  });
  it("matches the TypeScript model at random init", () => {
    const random = headStats(
      initModelWeights(configFor(DEFAULT_PARAMS)),
      corpus,
      heads.stride,
    );
    expect(mismatch(random, heads.random, 1e-12)).toBe(null);
  });
  it("says what the landing page says: local heads in block 1, a start-of-window head for spaces in block 2", () => {
    const at = (b: number, h: number) =>
      heads.trained.find((x) => x.block === b && x.head === h)!;
    expect(at(0, 0).self + at(0, 0).prev).toBeGreaterThan(0.75);
    expect(at(0, 1).self + at(0, 1).prev).toBeGreaterThan(0.85);
    expect(at(1, 1).spaceStart).toBeGreaterThan(0.8);
    heads.random.forEach((x) => expect(x.entropy).toBeGreaterThan(0.99));
    heads.trained.forEach((x) => expect(x.entropy).toBeLessThan(0.9));
  });
  it("returns zero for a head with no space rows", () => {
    const s = headStats(trainedWeights(), "abcdefghij", 1);
    expect(s[0]!.spaceRows).toBe(0);
    expect(s[0]!.spaceStart).toBe(0);
  });
});
