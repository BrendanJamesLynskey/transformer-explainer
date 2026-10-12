/** Batches, the split and BPE against scripts/training/data.py. */
import { describe, expect, it } from "vitest";

import bx from "../fixtures/training/bpe.json";
import gx from "../fixtures/training/gradients.json";
import results from "@/data/training/results.json";

import { CORPUS_IDS, CORPUS_TEXT } from "./corpus";

import { bpeEncode, bpeTrain, mergeWord, wordsOf } from "@/lib/training/bpe";
import { encodeChars, sampleBatch, splitIds, valBatch } from "@/lib/training/data";
import { mulberry32 } from "@/lib/transformer/random";

describe("the corpus and its split", () => {
  it("is the corpus results.md describes", () => {
    expect(CORPUS_TEXT.length).toBe(results.corpus.chars);
    const { train, val } = splitIds(CORPUS_IDS);
    expect(train.length).toBe(results.corpus.trainChars);
    expect(val.length).toBe(results.corpus.valChars);
  });
  it("draws the same windows as Python (mulberry32 starts)", () => {
    const { train } = splitIds(CORPUS_IDS);
    const b = sampleBatch(train, 3, 8, mulberry32(7));
    expect(b.inputs).toEqual(gx.inputs);
    expect(b.targets).toEqual(gx.targets);
    expect(b.inputs[0]!.slice(1)).toEqual(b.targets[0]!.slice(0, 7)); // teacher forcing
  });
  it("validates on 64 fixed windows", () => {
    const { val } = splitIds(CORPUS_IDS);
    const v = valBatch(val, 8);
    expect(v.inputs).toHaveLength(64);
    expect(v.starts[0]).toBe(0);
    expect(valBatch(val, 8)).toEqual(v);
  });
  it("refuses characters outside the alphabet", () => {
    expect(() => encodeChars("A")).toThrow(/alphabet/);
  });
});

describe("BPE = the Python reference", () => {
  const model = bpeTrain(CORPUS_TEXT, bx.merges.length);
  it("learns the same 64 merges, in the same order, with the same counts", () => {
    expect(model.merges).toEqual(bx.merges);
    expect(model.vocab).toEqual(bx.vocab);
    expect(model.history).toEqual(bx.history);
  });
  it("encodes the corpus and the excerpt to the same tokens", () => {
    expect(bpeEncode(CORPUS_TEXT, model.merges)).toHaveLength(bx.corpusTokens);
    expect(bpeEncode(bx.excerpt, model.merges)).toEqual(bx.excerptIds);
  });
  it("words keep their leading space; merges do not overlap", () => {
    expect(wordsOf("a  bc\nd")).toEqual(["a", " ", " bc", "\n", "d"]);
    expect(mergeWord([1, 1, 1], 1, 1, 9)).toEqual([9, 1]);
    expect(bpeTrain("", 3).merges).toEqual([]);
  });
});
