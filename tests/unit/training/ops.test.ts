/** Backward building blocks: products, GELU', softmax backward, params. */
import { describe, expect, it } from "vitest";

import {
  addInto,
  colSums,
  geluGrad,
  mm,
  mmTA,
  mmTB,
  softmaxBackward,
} from "@/lib/training/backward-ops";
import { crossEntropy, crossEntropyGrad } from "@/lib/training/loss";
import {
  cloneParams,
  countParams,
  fromWeights,
  paramNames,
  toWeights,
  zerosLike,
} from "@/lib/training/params";
import { PART1_MODEL } from "@/lib/training/trainer";
import { gelu } from "@/lib/transformer/gelu";
import { initModelWeights } from "@/lib/transformer/init";
import { matmul, transpose } from "@/lib/transformer/matmul";
import { mulberry32 } from "@/lib/transformer/random";

function rand(r: number, c: number, seed: number): number[][] {
  const g = mulberry32(seed);
  return Array.from({ length: r }, () =>
    Array.from({ length: c }, () => g() * 2 - 1),
  );
}

describe("the backward pass's matrix products", () => {
  const A = rand(5, 7, 1);
  const B = rand(7, 3, 2);
  const C = rand(5, 3, 3);
  it("mm = matmul, exactly", () => expect(mm(A, B)).toEqual(matmul(A, B)));
  it("mmTA = matmul(transpose(A), B), exactly", () =>
    expect(mmTA(A, C)).toEqual(matmul(transpose(A), C)));
  it("mmTB = matmul(A, transpose(B)), exactly", () =>
    expect(mmTB(C, B)).toEqual(matmul(C, transpose(B))));
  it("colSums and addInto", () => {
    expect(
      colSums([
        [1, 2],
        [3, 4],
      ]),
    ).toEqual([4, 6]);
    expect(addInto([[1, 2]], [[3, 4]])).toEqual([[4, 6]]);
  });
});

describe("GELU' and the softmax backward", () => {
  it("GELU' matches a central difference", () => {
    for (const u of [-3, -1, -0.2, 0, 0.3, 1.5, 4]) {
      const h = 1e-6;
      const fd = (gelu(u + h) - gelu(u - h)) / (2 * h);
      expect(Math.abs(geluGrad(u) - fd)).toBeLessThan(1e-8);
    }
  });
  it("softmax backward: each row of dS sums to 0", () => {
    const P = [
      [0.2, 0.3, 0.5],
      [1, 0, 0],
    ];
    const dS = softmaxBackward(P, [
      [1, 2, 3],
      [5, 6, 7],
    ]);
    for (const row of dS)
      expect(Math.abs(row.reduce((a, b) => a + b))).toBeLessThan(1e-15);
    expect(dS[1]).toEqual([0, 0, 0]); // a one-hot row passes nothing back
  });
});

describe("cross-entropy", () => {
  it("is ln V for equal logits, with gradient (1/V − y)/n", () => {
    const { perPosition, probs } = crossEntropy([[0, 0, 0, 0]], [2]);
    expect(perPosition[0]).toBeCloseTo(Math.log(4), 15);
    expect(crossEntropyGrad(probs, [2], 2)).toEqual([
      [0.125, 0.125, -0.375, 0.125],
    ]);
  });
});

describe("ParamMap", () => {
  const w = initModelWeights(PART1_MODEL);
  const p = fromWeights(w);
  it("lists every parameter in paramNames order: 5,376 numbers", () => {
    expect(Object.keys(p)).toEqual(paramNames(PART1_MODEL));
    expect(countParams(p)).toBe(5376);
  });
  it("round-trips through toWeights", () => expect(toWeights(p, 2)).toEqual(w));
  it("clones deeply and builds zeros", () => {
    const c = cloneParams(p);
    (c["tok_emb"] as number[][])[0]![0] = 99;
    expect((p["tok_emb"] as number[][])[0]![0]).not.toBe(99);
    expect(countParams(zerosLike(p))).toBe(5376);
    expect((zerosLike(p)["ln1"] ?? [0])[0]).toBe(0);
  });
});
