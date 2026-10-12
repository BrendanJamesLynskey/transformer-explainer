/*
 * src/lib/training/backward-ops.ts
 *
 * Operation: the backward pass of each of Part 1's elementary operations.
 * Shapes:    every function takes the upstream gradient dY (the same shape
 *            as the op's output) and what the forward pass used, and returns
 *            the gradients of the op's inputs and parameters.
 * Intuition: backprop is the chain rule applied one operation at a time,
 *            last to first. Each op only needs to know how its own output
 *            moves when its inputs move (its local derivative), multiplied
 *            by how the loss moves with its output (dY).
 *            Every formula here is checked against PyTorch autograd and
 *            finite differences in scripts/training/checks.py.
 * MDX:       /training/04-backprop.
 */
import { LAYERNORM_EPS } from "@/lib/transformer/layernorm";
import type { Matrix, Vector } from "@/lib/transformer/types";

/*
 * Matrix products for the backward pass. They add up in the same order as
 * matmul.ts would on the transposed inputs (k ascending), so the results
 * are identical, but they skip building the transposes: the backward pass
 * is most of a training step's time.
 */

/** A B, `[M, K] × [K, N]`. */
export function mm(A: Matrix, B: Matrix): Matrix {
  const N = B[0]?.length ?? 0;
  return A.map((a) => {
    const c = new Array<number>(N).fill(0);
    for (let k = 0; k < a.length; k++) {
      const x = a[k]!;
      const b = B[k]!;
      for (let j = 0; j < N; j++) c[j] = c[j]! + x * b[j]!;
    }
    return c;
  });
}

/** Aᵀ B, `[K, M]ᵀ × [K, N]` → `[M, N]` (e.g. dW = Xᵀ dY). */
export function mmTA(A: Matrix, B: Matrix): Matrix {
  const M = A[0]?.length ?? 0;
  const N = B[0]?.length ?? 0;
  const C: Matrix = Array.from({ length: M }, () =>
    new Array<number>(N).fill(0),
  );
  for (let i = 0; i < M; i++) {
    const c = C[i]!;
    for (let k = 0; k < A.length; k++) {
      const x = A[k]![i]!;
      const b = B[k]!;
      for (let j = 0; j < N; j++) c[j] = c[j]! + x * b[j]!;
    }
  }
  return C;
}

/** A Bᵀ, `[M, K] × [N, K]ᵀ` → `[M, N]` (e.g. dX = dY Wᵀ). */
export function mmTB(A: Matrix, B: Matrix): Matrix {
  return A.map((a) => {
    const c = new Array<number>(B.length).fill(0);
    for (let k = 0; k < a.length; k++) {
      const x = a[k]!;
      for (let j = 0; j < B.length; j++) c[j] = c[j]! + x * B[j]![k]!;
    }
    return c;
  });
}

/**
 * Backward of Y = X W (a linear layer without bias).
 *
 * @param X   Input `[S, I]` (saved by the forward pass).
 * @param W   Weight `[I, O]`.
 * @param dY  Upstream gradient `[S, O]`.
 * @returns   dX = dY Wᵀ `[S, I]` and dW = Xᵀ dY `[I, O]`.
 */
export function linearBackward(
  X: Matrix,
  W: Matrix,
  dY: Matrix,
): { dX: Matrix; dW: Matrix } {
  return { dX: mmTB(dY, W), dW: mmTA(X, dY) };
}

/** Column sums of a matrix: the gradient of a bias added to every row. */
export function colSums(m: Matrix): Vector {
  const n = m[0]?.length ?? 0;
  const out = new Array<number>(n).fill(0);
  for (const row of m) for (let j = 0; j < n; j++) out[j]! += row[j]!;
  return out;
}

/**
 * Backward of LayerNorm (layernorm.ts), one row at a time.
 *
 * Forward: x̂ = (x − μ)·r with r = 1/√(σ² + ε); y = γ ⊙ x̂ + β.
 * Backward (Ba et al. 2016; the standard derivation):
 *   dγ = Σ_rows dy ⊙ x̂,   dβ = Σ_rows dy,   g = dy ⊙ γ,
 *   dx = r · (g − mean(g) − x̂ · mean(g ⊙ x̂)).
 *
 * @param X      The LayerNorm's input `[S, D]` (μ, σ² and x̂ are recomputed
 *               exactly as layernorm.ts computes them).
 * @param gamma  γ `[D]`.
 * @param dY     Upstream gradient `[S, D]`.
 */
export function layernormBackward(
  X: Matrix,
  gamma: Vector,
  dY: Matrix,
): { dX: Matrix; dGamma: Vector; dBeta: Vector } {
  const D = gamma.length;
  const dGamma = new Array<number>(D).fill(0);
  const dBeta = new Array<number>(D).fill(0);
  const dX: Matrix = new Array<Vector>(X.length);
  for (let i = 0; i < X.length; i++) {
    const x = X[i]!;
    const dy = dY[i]!;
    let mean = 0;
    for (let k = 0; k < D; k++) mean += x[k]!;
    mean /= D;
    let varSum = 0;
    for (let k = 0; k < D; k++) {
      const d = x[k]! - mean;
      varSum += d * d;
    }
    const r = 1 / Math.sqrt(varSum / D + LAYERNORM_EPS);
    const xhat = new Array<number>(D);
    const g = new Array<number>(D);
    let gMean = 0;
    let gxMean = 0;
    for (let k = 0; k < D; k++) {
      xhat[k] = (x[k]! - mean) * r;
      g[k] = dy[k]! * gamma[k]!;
      dGamma[k]! += dy[k]! * xhat[k]!;
      dBeta[k]! += dy[k]!;
      gMean += g[k]!;
      gxMean += g[k]! * xhat[k]!;
    }
    gMean /= D;
    gxMean /= D;
    const row = new Array<number>(D);
    for (let k = 0; k < D; k++) {
      row[k] = r * (g[k]! - gMean - xhat[k]! * gxMean);
    }
    dX[i] = row;
  }
  return { dX, dGamma, dBeta };
}

const C_GELU = Math.sqrt(2 / Math.PI);

/**
 * GELU's derivative (tanh form, gelu.ts), at u:
 *   t = tanh(c (u + 0.044715 u³)),  c = √(2/π)
 *   GELU'(u) = ½(1 + t) + ½ u (1 − t²) c (1 + 3·0.044715 u²)
 */
export function geluGrad(u: number): number {
  const t = Math.tanh(C_GELU * (u + 0.044715 * u * u * u));
  return (
    0.5 * (1 + t) + 0.5 * u * (1 - t * t) * C_GELU * (1 + 3 * 0.044715 * u * u)
  );
}

/** Backward of the element-wise GELU: dU = dY ⊙ GELU'(U). */
export function geluBackward(U: Matrix, dY: Matrix): Matrix {
  return U.map((row, i) => row.map((u, j) => dY[i]![j]! * geluGrad(u)));
}

/**
 * Backward of a row softmax P = softmax(S) (masked positions have P = 0, so
 * they get no gradient):
 *   dS_ij = P_ij (dP_ij − Σ_k dP_ik P_ik).
 */
export function softmaxBackward(P: Matrix, dP: Matrix): Matrix {
  return P.map((p, i) => {
    const dp = dP[i]!;
    let dot = 0;
    for (let k = 0; k < p.length; k++) dot += dp[k]! * p[k]!;
    return p.map((pk, k) => pk * (dp[k]! - dot));
  });
}

/** Element-wise sum of two same-shape matrices into a new matrix. */
export function addInto(a: Matrix, b: Matrix): Matrix {
  return a.map((row, i) => row.map((v, j) => v + b[i]![j]!));
}
