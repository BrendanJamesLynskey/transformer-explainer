/**
 * Number and token formatting for the animations' captions and labels.
 *
 * Captions are built from model states by pure functions, and the frame
 * tests build the same caption from the Python reference's state, so the
 * formatting must be deterministic: three significant figures, a
 * typographic minus, and no locale-dependent separators.
 */

/** A typographic minus for negative numbers. */
export function minus(s: string): string {
  return s.replace(/^-/, "−");
}

/** `v` to `digits` significant figures, e.g. 0.0183, −1.27, 12.3. */
export function sig(v: number, digits = 3): string {
  if (v === 0 || Object.is(v, -0)) return "0";
  const a = Math.abs(v);
  if (a < 1e-4 || a >= 1e5) {
    const [m, e] = v.toExponential(digits - 1).split("e");
    return `${minus(String(Number(m)))}e${minus(String(Number(e)))}`;
  }
  return minus(String(Number(v.toPrecision(digits))));
}

/** A probability as a percentage with one decimal, e.g. 1.6%. */
export function pct(p: number): string {
  return `${(p * 100).toFixed(1)}%`;
}

/** A character as it should appear in a label: space and newline visible. */
export function showChar(c: string): string {
  if (c === " ") return "␣";
  if (c === "\n") return "↵";
  return c;
}

/** The Euclidean norm ‖v‖₂ (summed in index order). */
export function norm(v: readonly number[]): number {
  let s = 0;
  for (const x of v) s += x * x;
  return Math.sqrt(s);
}

/** The largest absolute value in a list (0 for an empty list). */
export function maxAbs(v: readonly (number | null)[]): number {
  let m = 0;
  for (const x of v) if (x !== null && Math.abs(x) > m) m = Math.abs(x);
  return m;
}
