/**
 * Deep comparison with a numeric tolerance: every number in `got` must be
 * within `tol` (absolute, or relative for large values) of the number at the
 * same place in `want`; everything else (nulls, strings, ids, array lengths,
 * keys) must be identical. Returns the first mismatch's path, or null.
 */
export function mismatch(
  got: unknown,
  want: unknown,
  tol = 1e-12,
  path = "$",
): string | null {
  if (typeof want === "number" && typeof got === "number") {
    if (Number.isInteger(want) && Number.isInteger(got) && Math.abs(want) > 1) {
      return got === want ? null : `${path}: ${got} ≠ ${want}`;
    }
    const scale = Math.max(1, Math.abs(want));
    return Math.abs(got - want) <= tol * scale
      ? null
      : `${path}: ${got} vs ${want} (|Δ| = ${Math.abs(got - want)})`;
  }
  if (Array.isArray(want)) {
    if (!Array.isArray(got) || got.length !== want.length)
      return `${path}: length ${Array.isArray(got) ? got.length : "n/a"} ≠ ${want.length}`;
    for (let i = 0; i < want.length; i++) {
      const m = mismatch(got[i], want[i], tol, `${path}[${i}]`);
      if (m) return m;
    }
    return null;
  }
  if (want !== null && typeof want === "object") {
    if (got === null || typeof got !== "object")
      return `${path}: not an object`;
    const g = got as Record<string, unknown>;
    const w = want as Record<string, unknown>;
    const keys = new Set([...Object.keys(g), ...Object.keys(w)]);
    for (const k of keys) {
      if (!(k in w)) return `${path}.${k}: unexpected key`;
      if (!(k in g)) return `${path}.${k}: missing key`;
      const m = mismatch(g[k], w[k], tol, `${path}.${k}`);
      if (m) return m;
    }
    return null;
  }
  return got === want ? null : `${path}: ${String(got)} ≠ ${String(want)}`;
}
