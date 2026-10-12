/**
 * `<R k="defaultRun.evals.-1.val" d={3} />`: a number from
 * src/data/training/results.json (results.md), printed in the text, so no
 * number on a Training page is typed by hand. Negative indices count from
 * the end of an array. Server Component; throws at build time if the path
 * does not exist.
 */
import results from "@/data/training/results.json";

export function lookup(path: string): unknown {
  let v: unknown = results;
  for (const part of path.split(".")) {
    if (Array.isArray(v)) {
      const i = Number(part);
      v = v[i < 0 ? v.length + i : i];
    } else if (v !== null && typeof v === "object") {
      v = (v as Record<string, unknown>)[part];
    } else {
      v = undefined;
    }
    if (v === undefined) throw new Error(`results.json has no ${path}`);
  }
  return v;
}

export function R({
  k,
  d,
  pct,
  sep,
}: {
  k: string;
  /** decimals */
  d?: number;
  /** print as a percentage */
  pct?: boolean;
  /** thousands separators */
  sep?: boolean;
}): JSX.Element {
  const v = lookup(k);
  let s: string;
  if (typeof v === "number") {
    const x = pct ? v * 100 : v;
    s = sep
      ? x.toLocaleString("en-GB", { maximumFractionDigits: d ?? 0 })
      : d !== undefined
        ? x.toFixed(d)
        : String(x);
    if (pct) s += "%";
  } else {
    s = String(v);
  }
  return <span data-result={k}>{s}</span>;
}
