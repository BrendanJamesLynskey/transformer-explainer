/**
 * `<R k="defaultRun.evals.-1.val" d={3} />`: a number from
 * src/data/training/results.json (results.md), printed in the text, so no
 * number on a Training page is typed by hand. Negative indices count from
 * the end of an array. Server Component; throws at build time if the path
 * does not exist.
 *
 * The numeric props are MDX expressions (`d={3}`), which reach this
 * component only because the chapters compile with `blockJS: false`
 * (src/lib/mdx/options.ts; brief 32). `sig={2}` prints tiny or huge values
 * to 2 significant figures in scientific form: "9.4 × 10⁻¹⁷".
 */
import results from "@/data/training/results.json";

const SUPERSCRIPT: Record<string, string> = {
  "-": "⁻",
  "+": "",
  "0": "⁰",
  "1": "¹",
  "2": "²",
  "3": "³",
  "4": "⁴",
  "5": "⁵",
  "6": "⁶",
  "7": "⁷",
  "8": "⁸",
  "9": "⁹",
};

/** `x` to `sig` significant figures; an exponent is written × 10ⁿ. */
export function toSignificant(x: number, sig: number): string {
  const [mantissa, exponent] = x.toPrecision(sig).split("e");
  if (exponent === undefined) return mantissa ?? "";
  const sup = [...exponent].map((c) => SUPERSCRIPT[c] ?? c).join("");
  return `${mantissa} × 10${sup}`;
}

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

/** How a number is printed: `<R>`'s props, shared with `<Check>` (brief 44). */
export type Format = {
  /** decimals */
  d?: number;
  /** print as a percentage */
  pct?: boolean;
  /** thousands separators */
  sep?: boolean;
  /** divide by this first (e.g. 1e9 to print bytes as GB, 3600 seconds as hours) */
  div?: number;
  /** significant figures, scientific form for small or large values */
  sig?: number;
};

/** The number as printed (before any "%"), and its value in those units. */
export function scaled(v: number, { pct, div }: Format): number {
  return (pct ? v * 100 : v) / (div ?? 1);
}

/** `v` printed as `<R>` prints it. */
export function formatValue(v: unknown, f: Format): string {
  if (typeof v !== "number") return String(v);
  const x = scaled(v, f);
  const { d, sep, sig } = f;
  let s =
    sig !== undefined
      ? toSignificant(x, sig)
      : sep
        ? x.toLocaleString("en-GB", { maximumFractionDigits: d ?? 0 })
        : d !== undefined
          ? x.toFixed(d)
          : String(x);
  if (f.pct) s += "%";
  return s;
}

export function R({ k, ...f }: { k: string } & Format): JSX.Element {
  return <span data-result={k}>{formatValue(lookup(k), f)}</span>;
}
