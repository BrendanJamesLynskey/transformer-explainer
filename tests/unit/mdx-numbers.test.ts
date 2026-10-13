/**
 * Brief 32: every number a chapter prints shows the precision its author
 * asked for. Each Decoder and Training chapter is compiled with the options
 * the site uses (`chapterMdxOptions`) and rendered to HTML with the real
 * `<R>` and every other component stubbed (children kept, so text nested in
 * a component is checked too).
 *
 * Before the fix next-mdx-remote's default `blockJS: true` deleted the
 * `d={3}` / `div={1e9}` props, so `<R>` printed raw floats such as
 * "0.5318892633600001": the long-float check below fails on that build.
 */
import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";

import { compileMDX } from "next-mdx-remote/rsc";
import { createElement, Fragment, type ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { R, toSignificant } from "@/components/training/R";
import { chapterMdxOptions } from "@/lib/mdx/options";

const CONTENT = path.resolve(__dirname, "../../content");
const chapters = ["decoder", "training"].flatMap((part) =>
  readdirSync(path.join(CONTENT, part))
    .filter((f) => f.endsWith(".mdx"))
    .map((f) => ({ part, file: f, path: path.join(CONTENT, part, f) })),
);

/** Every capitalised JSX name in the source, stubbed to render its children. */
function stubs(source: string): Record<string, unknown> {
  const names = new Set(
    [...source.matchAll(/<([A-Z][A-Za-z0-9]*)/g)].map((m) => m[1] ?? ""),
  );
  const map: Record<string, unknown> = {};
  for (const name of names) {
    map[name] = ({ children }: { children?: ReactNode }) =>
      createElement(Fragment, null, children);
  }
  map.R = R;
  return map;
}

/** The `d={n}` of each `<R … />` in source order (null: none, or `sep`). */
function requestedDecimals(source: string): (number | null)[] {
  return [...source.matchAll(/<R\s([^>]*?)\/>/g)].map((m) => {
    const props = m[1] ?? "";
    const d = /\bd=\{(\d+)\}/.exec(props);
    return d && !/\bsep\b/.test(props) ? Number(d[1]) : null;
  });
}

async function render(source: string): Promise<string> {
  const { content } = await compileMDX({
    source,
    options: chapterMdxOptions,
    components: stubs(source) as never,
  });
  return renderToStaticMarkup(content);
}

describe("chapter numbers (brief 32)", () => {
  it("finds all 7 Decoder and 16 Training chapters", () => {
    expect(chapters.filter((c) => c.part === "decoder")).toHaveLength(7);
    expect(chapters.filter((c) => c.part === "training")).toHaveLength(16);
  });

  it.each(chapters.map((c) => [`${c.part}/${c.file}`, c.path]))(
    "%s: no long raw float, and each <R d> prints d decimals",
    async (_name, file) => {
      const source = readFileSync(file, "utf-8");
      const html = await render(source);

      // Text only: KaTeX's MathML and attributes are not chapter text.
      const text = html
        .replace(/<annotation[\s\S]*?<\/annotation>/g, "")
        .replace(/<[^>]+>/g, " ");
      expect(text).not.toMatch(/\d\.\d{7,}/);

      const spans = [
        ...html.matchAll(/<span data-result="[^"]*">([^<]*)<\/span>/g),
      ].map((m) => m[1] ?? "");
      const wanted = requestedDecimals(source);
      expect(spans).toHaveLength(wanted.length);
      wanted.forEach((d, i) => {
        if (d === null) return;
        const printed = (spans[i] ?? "").replace(/%$/, "");
        const decimals = printed.includes(".")
          ? printed.split(".")[1]!.length
          : 0;
        expect({ i, printed, decimals }).toEqual({ i, printed, decimals: d });
      });
    },
  );

  it("sig: significant figures, × 10ⁿ for an exponent", () => {
    expect(toSignificant(9.411595741982881e-17, 2)).toBe("9.4 × 10⁻¹⁷");
    expect(toSignificant(7.610258360990496e-10, 2)).toBe("7.6 × 10⁻¹⁰");
    expect(toSignificant(1.5e21, 2)).toBe("1.5 × 10²¹");
    expect(toSignificant(0.4734391031174566, 2)).toBe("0.47");
  });

  it("keeps blockDangerousJS on", () => {
    expect(chapterMdxOptions.blockDangerousJS).toBe(true);
  });
});
