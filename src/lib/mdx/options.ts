/**
 * The options every chapter's MDX is compiled with (ChapterArticle, and the
 * unit test that renders every chapter: tests/unit/mdx-numbers.test.ts).
 *
 * `blockJS: false`: next-mdx-remote 6 defaults to `blockJS: true`, which
 * deletes every JavaScript expression from the MDX, including expression
 * props such as `<R k="…" d={3} />`. The component then never sees `d`, and
 * printed the raw float ("0.5318892633600001 s") instead of the 3 decimals
 * the chapter asked for (brief 32). The chapters are first-party files in
 * this repo, reviewed like the rest of the code, so expressions are allowed;
 * `blockDangerousJS: true` (the default, stated here so it stays on) still
 * rejects `eval`, `Function`, `process`, `require`, `constructor` and the
 * like.
 *
 * `remark-gfm` (as on the sister sites): GitHub tables, so chapter 9's
 * table of number formats renders as a table, not as lines of pipes.
 * Maths: `remark-math` turns `$…$` / `$$…$$` into maths nodes and
 * `rehype-katex` renders them to HTML on the server.
 */
import type { MDXRemoteProps } from "next-mdx-remote/rsc";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

export const chapterMdxOptions: NonNullable<MDXRemoteProps["options"]> = {
  blockJS: false,
  blockDangerousJS: true,
  mdxOptions: {
    remarkPlugins: [remarkGfm, remarkMath],
    rehypePlugins: [rehypeKatex],
  },
};
