/**
 * Pure Markdown → safe HTML renderer for comment bodies.
 *
 * Extracted from `comments.ts` so unit tests can exercise it without
 * importing the DB client.
 *
 * Pipeline (the same unified / remark → rehype family the lesson pages use
 * through `next-mdx-remote`):
 *
 *   remark-parse      Markdown text  → mdast (Markdown syntax tree)
 *   remark-gfm        + GitHub extras: autolinks, tables, ~~strike~~
 *   remark-rehype     mdast          → hast (HTML syntax tree). Raw HTML in
 *                     the comment is *dropped* here (no `allowDangerousHtml`),
 *                     so `<script>`, `<img onerror>` and friends never become
 *                     elements at all.
 *   rehype-sanitize   allow-list the tree (GitHub's schema): only known tags
 *                     and attributes survive, and `href`/`src` keep only safe
 *                     protocols (no `javascript:`, no `data:`).
 *   relNoFollow       every link gets `rel="nofollow noopener noreferrer"`.
 *   rehype-stringify  hast → HTML string.
 *
 * Why not DOMPurify? `isomorphic-dompurify` needs jsdom on the server, and
 * jsdom's dependency chain (`html-encoding-sniffer` → the ESM-only
 * `@exodus/bytes`) can't be `require()`d by Vercel's function runtime: every
 * comment list with a visible comment returned 500 in production while local
 * and CI builds passed. This pipeline is plain JavaScript with no DOM, and
 * webpack bundles it into the route like the rest of the app.
 */
import rehypeSanitize, { defaultSchema } from "rehype-sanitize";
import rehypeStringify from "rehype-stringify";
import remarkGfm from "remark-gfm";
import remarkParse from "remark-parse";
import remarkRehype from "remark-rehype";
import { unified } from "unified";

/** `rel` added to every link: no SEO credit for user links, no `window.opener`. */
export const COMMENT_LINK_REL = ["nofollow", "noopener", "noreferrer"];

/**
 * Sanitiser allow-list. GitHub's default schema, minus `<input>` (task-list
 * checkboxes are meaningless in a comment) and with `src` limited to http(s)
 * — the default already rejects `data:` and `javascript:`; this just says so.
 */
const schema = {
  ...defaultSchema,
  tagNames: (defaultSchema.tagNames ?? []).filter((t) => t !== "input"),
  protocols: {
    ...defaultSchema.protocols,
    src: ["http", "https"],
  },
};

/** Just enough of a hast node for the walk below (no `@types/hast` needed). */
type HastNode = {
  type: string;
  tagName?: string;
  properties?: Record<string, unknown>;
  children?: HastNode[];
};

/** rehype plugin: give every `<a>` the `rel` above (runs after sanitising). */
function relNoFollow() {
  const walk = (node: HastNode): void => {
    if (node.type === "element" && node.tagName === "a") {
      node.properties = { ...node.properties, rel: COMMENT_LINK_REL };
    }
    node.children?.forEach(walk);
  };
  return (tree: HastNode) => walk(tree);
}

const processor = unified()
  .use(remarkParse)
  .use(remarkGfm)
  .use(remarkRehype)
  .use(rehypeSanitize, schema)
  .use(relNoFollow)
  .use(rehypeStringify);

/**
 * Render a comment's Markdown to sanitised HTML.
 *
 * Synchronous: every plugin in the pipeline is sync, so `processSync` works
 * and callers (the comment routes) can render before they touch the DB.
 *
 * @param md  Comment body as typed by the user (Markdown, ≤ 2000 chars).
 * @returns   HTML safe to pass to `dangerouslySetInnerHTML`.
 */
export function renderCommentHtml(md: string): string {
  return String(processor.processSync(md)).trim();
}
