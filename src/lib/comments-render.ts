/**
 * Pure Markdown → safe HTML renderer for comment bodies.
 *
 * Extracted from `comments.ts` so unit tests can exercise it without
 * importing the DB client.
 *
 * `isomorphic-dompurify` initialises a jsdom instance at *import* time, and
 * jsdom tries to read its default stylesheet from a path that isn't
 * available during Next.js's build-time route data collection. Loading it
 * lazily — first call only, then cached — sidesteps that without changing
 * the public sync API.
 */
import { marked } from "marked";

type SanitizeFn = (
  dirty: string,
  cfg?: {
    USE_PROFILES?: { html?: boolean };
    FORBID_TAGS?: string[];
    FORBID_ATTR?: string[];
  },
) => string;

let sanitizeCache: SanitizeFn | null = null;
function getSanitize(): SanitizeFn {
  if (sanitizeCache) return sanitizeCache;
  // Synchronous require so `renderCommentHtml` can stay sync. The dynamic
  // path means Next's build-time module evaluator never touches jsdom.
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const mod = require("isomorphic-dompurify") as
    | { sanitize: SanitizeFn }
    | { default: { sanitize: SanitizeFn } };
  const purify = "sanitize" in mod ? mod : mod.default;
  sanitizeCache = purify.sanitize.bind(purify);
  return sanitizeCache;
}

export function renderCommentHtml(md: string): string {
  const rawHtml = marked.parse(md, { async: false }) as string;
  return getSanitize()(rawHtml, {
    USE_PROFILES: { html: true },
    FORBID_TAGS: ["style", "script"],
    FORBID_ATTR: ["style", "onerror", "onload", "onclick"],
  });
}
