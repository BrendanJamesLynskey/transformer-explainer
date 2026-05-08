/**
 * Pure Markdown → safe HTML renderer for comment bodies.
 *
 * Extracted from `comments.ts` so unit tests can exercise it without
 * importing the DB client.
 */
import DOMPurify from "isomorphic-dompurify";
import { marked } from "marked";

export function renderCommentHtml(md: string): string {
  const rawHtml = marked.parse(md, { async: false }) as string;
  return DOMPurify.sanitize(rawHtml, {
    USE_PROFILES: { html: true },
    FORBID_TAGS: ["style", "script"],
    FORBID_ATTR: ["style", "onerror", "onload", "onclick"],
  });
}
