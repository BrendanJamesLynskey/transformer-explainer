import { describe, expect, it } from "vitest";

import { renderCommentHtml } from "@/lib/comments-render";

describe("renderCommentHtml", () => {
  it("renders Markdown to HTML", () => {
    const html = renderCommentHtml("**bold** and *italic*");
    expect(html).toMatch(/<strong>bold<\/strong>/);
    expect(html).toMatch(/<em>italic<\/em>/);
  });

  it("strips raw <script> tags", () => {
    const html = renderCommentHtml("Hi <script>alert('xss')</script> there");
    expect(html).not.toMatch(/<script/i);
    expect(html).toMatch(/Hi/);
  });

  it("strips inline event handler attributes from raw HTML", () => {
    // Even if Markdown-as-HTML somehow lets a tag through, DOMPurify must
    // drop onclick/onload/onerror.
    const html = renderCommentHtml('<img src="x" onerror="alert(1)" />');
    expect(html).not.toMatch(/onerror=/i);
  });

  it("blocks javascript: URLs in links", () => {
    const html = renderCommentHtml("[bad](javascript:alert(1))");
    expect(html).not.toMatch(/javascript:alert/i);
  });
});
