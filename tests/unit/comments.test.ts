import { describe, expect, it } from "vitest";

import { renderCommentHtml } from "@/lib/comments-render";

describe("renderCommentHtml", () => {
  it("renders Markdown to HTML", () => {
    const html = renderCommentHtml("**bold** and *italic*");
    expect(html).toMatch(/<strong>bold<\/strong>/);
    expect(html).toMatch(/<em>italic<\/em>/);
  });

  it("renders code, lists and GFM extras", () => {
    const html = renderCommentHtml(
      "Use `softmax(x)`\n\n- one\n- two\n\n~~old~~ https://example.com",
    );
    expect(html).toMatch(/<code>softmax\(x\)<\/code>/);
    expect(html).toMatch(/<li>one<\/li>/);
    expect(html).toMatch(/<del>old<\/del>/);
    expect(html).toMatch(/<a href="https:\/\/example.com"/);
  });

  it("strips raw <script> tags", () => {
    // Inline raw HTML tags are dropped one by one, so the script's source
    // survives only as inert, escaped text inside the paragraph.
    const html = renderCommentHtml("Hi <script>alert('xss')</script> there");
    expect(html).not.toMatch(/<script/i);
    expect(html).toMatch(/Hi/);
    expect(html).toMatch(/there/);
  });

  it("drops a block-level <script> entirely", () => {
    const html = renderCommentHtml("Hi\n\n<script>\nalert('xss')\n</script>");
    expect(html).toBe("<p>Hi</p>");
  });

  it("drops raw HTML elements with inline event handlers", () => {
    for (const evil of [
      '<img src="x" onerror="alert(1)" />',
      '<a href="https://example.com" onclick="alert(1)">x</a>',
      '<div onmouseover="alert(1)">hover</div>',
      '<svg onload="alert(1)"></svg>',
    ]) {
      const html = renderCommentHtml(evil);
      expect(html).not.toMatch(/\son\w+=/i);
      expect(html).not.toMatch(/<(img|svg|div)/i);
    }
  });

  it("strips <style>, <iframe> and style attributes", () => {
    const html = renderCommentHtml(
      '<style>body{display:none}</style><iframe src="https://evil.example"></iframe><p style="color:red">x</p>',
    );
    expect(html).not.toMatch(/<style|<iframe|style=/i);
  });

  it("blocks javascript: URLs in links", () => {
    for (const md of [
      "[bad](javascript:alert(1))",
      "[bad](JaVaScRiPt:alert(1))",
      "[bad](  javascript:alert(1))",
      "<javascript:alert(1)>",
    ]) {
      const html = renderCommentHtml(md);
      // The text may still read "javascript:…"; no attribute may carry it.
      expect(html).not.toMatch(/(href|src)="\s*javascript:/i);
      expect(html).not.toMatch(/<a [^>]*href=/i);
    }
  });

  it("blocks data: and javascript: image sources", () => {
    const dataImg = renderCommentHtml(
      "![x](data:image/svg+xml;base64,PHN2ZyBvbmxvYWQ9YWxlcnQoMSk+)",
    );
    expect(dataImg).not.toMatch(/data:/i);
    const jsImg = renderCommentHtml("![x](javascript:alert(1))");
    expect(jsImg).not.toMatch(/src="\s*javascript:/i);
  });

  it("keeps https images", () => {
    const html = renderCommentHtml("![chart](https://example.com/a.png)");
    expect(html).toMatch(/<img src="https:\/\/example.com\/a.png"/);
  });

  it("gives every link rel=nofollow noopener noreferrer", () => {
    const html = renderCommentHtml(
      "[docs](https://example.com) and https://example.org",
    );
    const anchors = html.match(/<a [^>]*>/g) ?? [];
    expect(anchors).toHaveLength(2);
    for (const a of anchors) {
      expect(a).toMatch(/rel="nofollow noopener noreferrer"/);
    }
  });

  it("doesn't let a comment add task-list checkboxes or form inputs", () => {
    const html = renderCommentHtml("- [x] done\n\n<input value=1>");
    expect(html).not.toMatch(/<input/i);
  });

  it("returns an empty string for whitespace", () => {
    expect(renderCommentHtml("   \n")).toBe("");
  });
});
