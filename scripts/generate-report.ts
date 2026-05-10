/**
 * Render a Markdown report to a print-friendly PDF.
 *
 *   pnpm tsx scripts/generate-report.ts <input.md> <output.pdf>
 *
 * Uses `marked` (already in deps) for Markdown → HTML and Playwright's
 * built-in Chromium (already installed for e2e tests) for HTML → PDF.
 */
import { readFileSync } from "node:fs";
import path from "node:path";

import { chromium } from "@playwright/test";
import { marked } from "marked";

const STYLE = `
  @page {
    margin: 18mm 16mm 18mm 16mm;
    size: A4;
  }
  :root {
    --fg: #1a1a1a;
    --muted: #555;
    --rule: #ddd;
    --code-bg: #f4f4f4;
    --accent: #4f46e5;
    --pre-bg: #0f172a;
    --pre-fg: #e2e8f0;
  }
  * { box-sizing: border-box; }
  html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  body {
    font-family: 'Charter', 'Iowan Old Style', 'Georgia', serif;
    font-size: 10.5pt;
    line-height: 1.45;
    color: var(--fg);
    max-width: 100%;
    margin: 0;
  }
  h1 {
    font-size: 22pt;
    font-weight: 700;
    line-height: 1.15;
    margin: 0 0 0.5em;
    border-bottom: 2px solid var(--fg);
    padding-bottom: 0.2em;
  }
  h2 {
    font-size: 15pt;
    font-weight: 700;
    margin: 1.6em 0 0.4em;
    page-break-after: avoid;
  }
  h3 {
    font-size: 12pt;
    font-weight: 700;
    margin: 1.2em 0 0.3em;
    page-break-after: avoid;
  }
  h4 { font-size: 11pt; font-weight: 700; margin: 1em 0 0.3em; }
  p, ul, ol { margin: 0.4em 0 0.6em; }
  li { margin: 0.1em 0; }
  hr {
    border: 0;
    border-top: 1px solid var(--rule);
    margin: 1.6em 0;
  }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
  em { font-style: italic; }
  strong { font-weight: 700; }
  code {
    font-family: 'JetBrains Mono', 'Fira Code', Menlo, Consolas, monospace;
    font-size: 9pt;
    background: var(--code-bg);
    padding: 1px 4px;
    border-radius: 3px;
  }
  pre {
    background: var(--pre-bg);
    color: var(--pre-fg);
    padding: 10px 14px;
    border-radius: 4px;
    overflow-x: auto;
    font-size: 8.5pt;
    line-height: 1.35;
    page-break-inside: avoid;
    white-space: pre;
    margin: 0.6em 0 0.8em;
  }
  pre code {
    background: transparent;
    color: inherit;
    padding: 0;
    font-size: inherit;
  }
  blockquote {
    margin: 0.8em 0;
    padding: 0.4em 0.8em;
    border-left: 3px solid var(--accent);
    color: var(--muted);
    font-style: italic;
  }
  table {
    border-collapse: collapse;
    width: 100%;
    font-size: 9.5pt;
    margin: 0.7em 0 1em;
    page-break-inside: avoid;
  }
  th, td {
    border: 1px solid var(--rule);
    padding: 4px 8px;
    text-align: left;
    vertical-align: top;
  }
  th {
    background: #fafafa;
    font-weight: 700;
  }
  /* Avoid orphans on key elements */
  h1, h2, h3, h4 { break-after: avoid-page; }
  pre, table, blockquote { break-inside: avoid-page; }
`;

async function main() {
  const [, , inPath, outPath] = process.argv;
  if (!inPath || !outPath) {
    // eslint-disable-next-line no-console
    console.error(
      "usage: tsx scripts/generate-report.ts <input.md> <output.pdf>",
    );
    process.exit(2);
  }

  const md = readFileSync(inPath, "utf8");

  // The default Markdown renderer is fine; we don't need syntax
  // highlighting beyond monospace blocks.
  const bodyHtml = marked.parse(md, { async: false }) as string;

  const html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Transformer Explainer — Backend Tour</title>
  <style>${STYLE}</style>
</head>
<body>
${bodyHtml}
</body>
</html>`;

  const browser = await chromium.launch();
  const page = await browser.newPage({
    viewport: { width: 1024, height: 768 },
  });
  // file:// loads our temp HTML so relative styles / fonts work.
  await page.setContent(html, { waitUntil: "networkidle" });
  await page.pdf({
    path: outPath,
    format: "A4",
    printBackground: true,
    margin: { top: "18mm", right: "16mm", bottom: "18mm", left: "16mm" },
  });
  await browser.close();

  const abs = path.resolve(outPath);
  // eslint-disable-next-line no-console
  console.log(`wrote ${abs}`);
}

main().catch((err) => {
  // eslint-disable-next-line no-console
  console.error(err);
  process.exit(1);
});
