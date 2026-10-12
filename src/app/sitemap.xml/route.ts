/**
 * /sitemap.xml — one sitemap per address (src/lib/site/hosts.ts): the
 * training host lists Part 2's pages at the training host, every other host
 * lists Part 1's pages at the decoder host. Each page is listed only at its
 * canonical address.
 */
import { SECTIONS, TRAINING_SECTIONS } from "@/lib/mdx/sections";
import { sitemapUrls } from "@/lib/site/sitemap";

export const dynamic = "force-dynamic";

export function GET(request: Request): Response {
  const urls = sitemapUrls(
    request.headers.get("host"),
    SECTIONS.map((s) => s.slug),
    TRAINING_SECTIONS.map((s) => s.slug),
  );
  const body =
    `<?xml version="1.0" encoding="UTF-8"?>\n` +
    `<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n` +
    urls.map((u) => `  <url><loc>${u}</loc></url>\n`).join("") +
    `</urlset>\n`;
  return new Response(body, {
    headers: { "content-type": "application/xml; charset=utf-8" },
  });
}
