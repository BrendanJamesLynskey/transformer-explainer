/**
 * /robots.txt — everything may be crawled except the admin and API routes;
 * points at this address's own sitemap (see /sitemap.xml).
 */
import { sitemapOrigin } from "@/lib/site/sitemap";

export const dynamic = "force-dynamic";

export function GET(request: Request): Response {
  const origin = sitemapOrigin(request.headers.get("host"));
  const body = [
    "User-agent: *",
    "Allow: /",
    "Disallow: /admin",
    "Disallow: /api/",
    "",
    `Sitemap: ${origin}/sitemap.xml`,
    "",
  ].join("\n");
  return new Response(body, {
    headers: { "content-type": "text/plain; charset=utf-8" },
  });
}
