/**
 * Next.js config.
 *
 * MDX content is loaded at runtime via `next-mdx-remote` (per CLAUDE.md §3),
 * so we deliberately do NOT use `@next/mdx` for page-level MDX compilation.
 * Page extensions are the Next defaults plus none extra.
 *
 * @type {import('next').NextConfig}
 */
const nextConfig = {
  reactStrictMode: true,
  experimental: {
    typedRoutes: true,
  },
  // The site moved from Vercel's auto-assigned transformer-explainer-three.vercel.app to
  // transformer-decoder-explained.vercel.app (the GitHub OAuth callback now points there).
  // Send old links, path and query intact, to the new host.
  async redirects() {
    return [
      {
        source: "/:path*",
        has: [{ type: "host", value: "transformer-explainer-three.vercel.app" }],
        destination: "https://transformer-decoder-explained.vercel.app/:path*",
        permanent: true,
      },
    ];
  },
};

export default nextConfig;
