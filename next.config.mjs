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
};

export default nextConfig;
