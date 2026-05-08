/**
 * /learn/[slug] — render an MDX section with the three-layer toggle pinned
 * at the top.
 *
 * Server Component. Validates the slug against the catalogue, reads the
 * MDX from disk, and hands it to `next-mdx-remote/rsc` with our components
 * map.
 */
import Link from "next/link";
import { notFound } from "next/navigation";
import { MDXRemote } from "next-mdx-remote/rsc";

import { CommentSection } from "@/components/interactive/CommentSection";
import { EventTracker } from "@/components/interactive/EventTracker";
import { LayerToggle } from "@/components/interactive/LayerToggle";
import { ProgressTracker } from "@/components/interactive/ProgressTracker";
import { getSession } from "@/lib/auth";
import { mdxComponents } from "@/lib/mdx/components";
import {
  SECTIONS,
  getSectionMeta,
  isValidSlug,
  readSectionMdx,
} from "@/lib/mdx/sections";

export async function generateStaticParams(): Promise<{ slug: string }[]> {
  return SECTIONS.map((s) => ({ slug: s.slug }));
}

export async function generateMetadata({
  params,
}: {
  params: { slug: string };
}) {
  if (!isValidSlug(params.slug)) return {};
  const meta = getSectionMeta(params.slug);
  return {
    title: meta.title,
    description: meta.summary,
  };
}

export default async function SectionPage({
  params,
}: {
  params: { slug: string };
}): Promise<JSX.Element> {
  if (!isValidSlug(params.slug)) notFound();
  const meta = getSectionMeta(params.slug);
  const mdx = await readSectionMdx(params.slug);
  if (mdx === null) notFound();

  const idx = SECTIONS.findIndex((s) => s.slug === params.slug);
  const prev = idx > 0 ? SECTIONS[idx - 1] : null;
  const next = idx < SECTIONS.length - 1 ? SECTIONS[idx + 1] : null;

  const session = await getSession();
  const userId = (session?.user as { id?: string } | undefined)?.id ?? null;

  return (
    <article className="mx-auto max-w-3xl px-6 py-10">
      <div className="flex items-baseline justify-between gap-4">
        <div>
          <Link
            href="/learn"
            className="focus-ring rounded font-mono text-xs uppercase tracking-widest text-accent hover:underline"
          >
            ← /learn
          </Link>
          <h1 className="mt-2 text-3xl font-semibold tracking-tight">
            {meta.title}
          </h1>
          <p className="mt-1 text-sm text-neutral-500">{meta.summary}</p>
        </div>
        <LayerToggle />
      </div>

      <div className="prose prose-neutral dark:prose-invert mt-8 max-w-none">
        <MDXRemote source={mdx} components={mdxComponents} />
      </div>

      <nav className="mt-12 flex items-center justify-between border-t border-neutral-200 pt-6 text-sm dark:border-neutral-800">
        {prev ? (
          <Link
            href={`/learn/${prev.slug}`}
            className="focus-ring rounded text-neutral-600 hover:text-accent dark:text-neutral-400"
          >
            ← {prev.title}
          </Link>
        ) : (
          <span aria-hidden />
        )}
        {next ? (
          <Link
            href={`/learn/${next.slug}`}
            className="focus-ring rounded text-neutral-600 hover:text-accent dark:text-neutral-400"
          >
            {next.title} →
          </Link>
        ) : (
          <span aria-hidden />
        )}
      </nav>

      <CommentSection
        sectionSlug={params.slug}
        signedIn={userId !== null}
        currentUserId={userId}
      />
      <ProgressTracker sectionSlug={params.slug} signedIn={userId !== null} />
      <EventTracker sectionSlug={params.slug} />
    </article>
  );
}
