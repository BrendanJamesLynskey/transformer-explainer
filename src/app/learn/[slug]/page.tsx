/**
 * /learn/[slug] — a Part 1 (Inference) chapter: an MDX section with the
 * three-layer toggle pinned at the top.
 *
 * Server Component. Validates the slug against the catalogue, reads the
 * MDX from disk and renders it with the shared `ChapterArticle` (also used
 * by /training/[slug]). Only the KaTeX stylesheet (and its fonts) is
 * loaded, and only on chapter pages.
 */
import { notFound } from "next/navigation";

import { ChapterArticle } from "@/components/mdx/ChapterArticle";
import {
  SECTIONS,
  getSectionMeta,
  isValidSlug,
  readSectionMdx,
} from "@/lib/mdx/sections";

import "katex/dist/katex.min.css";

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
  const link = (s: (typeof SECTIONS)[number] | null | undefined) =>
    s ? { href: `/learn/${s.slug}`, title: s.title } : null;

  return (
    <ChapterArticle
      backHref="/learn"
      backLabel="← /learn"
      title={meta.title}
      summary={meta.summary}
      mdx={mdx}
      prev={link(prev)}
      next={link(next)}
      sectionKey={params.slug}
    />
  );
}
