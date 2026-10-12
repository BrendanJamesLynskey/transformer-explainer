/**
 * /training/[slug] — a Part 2 (Training) chapter, rendered by the same
 * `ChapterArticle` as Part 1's chapters. Chapters listed in
 * `TRAINING_SECTIONS` whose MDX has not shipped yet answer 404.
 */
import { notFound } from "next/navigation";

import { ChapterArticle } from "@/components/mdx/ChapterArticle";
import {
  TRAINING_SECTIONS,
  getTrainingMeta,
  isTrainingSlug,
  readTrainingMdx,
  trainingSectionKey,
} from "@/lib/mdx/sections";

import "katex/dist/katex.min.css";

export async function generateStaticParams(): Promise<{ slug: string }[]> {
  const ready: string[] = [];
  for (const s of TRAINING_SECTIONS) {
    if ((await readTrainingMdx(s.slug)) !== null) ready.push(s.slug);
  }
  return ready.map((slug) => ({ slug }));
}

export async function generateMetadata({
  params,
}: {
  params: { slug: string };
}) {
  if (!isTrainingSlug(params.slug)) return {};
  const meta = getTrainingMeta(params.slug);
  return {
    title: `${meta.title} · Training`,
    description: meta.summary,
  };
}

export default async function TrainingChapterPage({
  params,
}: {
  params: { slug: string };
}): Promise<JSX.Element> {
  if (!isTrainingSlug(params.slug)) notFound();
  const meta = getTrainingMeta(params.slug);
  const mdx = await readTrainingMdx(params.slug);
  if (mdx === null) notFound();

  const idx = TRAINING_SECTIONS.findIndex((s) => s.slug === params.slug);
  const readyNext = async () => {
    const n = TRAINING_SECTIONS[idx + 1];
    return n && (await readTrainingMdx(n.slug)) ? n : null;
  };
  const prev = idx > 0 ? TRAINING_SECTIONS[idx - 1]! : null;
  const next = await readyNext();
  const link = (s: (typeof TRAINING_SECTIONS)[number] | null) =>
    s ? { href: `/training/${s.slug}`, title: s.title } : null;

  return (
    <ChapterArticle
      backHref="/training"
      backLabel="← /training"
      partLabel={`Part 2 · Training · chapter ${idx + 1} of ${TRAINING_SECTIONS.length}`}
      title={meta.title}
      summary={meta.summary}
      mdx={mdx}
      prev={link(prev)}
      next={link(next)}
      sectionKey={trainingSectionKey(params.slug)}
    />
  );
}
