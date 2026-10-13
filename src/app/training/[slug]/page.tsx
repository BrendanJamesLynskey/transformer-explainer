/**
 * /training/[slug] — a Part 2 (Training) chapter, rendered by the same
 * `ChapterArticle` as Part 1's chapters. Chapters listed in
 * `TRAINING_SECTIONS` whose MDX has not shipped yet answer 404.
 */
import { notFound } from "next/navigation";

import { ChapterArticle } from "@/components/mdx/ChapterArticle";
import {
  TRAINING_DECKS,
  TRAINING_SECTIONS,
  getTrainingMeta,
  isTrainingSlug,
  readTrainingMdx,
  trainingDeckUrl,
  trainingSectionKey,
} from "@/lib/mdx/sections";
import { getSiteContext } from "@/lib/site/context";
import { siteHref, trainingCanonical } from "@/lib/site/hosts";

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
    // Part 2's address is the training host, wherever the page is served
    alternates: { canonical: trainingCanonical(`/training/${params.slug}`) },
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
  const ctx = getSiteContext();
  const next = await readyNext();
  const link = (s: (typeof TRAINING_SECTIONS)[number] | null) =>
    s ? { href: `/training/${s.slug}`, title: s.title } : null;

  return (
    <ChapterArticle
      backHref="/training"
      backLabel="← /training"
      partLabel={`Chapter ${idx + 1} of ${TRAINING_SECTIONS.length}`}
      title={meta.title}
      summary={meta.summary}
      mdx={mdx}
      // the first chapter steps back to the other Foundations site
      prev={
        link(prev) ?? {
          href: siteHref("/learn", ctx),
          title: "The decoder",
          crossSite: true,
        }
      }
      next={link(next)}
      sectionKey={trainingSectionKey(params.slug)}
      deck={{
        num: TRAINING_DECKS[params.slug].num,
        // the deck's short title (before any subtitle)
        title: TRAINING_DECKS[params.slug].title.split(": ")[0]!,
        href: trainingDeckUrl(params.slug),
      }}
    />
  );
}
