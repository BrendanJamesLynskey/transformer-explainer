/**
 * Each Training chapter's link-preview card: the chapter's number and title over the Training
 * site's card (see ../../_og/card.tsx), one per chapter at build time.
 */
import { TRAINING_SECTIONS } from "@/lib/mdx/sections";
import { TRAINING_SHARE } from "@/lib/share";

import { card, OG_SIZE, OG_TYPE } from "../../_og/card";
import { TrainingGlyph, TrainingMotif } from "../../_og/motif";

export const alt = "A chapter of Transformer Training Explained";
export const size = OG_SIZE;
export const contentType = OG_TYPE;

export function generateStaticParams(): { slug: string }[] {
  return TRAINING_SECTIONS.map((s) => ({ slug: s.slug }));
}

export default function Image({ params }: { params: { slug: string } }) {
  const i = Math.max(
    TRAINING_SECTIONS.findIndex((s) => s.slug === params.slug),
    0,
  );
  const s = TRAINING_SECTIONS[i]!;
  const n = Number.parseInt(s.slug, 10);
  return card(
    { number: Number.isNaN(n) ? i + 1 : n, title: s.title },
    TRAINING_SHARE,
    { glyph: <TrainingGlyph size={52} />, motif: <TrainingMotif /> },
  );
}
