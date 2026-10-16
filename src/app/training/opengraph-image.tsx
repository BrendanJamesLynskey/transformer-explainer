/**
 * The Training site's link-preview card (its own address, the same deployment): see
 * ../_og/card.tsx, drawn with Training's name, claim, glyph and loss curve.
 */
import { TRAINING_SHARE } from "@/lib/share";

import { card, OG_SIZE, OG_TYPE } from "../_og/card";
import { TrainingGlyph, TrainingMotif } from "../_og/motif";

export const alt = `${TRAINING_SHARE.name}: ${TRAINING_SHARE.claim}`;
export const size = OG_SIZE;
export const contentType = OG_TYPE;

export default function Image() {
  return card(undefined, TRAINING_SHARE, {
    glyph: <TrainingGlyph size={52} />,
    motif: <TrainingMotif />,
  });
}
