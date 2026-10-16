/**
 * How this site presents itself when a link to it is shared (brief 39): its name, track, one-line
 * claim and how it is checked, as drawn on the link-preview card (src/app/_og/card.tsx), and the
 * Open Graph and Twitter tags for each page. Every page passes its own path to `withShare`, so
 * a shared chapter link names that chapter, not the home page.
 */
import type { Metadata } from "next";

import { DECODER_ORIGIN as SITE_URL, TRAINING_ORIGIN } from "@/lib/site/hosts";

export type Share = {
  name: string;
  /** The track in the site menu (null: the site sits outside the four tracks). */
  track: string | null;
  /** The card's top line, when it is not "<TRACK> TRACK". */
  kicker?: string;
  /** The one-line claim under the name on the card. */
  claim: string;
  /** How the site is checked, in one line (from the About page). */
  checkedShort: string;
  /** The site's colour on its card and favicon. */
  color: string;
  url: string;
  description: string;
  /** The site's card (the opengraph-image route); a page whose segment has its own card overrides it. */
  image?: string;
};

export const SHARE: Share = {
  name: "Transformer Decoder Explained",
  track: "Foundations",
  claim: "How a decoder-only transformer runs, one forward pass",
  checkedShort:
    "The in-browser transformer is checked against PyTorch, operation by operation, to within 1e-5",
  color: "#0891b2",
  url: SITE_URL,
  description:
    "How a decoder-only transformer runs, and how it is trained: a graphical, interactive explainer. Watch every operation execute, then train the same model in your browser.",
};

/** The Training site, the same deployment at its own address (src/lib/site/hosts.ts). */
export const TRAINING_SHARE: Share = {
  name: "Transformer Training Explained",
  track: "Foundations",
  claim: "How the same decoder is trained, from one step to many GPUs",
  checkedShort:
    "Its NumPy reference matches PyTorch autograd to 1e-12; the in-browser trainer matches it step for step",
  color: "#c2410c",
  url: TRAINING_ORIGIN,
  image: "/training/opengraph-image",
  description:
    "How a decoder is trained, from the loss and backprop to optimisers and many GPUs, with the Decoder site's model training in your browser.",
};

/**
 * `m` plus Open Graph and Twitter tags for the page at `path` (titled as `m` titles it). A page
 * whose segment draws its own card (a chapter) passes `ownCard`, so that card is used instead of
 * the site's.
 */
export function withShare<M extends Metadata>(
  path: string,
  m: M,
  share: Share = SHARE,
  ownCard = false,
): M & Pick<Metadata, "openGraph" | "twitter"> {
  const t = m.title;
  const own =
    typeof t === "string"
      ? t
      : t && typeof t === "object" && "absolute" in t
        ? t.absolute
        : null;
  const title =
    own && !own.endsWith(share.name)
      ? `${own} · ${share.name}`
      : (own ?? share.name);
  const description =
    typeof m.description === "string" ? m.description : share.description;
  // a page's own openGraph replaces its parent's, card included, so name the site's card here
  const card = share.image ?? "/opengraph-image";
  const alt = `${share.name}: ${share.claim}`;
  return {
    ...m,
    openGraph: {
      title,
      description,
      url: path,
      siteName: share.name,
      type: path === "/" ? "website" : "article",
      locale: "en_GB",
      ...(ownCard
        ? {}
        : {
            images: [
              { url: card, width: 1200, height: 630, alt, type: "image/png" },
            ],
          }),
    },
    twitter: {
      card: "summary_large_image",
      title,
      description,
      ...(ownCard
        ? {}
        : {
            images: [
              { url: card.replace(/opengraph-image$/, "twitter-image"), alt },
            ],
          }),
    },
  };
}
