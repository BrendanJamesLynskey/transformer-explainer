/**
 * Sitemap contents for each of the site's two addresses (see ./hosts.ts).
 * Pure, so it is unit-tested directly.
 */
import { DECODER_ORIGIN, TRAINING_ORIGIN, TRAINING_HOST } from "./hosts";

/** Is this request for the training host (production or local stand-in)? */
function isTrainingHost(host: string | null | undefined): boolean {
  const h = (host ?? "").trim().toLowerCase();
  return h === TRAINING_HOST || /^training\.localhost(:\d+)?$/.test(h);
}

/** The canonical origin whose pages this host's sitemap lists. */
export function sitemapOrigin(host: string | null | undefined): string {
  return isTrainingHost(host) ? TRAINING_ORIGIN : DECODER_ORIGIN;
}

/**
 * Every page this host's sitemap lists, as absolute canonical URLs: Part 2
 * on the training host, Part 1 (and the site's other pages) elsewhere.
 */
export function sitemapUrls(
  host: string | null | undefined,
  learnSlugs: readonly string[],
  trainingSlugs: readonly string[],
): string[] {
  if (isTrainingHost(host)) {
    return [
      `${TRAINING_ORIGIN}/training`,
      ...trainingSlugs.map((s) => `${TRAINING_ORIGIN}/training/${s}`),
    ];
  }
  return [
    `${DECODER_ORIGIN}/`,
    `${DECODER_ORIGIN}/foundations`,
    `${DECODER_ORIGIN}/learn`,
    ...learnSlugs.map((s) => `${DECODER_ORIGIN}/learn/${s}`),
    `${DECODER_ORIGIN}/playground`,
    `${DECODER_ORIGIN}/experiments`,
    `${DECODER_ORIGIN}/about`,
  ];
}
