/**
 * The family's visual language (explained_sites_visual_standard.md §3):
 * Okabe and Ito's colour-blind-safe palette, the same in light and dark
 * mode, shared with the other *-explained sites.
 *
 * This site gives each part of the decoder one colour, used in every
 * animation and in the equations beside them (globals.css, `.hl-*`):
 * token ids and the embedding table blue, positions orange, attention
 * vermillion, the FFN green, LayerNorm sky blue, logits and sampling
 * purple. Colours are fills and strokes, never text on white. Values in
 * heat maps are diverging: positive blue, negative orange, the strength
 * of the colour the size, and the sign is also printed in the captions
 * (never colour alone). "Active" is a highlight, "done" is muted; masked
 * (−∞) scores are grey plus a hatch pattern.
 *
 * Okabe, M. and Ito, K. (2008), "Color Universal Design (CUD): how to make
 * figures and presentations that are friendly to colorblind people",
 * https://jfly.uni-koeln.de/color/
 */

export const OKABE_ITO = {
  black: "#000000",
  orange: "#E69F00",
  sky: "#56B4E9",
  green: "#009E73",
  yellow: "#F0E442",
  blue: "#0072B2",
  vermillion: "#D55E00",
  purple: "#CC79A7",
} as const;

/** One colour per memory level, the same on every site in the family. */
export const LEVEL_COLOUR = {
  reg: OKABE_ITO.orange,
  smem: OKABE_ITO.green,
  l2: OKABE_ITO.sky,
  hbm: OKABE_ITO.purple,
} as const;

/** One colour per part of the decoder. */
export const PART_COLOUR = {
  tok: OKABE_ITO.blue,
  emb: OKABE_ITO.blue,
  pos: OKABE_ITO.orange,
  attn: OKABE_ITO.vermillion,
  ffn: OKABE_ITO.green,
  ln: OKABE_ITO.sky,
  logits: OKABE_ITO.purple,
  sample: OKABE_ITO.purple,
} as const;
export type Part = keyof typeof PART_COLOUR;

/** States of an element in an animation. */
export const STATE_COLOUR = {
  active: OKABE_ITO.blue,
  error: OKABE_ITO.vermillion,
} as const;

/** Muted ("done", "idle") greys: Tailwind neutral-400 and neutral-600. */
export const MUTED = { light: "#a3a3a3", dark: "#525252" } as const;

/**
 * A diverging heat-map fill for `v` on a scale of ±`max`: positive values
 * blue, negative orange, opacity growing with |v| (a floor keeps tiny
 * values visible against the background in both modes).
 */
export function heat(
  v: number,
  max: number,
): { fill: string; fillOpacity: number } {
  const t = max > 0 ? Math.min(1, Math.abs(v) / max) : 0;
  return {
    fill: v >= 0 ? OKABE_ITO.blue : OKABE_ITO.orange,
    fillOpacity: 0.08 + 0.85 * t,
  };
}
