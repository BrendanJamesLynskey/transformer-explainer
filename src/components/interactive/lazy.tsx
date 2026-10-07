"use client";

/**
 * Code-split animations (the companion sites' pattern): each loads its own
 * chunk after the page shell, so lesson pages stay light. Each takes its
 * equation as server-rendered children (`<Eq>`).
 *
 * The placeholder reserves roughly the panel's height (measured at phone,
 * small-tablet and desktop widths), so the text below doesn't jump when the
 * animation arrives (Lighthouse's layout-shift score). The classes are
 * written out in full so Tailwind generates them.
 */
import dynamic from "next/dynamic";

function placeholder(heights: string) {
  return function Placeholder(): JSX.Element {
    return (
      <p
        data-pending-widget
        className={`my-8 text-sm text-neutral-600 dark:text-neutral-400 ${heights}`}
      >
        Loading the animation…
      </p>
    );
  };
}

export const OverviewHero = dynamic(() => import("./OverviewHero"), {
  ssr: false,
  loading: placeholder("min-h-[68rem] sm:min-h-[76rem] md:min-h-[49rem]"),
});
export const EmbeddingAnimation = dynamic(
  () => import("./EmbeddingAnimation"),
  {
    ssr: false,
    loading: placeholder("min-h-[56rem] sm:min-h-[60rem] md:min-h-[52rem]"),
  },
);
export const AttentionAnimation = dynamic(
  () => import("./AttentionAnimation"),
  {
    ssr: false,
    loading: placeholder("min-h-[76rem] sm:min-h-[95rem] md:min-h-[52rem]"),
  },
);
export const FFNAnimation = dynamic(() => import("./FFNAnimation"), {
  ssr: false,
  loading: placeholder("min-h-[65rem] sm:min-h-[76rem] md:min-h-[49rem]"),
});
