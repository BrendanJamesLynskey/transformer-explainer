"use client";

/**
 * Code-split Part 2 widgets: each loads its own chunk after the page shell
 * (as Part 1's `interactive/lazy.tsx`), with a placeholder that reserves
 * roughly the panel's height so the text below doesn't jump.
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

export const TrainingLoopHero = dynamic(() => import("./TrainingLoopHero"), {
  ssr: false,
  loading: placeholder("min-h-[56rem] md:min-h-[44rem]"),
});
export const LiveTrainer = dynamic(() => import("./LiveTrainer"), {
  ssr: false,
  loading: placeholder("min-h-[34rem]"),
});
export const BpeAnimation = dynamic(() => import("./BpeAnimation"), {
  ssr: false,
  loading: placeholder("min-h-[46rem] md:min-h-[40rem]"),
});
export const ObjectiveAnimation = dynamic(
  () => import("./ObjectiveAnimation"),
  {
    ssr: false,
    loading: placeholder("min-h-[50rem] md:min-h-[44rem]"),
  },
);
export const BackpropAnimation = dynamic(() => import("./BackpropAnimation"), {
  ssr: false,
  loading: placeholder("min-h-[52rem] md:min-h-[46rem]"),
});
export const Optim2dAnimation = dynamic(() => import("./Optim2dAnimation"), {
  ssr: false,
  loading: placeholder("min-h-[44rem] md:min-h-[46rem]"),
});
// Brief 29B (chapters 6-10)
export const ScheduleAnimation = dynamic(() => import("./ScheduleAnimation"), {
  ssr: false,
  loading: placeholder("min-h-[44rem] md:min-h-[40rem]"),
});
export const DepthAnimation = dynamic(() => import("./DepthAnimation"), {
  ssr: false,
  loading: placeholder("min-h-[56rem] md:min-h-[40rem]"),
});
export const AttentionGrowthAnimation = dynamic(
  () => import("./AttentionGrowthAnimation"),
  {
    ssr: false,
    loading: placeholder("min-h-[56rem] md:min-h-[42rem]"),
  },
);
export const RoundingAnimation = dynamic(() => import("./RoundingAnimation"), {
  ssr: false,
  loading: placeholder("min-h-[40rem] md:min-h-[38rem]"),
});
export const MemoryAnimation = dynamic(() => import("./MemoryAnimation"), {
  ssr: false,
  loading: placeholder("min-h-[56rem] md:min-h-[46rem]"),
});
