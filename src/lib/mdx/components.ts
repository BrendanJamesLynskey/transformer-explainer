/**
 * MDX components map.
 *
 * Mapped here rather than via JSX in a `.tsx` so server code that needs
 * the map (e.g. the `[slug]` page) doesn't drag in client-only widgets
 * unnecessarily. The animations come through `lazy.tsx`, so each chapter
 * loads only its own animation's code.
 */
import type { MDXRemoteProps } from "next-mdx-remote/rsc";

import { AttentionWidget } from "@/components/interactive/AttentionWidget";
import { EmbeddingWidget } from "@/components/interactive/EmbeddingWidget";
import { FFNWidget } from "@/components/interactive/FFNWidget";
import { Layer } from "@/components/interactive/Layer";
import {
  AttentionAnimation,
  EmbeddingAnimation,
  FFNAnimation,
  GenerationAnimation,
  LayerNormAnimation,
  OverviewHero,
  StackingAnimation,
} from "@/components/interactive/lazy";
import { LayerNormWidget } from "@/components/interactive/LayerNormWidget";
import { SamplingWidget } from "@/components/interactive/SamplingWidget";
import { StackingWidget } from "@/components/interactive/StackingWidget";
import { CheckYourself } from "@/components/learn/CheckYourself";
import { Keep, Remember } from "@/components/learn/Remember";
import { Check, V } from "@/components/mdx/Check";
import { Eq } from "@/components/mdx/Eq";
import { AttentionGrowthPanel } from "@/components/training/AttentionGrowthPanel";
import { BadBatchChart } from "@/components/training/BadBatchChart";
import { BatchSizeChart } from "@/components/training/BatchSizeChart";
import { CoordCheckChart } from "@/components/training/CoordCheckChart";
import {
  GoodputChart,
  MonitorCharts,
  ResumeChart,
} from "@/components/training/JobCharts";
import {
  BackpropAnimation,
  BpeAnimation,
  DemoSyncAnimation,
  DepthAnimation,
  MemoryAnimation,
  ObjectiveAnimation,
  Optim2dAnimation,
  PipelineAnimation,
  RingAllReduceAnimation,
  RoundingAnimation,
  SftAnimation,
  TrainingLoopHero,
} from "@/components/training/lazy";
import { LiveTrainerPanel } from "@/components/training/LiveTrainerPanel";
import { LrSweepChart } from "@/components/training/LrSweepChart";
import { LrTrainerPanel } from "@/components/training/LrTrainerPanel";
import { MupChart } from "@/components/training/MupChart";
import { OptimizerRuns } from "@/components/training/OptimizerRuns";
import { PrecisionChart } from "@/components/training/PrecisionChart";
import { PrePostChart } from "@/components/training/PrePostChart";
import { R } from "@/components/training/R";
import { SchedulePanel } from "@/components/training/SchedulePanel";
import { SlowLinksCharts } from "@/components/training/SlowLinksCharts";
import {
  DataParallelPanel,
  GoodputPanel,
  IsoflopPanel,
  LayoutPanel,
  WanPanel,
} from "@/components/training/SystemsPanels";
import { ValidationChart } from "@/components/training/ValidationChart";

export const mdxComponents: NonNullable<MDXRemoteProps["components"]> = {
  Layer,
  EmbeddingWidget,
  AttentionWidget,
  FFNWidget,
  LayerNormWidget,
  StackingWidget,
  SamplingWidget,
  // Brief 27: model-driven animations (client-side traces) and the
  // server-rendered equations that sit beside them.
  Eq,
  OverviewHero,
  EmbeddingAnimation,
  AttentionAnimation,
  FFNAnimation,
  LayerNormAnimation,
  StackingAnimation,
  GenerationAnimation,
  // Part 2 (Training): animations, the live trainer, recorded results.
  TrainingLoopHero,
  LiveTrainerPanel,
  BpeAnimation,
  ObjectiveAnimation,
  BackpropAnimation,
  Optim2dAnimation,
  OptimizerRuns,
  R,
  // Part 2, chapters 6-10 (brief 29B)
  SchedulePanel,
  LrTrainerPanel,
  LrSweepChart,
  BatchSizeChart,
  DepthAnimation,
  PrePostChart,
  CoordCheckChart,
  MupChart,
  AttentionGrowthPanel,
  BadBatchChart,
  RoundingAnimation,
  PrecisionChart,
  MemoryAnimation,
  // Part 2, chapters 11-16 (brief 29C)
  RingAllReduceAnimation,
  DataParallelPanel,
  PipelineAnimation,
  LayoutPanel,
  ValidationChart,
  DemoSyncAnimation,
  SlowLinksCharts,
  WanPanel,
  IsoflopPanel,
  GoodputPanel,
  MonitorCharts,
  ResumeChart,
  GoodputChart,
  SftAnimation,
  // Brief 44, the learning layer: three self-checks and "What to remember"
  // at the foot of every chapter (src/components/learn/ is shared with the
  // other *-explained sites; Check and V resolve this site's numbers).
  CheckYourself,
  Check,
  V,
  Remember,
  Keep,
};
