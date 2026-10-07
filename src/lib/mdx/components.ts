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
import { Eq } from "@/components/mdx/Eq";

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
};
