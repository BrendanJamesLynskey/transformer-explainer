/**
 * MDX components map.
 *
 * Mapped here rather than via JSX in a `.tsx` so server code that needs
 * the map (e.g. the `[slug]` page) doesn't drag in client-only widgets
 * unnecessarily.
 */
import type { MDXRemoteProps } from "next-mdx-remote/rsc";

import { AttentionWidget } from "@/components/interactive/AttentionWidget";
import { EmbeddingWidget } from "@/components/interactive/EmbeddingWidget";
import { Layer } from "@/components/interactive/Layer";

export const mdxComponents: NonNullable<MDXRemoteProps["components"]> = {
  Layer,
  EmbeddingWidget,
  AttentionWidget,
};
