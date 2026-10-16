/**
 * The Training site's segment: its pages' link-preview tags and cards resolve against the
 * training host (src/lib/site/hosts.ts), the address the Training site is shared at. Renders its
 * pages unchanged.
 */
import type { Metadata } from "next";
import type { ReactNode } from "react";

import { TRAINING_ORIGIN } from "@/lib/site/hosts";

export const metadata: Metadata = {
  metadataBase: new URL(TRAINING_ORIGIN),
};

export default function TrainingLayout({
  children,
}: {
  children: ReactNode;
}): ReactNode {
  return children;
}
