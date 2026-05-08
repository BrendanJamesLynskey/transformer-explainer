/**
 * Root layout for the App Router.
 *
 * Server Component — see CLAUDE.md §5 ("Server Components by default").
 * Phase 0 keeps this minimal: HTML scaffold, font imports via Tailwind, and
 * the global stylesheet. Phase 3 will add the three-layer toggle and theme
 * provider; Phase 2 adds the auth-aware header.
 */
import type { Metadata } from "next";
import type { ReactNode } from "react";

import { SiteHeader } from "@/components/ui/SiteHeader";
import { env } from "@/lib/env";

import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL(env.NEXT_PUBLIC_SITE_URL),
  title: {
    default: "Transformer Explainer",
    template: "%s · Transformer Explainer",
  },
  description:
    "A graphical, interactive explainer of the Transformer decoder. Type in tokens and watch every operation execute step-by-step.",
};

export default function RootLayout({
  children,
}: {
  children: ReactNode;
}): JSX.Element {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen font-sans antialiased">
        <SiteHeader />
        {children}
      </body>
    </html>
  );
}
