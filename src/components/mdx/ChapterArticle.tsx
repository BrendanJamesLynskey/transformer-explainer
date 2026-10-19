/**
 * One chapter page, shared by Part 1 (`/learn/[slug]`) and Part 2
 * (`/training/[slug]`): the title, the three-layer toggle, the MDX, the
 * previous/next links, the comments, progress and analytics.
 *
 * Links inside the MDX to the other part (`/learn/…` from a Training
 * chapter) go to the other part's host (`src/lib/site/hosts.ts`).
 *
 * Server Component. Maths: `remark-math` turns `$…$` / `$$…$$` into maths
 * nodes and `rehype-katex` renders them to HTML on the server, so the Maths
 * layer arrives as finished markup and no KaTeX JavaScript is shipped (the
 * compile options are in `src/lib/mdx/options.ts`). The text is styled by
 * the `.mdx-content` rules in `globals.css`.
 */
import Link from "next/link";
import type { ComponentPropsWithoutRef } from "react";
import { MDXRemote } from "next-mdx-remote/rsc";

import { CommentSection } from "@/components/interactive/CommentSection";
import { CHECK_YOURSELF_ID } from "@/components/learn/CheckYourself";
import { EventTracker } from "@/components/interactive/EventTracker";
import { LayerToggle } from "@/components/interactive/LayerToggle";
import { ProgressTracker } from "@/components/interactive/ProgressTracker";
import { getSession } from "@/lib/auth";
import { runOrFallback } from "@/lib/db-fallback";
import { mdxComponents } from "@/lib/mdx/components";
import { chapterMdxOptions } from "@/lib/mdx/options";
import { getSiteContext } from "@/lib/site/context";
import { siteHref } from "@/lib/site/hosts";

/** `crossSite`: the other Foundations site, reached with a full page load
 * (the root layout's header must re-render to name it). */
type NavLink = { href: string; title: string; crossSite?: boolean } | null;

/** The chapter's slide deck (Training chapters). */
type Deck = { num: string; title: string; href: string };

export async function ChapterArticle({
  backHref,
  backLabel,
  partLabel,
  title,
  summary,
  mdx,
  prev,
  next,
  sectionKey,
  deck,
}: {
  backHref: string;
  backLabel: string;
  /** e.g. "Chapter 1 of 16" (Training); omitted on Decoder pages */
  partLabel?: string;
  title: string;
  summary: string;
  mdx: string;
  prev: NavLink;
  next: NavLink;
  /** the key comments, progress and analytics are stored under */
  sectionKey: string;
  /** the chapter's slide deck, linked under the summary */
  deck?: Deck;
}): Promise<JSX.Element> {
  const session = await runOrFallback("learn:auth", getSession, null);
  const userId = (session?.user as { id?: string } | undefined)?.id ?? null;
  const ctx = getSiteContext();
  const components = {
    ...mdxComponents,
    a: (props: ComponentPropsWithoutRef<"a">) => (
      <a {...props} href={props.href ? siteHref(props.href, ctx) : undefined} />
    ),
  };

  return (
    <article className="mx-auto max-w-3xl px-6 py-10">
      <div className="flex flex-wrap items-baseline justify-between gap-4">
        <div>
          <Link
            href={backHref}
            className="focus-ring rounded font-mono text-xs uppercase tracking-widest text-accent hover:underline dark:text-indigo-300"
          >
            {backLabel}
          </Link>
          {partLabel && (
            <p className="mt-2 text-xs uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
              {partLabel}
            </p>
          )}
          <h1 className="mt-2 text-3xl font-semibold tracking-tight">
            {title}
          </h1>
          <p className="mt-1 text-sm text-neutral-500 dark:text-neutral-400">
            {summary}
          </p>
          {mdx.includes("<CheckYourself") && (
            <p className="mt-2 text-sm">
              <a
                href={`#${CHECK_YOURSELF_ID}`}
                data-check-yourself-link
                className="focus-ring rounded text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent dark:text-indigo-300"
              >
                Check yourself ↓
              </a>
              <span className="text-neutral-500 dark:text-neutral-400">
                {" "}
                three questions and what to remember, at the end
              </span>
            </p>
          )}
        </div>
        <LayerToggle />
      </div>
      {deck && (
        <p className="mt-3 text-sm text-neutral-500 dark:text-neutral-400">
          Slides for this chapter:{" "}
          <a
            href={deck.href}
            data-chapter-deck
            className="focus-ring rounded text-accent underline underline-offset-2 dark:text-indigo-300"
          >
            Transformer Training {deck.num}, {deck.title}
          </a>
        </p>
      )}

      <div className="mdx-content mt-8">
        <MDXRemote
          source={mdx}
          components={components}
          options={chapterMdxOptions}
        />
      </div>

      <nav className="mt-12 flex items-center justify-between border-t border-neutral-200 pt-6 text-sm dark:border-neutral-800">
        {prev ? (
          prev.crossSite ? (
            <a
              href={prev.href}
              className="focus-ring rounded text-neutral-600 hover:text-accent dark:text-neutral-400"
            >
              ← {prev.title}
            </a>
          ) : (
            <Link
              href={prev.href}
              className="focus-ring rounded text-neutral-600 hover:text-accent dark:text-neutral-400"
            >
              ← {prev.title}
            </Link>
          )
        ) : (
          <span aria-hidden />
        )}
        {next ? (
          next.crossSite ? (
            <a
              href={next.href}
              className="focus-ring rounded text-neutral-600 hover:text-accent dark:text-neutral-400"
            >
              {next.title} →
            </a>
          ) : (
            <Link
              href={next.href}
              className="focus-ring rounded text-neutral-600 hover:text-accent dark:text-neutral-400"
            >
              {next.title} →
            </Link>
          )
        ) : (
          <span aria-hidden />
        )}
      </nav>

      <CommentSection
        sectionSlug={sectionKey}
        signedIn={userId !== null}
        currentUserId={userId}
      />
      <ProgressTracker sectionSlug={sectionKey} signedIn={userId !== null} />
      <EventTracker sectionSlug={sectionKey} />
    </article>
  );
}
