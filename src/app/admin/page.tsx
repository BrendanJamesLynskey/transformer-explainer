/**
 * /admin — analytics dashboard.
 *
 * Server component. Reads aggregates straight from `lib/analytics`.
 * Non-admins get a 404 (we don't want to leak that the route exists).
 */
import Link from "next/link";
import { notFound } from "next/navigation";

import { HealthCard } from "@/components/admin/HealthCard";
import { Sparkline } from "@/components/admin/Sparkline";
import { SectionFunnelTable } from "@/components/admin/SectionFunnelTable";
import {
  dailyActive,
  eventCounts,
  recentComments,
  sectionFunnel,
  topExperiments,
} from "@/lib/analytics";
import { auth, isAdmin } from "@/lib/auth";
import { fallbackStats } from "@/lib/db-fallback";
import { checkHealth } from "@/lib/health";

export const dynamic = "force-dynamic";

export const metadata = {
  title: "Admin",
  robots: { index: false, follow: false },
};

export default async function AdminPage(): Promise<JSX.Element> {
  const session = await auth();
  const login = (session?.user as { githubLogin?: string | null } | undefined)
    ?.githubLogin;
  if (!isAdmin(login)) notFound();

  const [health, dau, funnel, top, comments, kinds] = await Promise.all([
    checkHealth(),
    dailyActive(60),
    sectionFunnel(),
    topExperiments(10),
    recentComments(10),
    eventCounts(30),
  ]);

  return (
    <main className="mx-auto max-w-4xl space-y-12 px-6 py-12">
      <header>
        <p className="font-mono text-xs uppercase tracking-widest text-accent dark:text-indigo-300">
          /admin
        </p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight">
          Dashboard
        </h1>
        <p className="mt-2 text-sm text-neutral-600 dark:text-neutral-400">
          Aggregates over all events. Live data — no caching.
        </p>
      </header>

      <section data-testid="health-card" className="space-y-3">
        <h2 className="text-lg font-medium">Database health</h2>
        <HealthCard report={health} fallbacks={fallbackStats()} />
      </section>

      <section data-testid="dau-card" className="space-y-3">
        <h2 className="text-lg font-medium">Daily active sessions</h2>
        <Sparkline points={dau} />
      </section>

      <section data-testid="funnel-card" className="space-y-3">
        <h2 className="text-lg font-medium">Section funnel</h2>
        <p className="text-xs text-neutral-500 dark:text-neutral-400">
          Views → Interacted → Completed. Anonymous-inclusive.
        </p>
        <SectionFunnelTable rows={funnel} />
      </section>

      <section data-testid="kinds-card" className="space-y-3">
        <h2 className="text-lg font-medium">Events by kind (30 days)</h2>
        {kinds.length === 0 ? (
          <p className="font-mono text-xs text-neutral-500 dark:text-neutral-400">
            no events yet
          </p>
        ) : (
          <ul className="grid grid-cols-1 gap-x-8 gap-y-1 text-sm sm:grid-cols-2">
            {kinds.map((k) => (
              <li key={k.kind} className="flex justify-between gap-4">
                <span className="font-mono text-xs">{k.kind}</span>
                <span className="tabular-nums">{k.count}</span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section data-testid="top-card" className="space-y-3">
        <h2 className="text-lg font-medium">Top experiments</h2>
        {top.length === 0 ? (
          <p className="font-mono text-xs text-neutral-500 dark:text-neutral-400">
            no experiments
          </p>
        ) : (
          <ol className="space-y-1 text-sm">
            {top.map((e) => (
              <li key={e.slug} className="flex items-baseline gap-3">
                <Link
                  href={`/experiments/${e.slug}`}
                  className="focus-ring rounded text-neutral-900 hover:text-accent dark:text-neutral-100"
                >
                  {e.name}
                </Link>
                <span className="font-mono text-xs text-neutral-500 dark:text-neutral-400">
                  {e.slug}
                </span>
                <span className="ml-auto tabular-nums text-neutral-600 dark:text-neutral-400">
                  {e.views} view{e.views === 1 ? "" : "s"}
                </span>
              </li>
            ))}
          </ol>
        )}
      </section>

      <section data-testid="comments-card" className="space-y-3">
        <h2 className="text-lg font-medium">Recent comments</h2>
        {comments.length === 0 ? (
          <p className="font-mono text-xs text-neutral-500 dark:text-neutral-400">
            no comments yet
          </p>
        ) : (
          <ul className="space-y-3 text-sm">
            {comments.map((c) => (
              <li
                key={c.id}
                className="rounded border border-neutral-200 bg-white p-3 dark:border-neutral-800 dark:bg-neutral-950"
              >
                <p className="font-mono text-[11px] uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
                  @{c.authorLogin ?? "unknown"} · {c.sectionSlug} ·{" "}
                  {c.createdAt.toISOString().slice(0, 16).replace("T", " ")}Z
                </p>
                <p className="mt-1 line-clamp-3 text-neutral-800 dark:text-neutral-200">
                  {c.body}
                </p>
              </li>
            ))}
          </ul>
        )}
      </section>
    </main>
  );
}
