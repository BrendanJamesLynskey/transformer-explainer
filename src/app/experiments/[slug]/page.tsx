/**
 * /experiments/[slug] — single-experiment viewer.
 *
 * Read-only for non-owners. Shows the saved input + config and exposes a
 * "Fork" button (signed-in users) and a "Delete" button (owners). The
 * "Open in playground" link round-trips through the URL so the
 * playground widgets pick up the saved input.
 */
import Link from "next/link";
import { notFound } from "next/navigation";

import { EventTracker } from "@/components/interactive/EventTracker";
import { ForkButton } from "@/components/interactive/ForkButton";
import { auth } from "@/lib/auth";
import { getBySlug, incrementViews } from "@/lib/experiments";

export const dynamic = "force-dynamic";

export async function generateMetadata({
  params,
}: {
  params: { slug: string };
}) {
  const exp = await getBySlug(params.slug);
  return exp ? { title: exp.name, description: exp.inputText } : {};
}

export default async function ExperimentPage({
  params,
}: {
  params: { slug: string };
}): Promise<JSX.Element> {
  const exp = await getBySlug(params.slug);
  if (!exp) notFound();

  const session = await auth();
  const userId = (session?.user as { id?: string } | undefined)?.id;

  if (exp.visibility === "private" && userId !== exp.ownerId) notFound();

  // Best-effort: bump the view count for non-owner reads.
  if (userId !== exp.ownerId) await incrementViews(exp.slug);

  const isOwner = userId !== undefined && userId === exp.ownerId;
  const config = exp.configJson as Record<string, number>;

  return (
    <main className="mx-auto max-w-3xl px-6 py-10">
      <Link
        href="/experiments"
        className="focus-ring rounded font-mono text-xs uppercase tracking-widest text-accent hover:underline"
      >
        ← /experiments
      </Link>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">{exp.name}</h1>
      <p className="mt-1 text-sm text-neutral-500">
        {exp.visibility} · created{" "}
        {new Date(exp.createdAt).toLocaleDateString()} · {exp.viewCount} views
      </p>

      <section className="mt-8 rounded-lg border border-neutral-200 bg-neutral-50 p-4 dark:border-neutral-800 dark:bg-neutral-900">
        <h2 className="text-sm font-semibold text-neutral-800 dark:text-neutral-200">
          Input
        </h2>
        <pre className="mt-2 overflow-x-auto font-mono text-sm">
          {JSON.stringify(exp.inputText) || '""'}
        </pre>
      </section>

      <section className="mt-4 rounded-lg border border-neutral-200 bg-neutral-50 p-4 dark:border-neutral-800 dark:bg-neutral-900">
        <h2 className="text-sm font-semibold text-neutral-800 dark:text-neutral-200">
          Configuration
        </h2>
        <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 font-mono text-xs sm:grid-cols-3">
          {Object.entries(config).map(([k, v]) => (
            <div key={k} className="flex justify-between gap-2">
              <dt className="text-neutral-500">{k}</dt>
              <dd>{String(v)}</dd>
            </div>
          ))}
        </dl>
      </section>

      <div className="mt-8 flex flex-wrap items-center gap-3">
        <Link
          href="/playground"
          className="focus-ring rounded bg-accent px-4 py-1.5 text-sm font-medium text-accent-fg hover:opacity-90"
        >
          Open in playground
        </Link>
        {!isOwner && session?.user && exp.visibility !== "private" && (
          <ForkButton slug={exp.slug} />
        )}
        {isOwner && (
          <span className="font-mono text-xs text-neutral-500">
            You own this experiment.
          </span>
        )}
      </div>
      <EventTracker
        pageKind="exp_view"
        meta={{ slug: exp.slug }}
        sectionSlug={null}
      />
    </main>
  );
}
