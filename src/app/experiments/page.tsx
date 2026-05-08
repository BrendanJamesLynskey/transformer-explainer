/**
 * /experiments — public, paginated list of saved experiments.
 *
 * Server Component reading directly via the data layer (no `/api/*`
 * round-trip) — first paint with no client-side fetching.
 */
import Link from "next/link";

import { listPublic } from "@/lib/experiments";

export const dynamic = "force-dynamic";

export const metadata = {
  title: "Experiments",
  description: "Public experiments saved by other learners.",
};

export default async function ExperimentsIndex(): Promise<JSX.Element> {
  const items = await listPublic(50);

  return (
    <main className="mx-auto max-w-4xl px-6 py-10">
      <p className="font-mono text-xs uppercase tracking-widest text-accent">
        /experiments
      </p>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">
        Public experiments
      </h1>
      <p className="mt-3 text-neutral-600 dark:text-neutral-300">
        What other readers have saved. Click one to view it; sign in to fork
        into your own account.
      </p>

      {items.length === 0 ? (
        <p className="mt-12 text-sm text-neutral-500">
          No public experiments yet. Be the first — head to the{" "}
          <Link href="/playground" className="underline">
            playground
          </Link>
          .
        </p>
      ) : (
        <ul className="mt-10 grid gap-3 sm:grid-cols-2">
          {items.map((e) => (
            <li
              key={e.id}
              className="rounded-lg border border-neutral-200 bg-white p-4 dark:border-neutral-800 dark:bg-neutral-950"
            >
              <Link
                href={`/experiments/${e.slug}`}
                className="focus-ring rounded text-base font-medium text-neutral-900 hover:text-accent dark:text-neutral-100"
              >
                {e.name}
              </Link>
              <p className="mt-1 line-clamp-2 font-mono text-xs text-neutral-500">
                {e.inputText || "(empty input)"}
              </p>
              <p className="mt-2 text-[10px] uppercase tracking-widest text-neutral-400">
                {new Date(e.createdAt).toLocaleDateString()} · {e.viewCount}{" "}
                views
              </p>
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}
