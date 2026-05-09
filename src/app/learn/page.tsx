/**
 * /learn — index of explainer sections.
 *
 * Server Component. Reads the section catalogue (single source of truth in
 * `src/lib/mdx/sections.ts`) and lists each one with its title and summary.
 */
import Link from "next/link";

import { getSession } from "@/lib/auth";
import { runOrFallback } from "@/lib/db-fallback";
import { SECTIONS, readSectionMdx } from "@/lib/mdx/sections";
import { listForUser } from "@/lib/progress";

// Reads the current session + the user's progress on every request.
export const dynamic = "force-dynamic";

export const metadata = {
  title: "Learn",
  description:
    "Step-by-step walk-through of the Transformer decoder, one operation at a time.",
};

export default async function LearnIndex(): Promise<JSX.Element> {
  // Pre-flight: which sections actually have an MDX file shipped today.
  const status = await Promise.all(
    SECTIONS.map(async (s) => ({
      ...s,
      ready: (await readSectionMdx(s.slug)) !== null,
    })),
  );

  // Tolerate a missing/unreachable DB so the index renders on a fresh
  // clone with placeholder env values.
  const session = await runOrFallback("learn-index:auth", getSession, null);
  const userId = (session?.user as { id?: string } | undefined)?.id ?? null;
  const progressRows = userId
    ? await runOrFallback("learn-index:progress", () => listForUser(userId), [])
    : [];
  const progressMap = new Map(
    progressRows.map((p) => [p.sectionSlug, p.status]),
  );

  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <p className="font-mono text-xs uppercase tracking-widest text-accent">
        /learn
      </p>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">
        The decoder, step by step
      </h1>
      <p className="mt-4 text-neutral-600 dark:text-neutral-300">
        Read in order — each section builds on the one before it. Toggle layers
        (Concept / Maths / Code) inside any section to control how deep the
        explanation goes.
      </p>

      <ol className="mt-10 divide-y divide-neutral-200 dark:divide-neutral-800">
        {status.map((s, i) => (
          <li key={s.slug} className="py-5">
            <div className="flex items-baseline gap-3">
              <span className="font-mono text-xs text-neutral-500">
                {String(i + 1).padStart(2, "0")}
              </span>
              {s.ready ? (
                <Link
                  href={`/learn/${s.slug}`}
                  className="focus-ring rounded text-lg font-medium text-neutral-900 hover:text-accent dark:text-neutral-100"
                >
                  {s.title}
                </Link>
              ) : (
                <span
                  aria-disabled
                  className="text-lg font-medium text-neutral-400 dark:text-neutral-600"
                >
                  {s.title}{" "}
                  <span className="ml-1 rounded bg-neutral-200 px-1.5 py-0.5 text-[10px] font-normal uppercase tracking-wide text-neutral-600 dark:bg-neutral-800 dark:text-neutral-400">
                    coming soon
                  </span>
                </span>
              )}
              {(() => {
                const pStatus = progressMap.get(s.slug);
                if (!pStatus || pStatus === "not_started") return null;
                const label =
                  pStatus === "completed" ? "✓ done" : "in progress";
                const cls =
                  pStatus === "completed"
                    ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-200"
                    : "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-200";
                return (
                  <span
                    className={`ml-2 rounded px-1.5 py-0.5 text-[10px] font-normal uppercase tracking-wide ${cls}`}
                  >
                    {label}
                  </span>
                );
              })()}
            </div>
            <p className="mt-1 pl-9 text-sm text-neutral-600 dark:text-neutral-400">
              {s.summary}
            </p>
          </li>
        ))}
      </ol>
    </main>
  );
}
