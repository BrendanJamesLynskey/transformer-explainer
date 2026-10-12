/**
 * /training — Part 2's index: how the decoder of Part 1 is trained.
 *
 * Server Component. Lists the Training chapters (TRAINING_SECTIONS) with the
 * reader's progress, like /learn; chapters not shipped yet say so.
 */
import Link from "next/link";

import results from "@/data/training/results.json";
import { getSession } from "@/lib/auth";
import { runOrFallback } from "@/lib/db-fallback";
import {
  TRAINING_DECK_HUB,
  TRAINING_SECTIONS,
  readTrainingMdx,
  trainingSectionKey,
} from "@/lib/mdx/sections";
import { listForUser } from "@/lib/progress";
import { getSiteContext } from "@/lib/site/context";
import { siteHref, trainingCanonical } from "@/lib/site/hosts";

// Reads the current session + the user's progress on every request.
export const dynamic = "force-dynamic";

export const metadata = {
  title: "Training",
  description:
    "Part 2: how the decoder is trained, from the loss and backprop to optimisers, with Part 1's model training in your browser.",
  alternates: { canonical: trainingCanonical("/training") },
};

export default async function TrainingIndex(): Promise<JSX.Element> {
  const status = await Promise.all(
    TRAINING_SECTIONS.map(async (s) => ({
      ...s,
      ready: (await readTrainingMdx(s.slug)) !== null,
    })),
  );
  const session = await runOrFallback("training-index:auth", getSession, null);
  const userId = (session?.user as { id?: string } | undefined)?.id ?? null;
  const progressRows = userId
    ? await runOrFallback(
        "training-index:progress",
        () => listForUser(userId),
        [],
      )
    : [];
  const progressMap = new Map(
    progressRows.map((p) => [p.sectionSlug, p.status]),
  );
  const ctx = getSiteContext();
  const run = results.defaultRun;
  const first = run.evals[0]!.val;
  const last = run.evals[run.evals.length - 1]!.val;

  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <p className="font-mono text-xs uppercase tracking-widest text-accent dark:text-indigo-300">
        /training · Part 2
      </p>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">
        Training the decoder
      </h1>
      <p className="mt-4 text-neutral-600 dark:text-neutral-300">
        Part 1 ran a decoder with random weights. Part 2 trains it: the same
        model, from the same initial weights, on a public-domain book, in your
        browser. Chapter 1 takes it from a validation loss of {first.toFixed(2)}{" "}
        (random guessing over 64 characters) to {last.toFixed(2)} in{" "}
        {run.config.steps.toLocaleString("en-GB")} steps. Every number on these
        pages comes from a recorded, seeded run (
        <a
          href="https://github.com/BrendanJamesLynskey/transformer-explainer/blob/main/results.md"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          results.md
        </a>
        ), and the in-browser trainer is checked step by step against a
        reference whose gradients match PyTorch&rsquo;s.
      </p>
      <p className="mt-3 text-neutral-600 dark:text-neutral-300">
        New to the model itself? Start with{" "}
        <Link
          href={siteHref("/learn", ctx)}
          className="focus-ring rounded text-accent underline underline-offset-2 dark:text-indigo-300"
        >
          Part 1, Inference
        </Link>
        . Prefer slides? Every chapter has a deck in the{" "}
        <a
          href={TRAINING_DECK_HUB}
          className="focus-ring rounded text-accent underline underline-offset-2 dark:text-indigo-300"
        >
          Transformer Training deck series
        </a>
        .
      </p>

      <ol className="mt-10 divide-y divide-neutral-200 dark:divide-neutral-800">
        {status.map((s, i) => {
          const pStatus = progressMap.get(trainingSectionKey(s.slug));
          return (
            <li key={s.slug} className="py-5">
              <div className="flex items-baseline gap-3">
                <span className="font-mono text-xs text-neutral-500 dark:text-neutral-400">
                  {String(i + 1).padStart(2, "0")}
                </span>
                {s.ready ? (
                  <Link
                    href={`/training/${s.slug}`}
                    className="focus-ring rounded text-lg font-medium text-neutral-900 hover:text-accent dark:text-neutral-100"
                  >
                    {s.title}
                  </Link>
                ) : (
                  <span
                    aria-disabled
                    className="text-lg font-medium text-neutral-500 dark:text-neutral-400"
                  >
                    {s.title}{" "}
                    <span className="ml-1 rounded bg-neutral-200 px-1.5 py-0.5 text-[10px] font-normal uppercase tracking-wide text-neutral-700 dark:bg-neutral-800 dark:text-neutral-300">
                      coming soon
                    </span>
                  </span>
                )}
                {pStatus && pStatus !== "not_started" && (
                  <span
                    className={`ml-2 rounded px-1.5 py-0.5 text-[10px] font-normal uppercase tracking-wide ${
                      pStatus === "completed"
                        ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-200"
                        : "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-200"
                    }`}
                  >
                    {pStatus === "completed" ? "✓ done" : "in progress"}
                  </span>
                )}
              </div>
              <p className="mt-1 pl-9 text-sm text-neutral-600 dark:text-neutral-400">
                {s.summary}
              </p>
            </li>
          );
        })}
      </ol>
    </main>
  );
}
