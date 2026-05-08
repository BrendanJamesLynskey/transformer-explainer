"use client";

/**
 * Single-purpose button that POSTs to /api/experiments/[slug]/fork and
 * navigates the user to the new experiment's slug.
 */
import { useRouter } from "next/navigation";
import { useState } from "react";

export function ForkButton({ slug }: { slug: string }): JSX.Element {
  const router = useRouter();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleFork() {
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(`/api/experiments/${slug}/fork`, {
        method: "POST",
      });
      const json = (await res.json()) as
        | { ok: true; data: { slug: string } }
        | { ok: false; error: string };
      if (json.ok) {
        router.push(`/experiments/${json.data.slug}`);
      } else {
        setError(json.error);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <button
        type="button"
        onClick={handleFork}
        disabled={busy}
        className="focus-ring rounded border border-neutral-300 px-3 py-1.5 text-sm text-neutral-700 hover:bg-neutral-100 disabled:opacity-40 dark:border-neutral-700 dark:text-neutral-300 dark:hover:bg-neutral-900"
      >
        {busy ? "Forking…" : "Fork to my account"}
      </button>
      {error && (
        <p role="alert" className="text-sm text-red-700 dark:text-red-300">
          {error}
        </p>
      )}
    </>
  );
}
