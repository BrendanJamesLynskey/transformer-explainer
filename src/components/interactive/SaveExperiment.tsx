"use client";

/**
 * "Save as experiment" form on /playground.
 *
 * Reads the embedding widget's input as the prompt (it's the canonical
 * one shared via the preset broadcast) and the default playground config.
 * Phase-7 keeps this minimal — name + visibility, then POST to /api.
 */
import { useRouter } from "next/navigation";
import { useState } from "react";

const DEFAULT_CONFIG = {
  seq_len: 8,
  d_model: 16,
  n_heads: 2,
  d_ff: 32,
  n_blocks: 2,
  vocab_size: 64,
  seed: 42,
};

type Visibility = "public" | "unlisted" | "private";

export function SaveExperiment({
  signedIn,
}: {
  signedIn: boolean;
}): JSX.Element {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("My experiment");
  const [visibility, setVisibility] = useState<Visibility>("public");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!signedIn) {
    return (
      <p className="mt-4 rounded border border-dashed border-neutral-300 px-4 py-3 text-xs text-neutral-500 dark:border-neutral-700">
        Sign in (top-right) to save experiments.
      </p>
    );
  }

  async function handleSave() {
    // Read the live input from the embedding widget — it's the canonical
    // prompt (the preset broadcast updates it across widgets).
    const inputText =
      (
        document.querySelector(
          'input[aria-label="Embedding widget input text"]',
        ) as HTMLInputElement | null
      )?.value ?? "";
    const seedRaw =
      (
        document.querySelector(
          'input[aria-label="Embedding widget seed"]',
        ) as HTMLInputElement | null
      )?.value ?? "42";
    const seed = Number.isFinite(Number(seedRaw)) ? Number(seedRaw) : 42;

    setBusy(true);
    setError(null);
    try {
      const res = await fetch("/api/experiments", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          name,
          visibility,
          inputText,
          config: { ...DEFAULT_CONFIG, seed },
          layout: {},
        }),
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
    <div className="mt-4 rounded-lg border border-neutral-200 bg-neutral-50 p-4 dark:border-neutral-800 dark:bg-neutral-900">
      {!open ? (
        <button
          type="button"
          onClick={() => setOpen(true)}
          className="focus-ring rounded bg-accent px-4 py-1.5 text-sm font-medium text-accent-fg hover:opacity-90"
        >
          Save as experiment
        </button>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2">
          <label className="flex flex-col gap-1 text-sm">
            <span className="font-medium text-neutral-700 dark:text-neutral-300">
              Name
            </span>
            <input
              type="text"
              value={name}
              maxLength={80}
              onChange={(e) => setName(e.target.value)}
              aria-label="Experiment name"
              className="focus-ring rounded border border-neutral-300 bg-white px-2 py-1 text-sm dark:border-neutral-700 dark:bg-neutral-950"
            />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            <span className="font-medium text-neutral-700 dark:text-neutral-300">
              Visibility
            </span>
            <select
              value={visibility}
              onChange={(e) => setVisibility(e.target.value as Visibility)}
              aria-label="Experiment visibility"
              className="focus-ring rounded border border-neutral-300 bg-white px-2 py-1 text-sm dark:border-neutral-700 dark:bg-neutral-950"
            >
              <option value="public">Public — listed everywhere</option>
              <option value="unlisted">Unlisted — link-only</option>
              <option value="private">Private — only you</option>
            </select>
          </label>
          <div className="flex items-center gap-3 sm:col-span-2">
            <button
              type="button"
              onClick={handleSave}
              disabled={busy}
              className="focus-ring rounded bg-accent px-4 py-1.5 text-sm font-medium text-accent-fg hover:opacity-90 disabled:opacity-40"
            >
              {busy ? "Saving…" : "Save"}
            </button>
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="focus-ring rounded border border-neutral-300 px-3 py-1.5 text-xs text-neutral-600 hover:bg-neutral-100 dark:border-neutral-700 dark:text-neutral-300 dark:hover:bg-neutral-900"
            >
              Cancel
            </button>
            {error && (
              <p
                role="alert"
                className="text-sm text-red-700 dark:text-red-300"
              >
                {error}
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
