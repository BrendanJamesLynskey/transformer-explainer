"use client";

/**
 * Named-seed presets for the playground (SPEC §14 Q4). Each preset is a
 * `{ text, seed }` pair that we broadcast as a `te:preset` CustomEvent;
 * the widgets that own those inputs listen for the event and adopt the
 * values. Lightweight, no Context required.
 */
import { useState } from "react";

type Preset = {
  id: string;
  label: string;
  text: string;
  seed: number;
  blurb: string;
};

const PRESETS: Preset[] = [
  {
    id: "hello",
    label: "Hello",
    text: "hello!",
    seed: 42,
    blurb: "The default seed. A reasonable middle ground.",
  },
  {
    id: "recent",
    label: "Recent-bias",
    text: "the cat",
    seed: 7,
    blurb: "Attention tends to pile onto the latest token at this seed.",
  },
  {
    id: "spread",
    label: "Spread",
    text: "abcabca",
    seed: 19,
    blurb: "More even attention across the sequence.",
  },
];

export function PlaygroundPresets(): JSX.Element {
  const [active, setActive] = useState<string | null>(null);

  function apply(p: Preset) {
    setActive(p.id);
    window.dispatchEvent(
      new CustomEvent("te:preset", {
        detail: { text: p.text, seed: p.seed },
      }),
    );
  }

  return (
    <div className="mt-6 rounded-lg border border-neutral-200 bg-neutral-50 p-3 dark:border-neutral-800 dark:bg-neutral-900">
      <p className="font-mono text-xs uppercase tracking-widest text-neutral-500">
        Show me an example
      </p>
      <div className="mt-2 flex flex-wrap gap-2">
        {PRESETS.map((p) => (
          <button
            key={p.id}
            type="button"
            onClick={() => apply(p)}
            aria-pressed={active === p.id}
            className={`focus-ring rounded border px-3 py-1.5 text-xs ${
              active === p.id
                ? "border-accent bg-accent text-accent-fg"
                : "border-neutral-300 text-neutral-700 hover:bg-neutral-100 dark:border-neutral-700 dark:text-neutral-300 dark:hover:bg-neutral-800"
            }`}
            title={p.blurb}
          >
            {p.label}
          </button>
        ))}
      </div>
    </div>
  );
}
