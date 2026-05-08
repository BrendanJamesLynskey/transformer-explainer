/**
 * /playground — full-pipeline sandbox.
 *
 * Server Component shell that mounts the four interactive widgets in a
 * single page. Each one runs against the same compute APIs the
 * /learn/* widgets use, so a learner can flip between targeted
 * explanations and the full sandbox without re-learning the controls.
 *
 * SPEC §6.3 + §14 Q4 (presets) — the named-seed presets land as a top
 * strip that just nudges the inputs; saved-experiments come in Phase 7.
 */
import { AttentionWidget } from "@/components/interactive/AttentionWidget";
import { EmbeddingWidget } from "@/components/interactive/EmbeddingWidget";
import { FFNWidget } from "@/components/interactive/FFNWidget";
import { PlaygroundPresets } from "@/components/interactive/PlaygroundPresets";
import { SamplingWidget } from "@/components/interactive/SamplingWidget";
import { SaveExperiment } from "@/components/interactive/SaveExperiment";
import { StackingWidget } from "@/components/interactive/StackingWidget";
import { getSession } from "@/lib/auth";

export const metadata = {
  title: "Playground",
  description:
    "Type tokens, watch every operation execute, and sample the next token end-to-end.",
};

export default async function PlaygroundPage(): Promise<JSX.Element> {
  const session = await getSession();
  return (
    <main className="mx-auto max-w-4xl px-6 py-10">
      <p className="font-mono text-xs uppercase tracking-widest text-accent">
        /playground
      </p>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">
        End-to-end playground
      </h1>
      <p className="mt-3 text-neutral-600 dark:text-neutral-300">
        Same compute path as the explainer pages, all on one screen. Pick a
        preset to load a curated seed, or change the inputs in any widget.
      </p>

      <PlaygroundPresets />
      <SaveExperiment signedIn={!!session?.user} />

      <Section title="1 · Embeddings">
        <EmbeddingWidget />
      </Section>
      <Section title="2 · Attention">
        <AttentionWidget />
      </Section>
      <Section title="3 · Feed-forward">
        <FFNWidget />
      </Section>
      <Section title="4 · Stacking">
        <StackingWidget />
      </Section>
      <Section title="5 · Sampling">
        <SamplingWidget />
      </Section>
    </main>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}): JSX.Element {
  return (
    <section className="mt-10">
      <h2 className="font-mono text-sm uppercase tracking-widest text-neutral-500">
        {title}
      </h2>
      {children}
    </section>
  );
}
