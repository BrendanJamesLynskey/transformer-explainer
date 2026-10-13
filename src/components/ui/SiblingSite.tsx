/**
 * The link from one Foundations site to the other, at the top of each
 * chapter index: Decoder's `/learn` says "Next: Training →", Training's
 * `/training` says "← The decoder". Each points at the other site's chapter
 * index on its own host (`siteHref`), so the reader always knows which site
 * they are going to, and never meets "Part 1" / "Part 2".
 *
 * Server Component.
 */
import { getSiteContext } from "@/lib/site/context";
import { siteHref } from "@/lib/site/hosts";

const SIBLINGS = {
  decoder: {
    to: "/training",
    label: "Next: Training →",
    blurb: "how this model learns, from one step to many GPUs",
  },
  training: {
    to: "/learn",
    label: "← The decoder",
    blurb: "how the model you train here runs, one forward pass",
  },
} as const;

export function SiblingSite({
  from,
}: {
  from: keyof typeof SIBLINGS;
}): JSX.Element {
  const s = SIBLINGS[from];
  return (
    <p
      data-sibling-site={from === "decoder" ? "training" : "decoder"}
      className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-neutral-600 dark:text-neutral-400"
    >
      {/* a full page load, never a client-side transition: the header,
          rendered once by the root layout, must name the other site */}
      <a
        href={siteHref(s.to, getSiteContext())}
        className="focus-ring inline-flex min-h-11 items-center rounded-md border border-neutral-300 px-3 font-medium text-neutral-900 hover:border-accent dark:border-neutral-700 dark:text-neutral-100 dark:hover:border-indigo-400"
      >
        {s.label}
      </a>
      <span>{s.blurb}</span>
    </p>
  );
}
