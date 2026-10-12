"use client";

/**
 * The header's two sections: Part 1, Inference (`/learn`) and Part 2,
 * Training (`/training`), with the current one marked (`aria-current`).
 * A client component only to read the path.
 */
import Link from "next/link";
import { usePathname } from "next/navigation";

const LINK =
  "focus-ring rounded px-2 py-1 text-neutral-700 hover:text-neutral-950 dark:text-neutral-300 dark:hover:text-white";
const CURRENT =
  "font-medium text-neutral-950 underline decoration-accent decoration-2 underline-offset-8 dark:text-white dark:decoration-indigo-300";

export function PartNav(): JSX.Element {
  const path = usePathname() ?? "/";
  const parts = [
    {
      href: "/learn",
      label: "Inference",
      part: "Part 1",
      title: "Part 1: how the decoder runs",
    },
    {
      href: "/training",
      label: "Training",
      part: "Part 2",
      title: "Part 2: how it is trained",
    },
  ];
  return (
    <>
      {parts.map((p) => {
        const on = path === p.href || path.startsWith(`${p.href}/`);
        return (
          <Link
            key={p.href}
            href={p.href}
            title={p.title}
            // the site switch also has "Inference" and "Training": keep
            // the names distinct (and still containing the visible word)
            aria-label={`${p.part}: ${p.label}`}
            aria-current={on ? "page" : undefined}
            className={`${LINK} ${on ? CURRENT : ""}`}
          >
            {p.label}
          </Link>
        );
      })}
    </>
  );
}
