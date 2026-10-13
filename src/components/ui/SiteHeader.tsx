/**
 * Site-wide header.
 *
 * Server Component. Reads the current session via `auth()` and renders a
 * link to the `/signin` page, or a sign-out button driven by a Server
 * Action.
 *
 * One deployment serves two sites of the Foundations track, each at its own
 * address (`src/lib/site/hosts.ts`): Decoder (`/learn`, the playground) and
 * Training (`/training`). The header names the site the page belongs to, as
 * its place in the companion sites ("Foundations › Decoder" or
 * "Foundations › Training", from the site switch's `SitePlace`, so it never
 * drifts from the menu), and its nav lists only that site's pages: Learn
 * (its chapters), Playground on Decoder, About. The other site is one entry
 * in the site menu, and the chapter indexes link it ("Next: Training →",
 * "← The decoder").
 */
import Link from "next/link";

import { getSession, isAdmin, signOut } from "@/lib/auth";
import { runOrFallback } from "@/lib/db-fallback";
import { getSiteContext } from "@/lib/site/context";
import { siteHref } from "@/lib/site/hosts";

import { placeOf, SitePlace, SiteSwitch } from "./SiteSwitch";

async function signOutAction() {
  "use server";
  await signOut({ redirectTo: "/" });
}

export async function SiteHeader(): Promise<JSX.Element> {
  const session = await runOrFallback("header:auth", getSession, null);
  const user = session?.user;
  const login = (user as { githubLogin?: string | null } | undefined)
    ?.githubLogin;
  const admin = isAdmin(login);
  const ctx = getSiteContext();
  const training = ctx.site === "training";
  const href = (path: string) => siteHref(path, ctx);

  return (
    <header className="border-b border-neutral-200 dark:border-neutral-800">
      <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-x-4 gap-y-2 px-4 py-3 sm:px-6">
        <Link
          href={training ? href("/training") : href("/")}
          title={placeOf(ctx.site).title}
          data-site-name={ctx.site}
          data-site-brand
          className="focus-ring whitespace-nowrap rounded text-sm tracking-tight"
        >
          <SitePlace current={ctx.site} />
        </Link>
        <nav
          aria-label="Site"
          className="flex flex-wrap items-center gap-1 text-sm sm:gap-3"
        >
          <Link
            href={training ? href("/training") : href("/learn")}
            data-site-chapters
            className="focus-ring rounded px-2 py-1 text-neutral-700 hover:text-neutral-950 dark:text-neutral-300 dark:hover:text-white"
          >
            Learn
          </Link>
          {!training && (
            <Link
              href={href("/playground")}
              className="focus-ring rounded px-2 py-1 text-neutral-700 hover:text-neutral-950 dark:text-neutral-300 dark:hover:text-white"
            >
              Playground
            </Link>
          )}
          <Link
            href={href("/about")}
            className="focus-ring rounded px-2 py-1 text-neutral-700 hover:text-neutral-950 dark:text-neutral-300 dark:hover:text-white"
          >
            About
          </Link>
          <SiteSwitch current={ctx.site} />
          {admin && (
            <Link
              href={href("/admin")}
              className="focus-ring rounded px-2 py-1 text-accent hover:underline dark:text-indigo-300"
            >
              Admin
            </Link>
          )}
          {user ? (
            <form action={signOutAction}>
              <button
                type="submit"
                aria-label={`Sign out (${login ?? user.email ?? ""})`}
                className="focus-ring rounded border border-neutral-300 px-3 py-1 text-neutral-700 hover:bg-neutral-100 dark:border-neutral-700 dark:text-neutral-300 dark:hover:bg-neutral-900"
              >
                Sign out
                {login ? (
                  <span className="ml-2 font-mono text-xs text-neutral-500 dark:text-neutral-400">
                    @{login}
                  </span>
                ) : null}
              </button>
            </form>
          ) : (
            <Link
              href="/signin"
              className="focus-ring rounded bg-accent px-3 py-1 text-accent-fg hover:opacity-90"
            >
              Sign in
            </Link>
          )}
        </nav>
      </div>
    </header>
  );
}
