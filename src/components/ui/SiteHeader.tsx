/**
 * Site-wide header.
 *
 * Server Component. Reads the current session via `auth()` and renders a
 * link to the `/signin` page, or a sign-out button driven by a Server
 * Action. Phase 3 will add the
 * three-layer-toggle and theme switcher next to the auth controls.
 */
import Link from "next/link";

import { getSession, isAdmin, signOut } from "@/lib/auth";
import { runOrFallback } from "@/lib/db-fallback";

import { SiteSwitch } from "./SiteSwitch";

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

  return (
    <header className="border-b border-neutral-200 dark:border-neutral-800">
      <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-x-4 gap-y-2 px-4 py-3 sm:px-6">
        <Link
          href="/"
          className="focus-ring rounded font-mono text-sm font-medium tracking-tight"
        >
          transformer-explainer
        </Link>
        <nav className="flex flex-wrap items-center gap-1 text-sm sm:gap-3">
          <Link
            href="/learn"
            className="focus-ring rounded px-2 py-1 text-neutral-700 hover:text-neutral-950 dark:text-neutral-300 dark:hover:text-white"
          >
            Learn
          </Link>
          <Link
            href="/playground"
            className="focus-ring rounded px-2 py-1 text-neutral-700 hover:text-neutral-950 dark:text-neutral-300 dark:hover:text-white"
          >
            Playground
          </Link>
          <Link
            href="/about"
            className="focus-ring rounded px-2 py-1 text-neutral-700 hover:text-neutral-950 dark:text-neutral-300 dark:hover:text-white"
          >
            About
          </Link>
          <SiteSwitch current="decoder" />
          {admin && (
            <Link
              href="/admin"
              className="focus-ring rounded px-2 py-1 text-accent hover:underline"
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
                  <span className="ml-2 font-mono text-xs text-neutral-500">
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
