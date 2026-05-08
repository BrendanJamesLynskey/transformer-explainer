/**
 * Site-wide header.
 *
 * Server Component. Reads the current session via `auth()` and renders a
 * sign-in or sign-out button driven by Server Actions. Phase 3 will add the
 * three-layer-toggle and theme switcher next to the auth controls.
 */
import Link from "next/link";

import { getSession, isAdmin, signIn, signOut } from "@/lib/auth";

async function signInAction() {
  "use server";
  await signIn("github", { redirectTo: "/" });
}

async function signOutAction() {
  "use server";
  await signOut({ redirectTo: "/" });
}

export async function SiteHeader(): Promise<JSX.Element> {
  const session = await getSession();
  const user = session?.user;
  const login = (user as { githubLogin?: string | null } | undefined)
    ?.githubLogin;
  const admin = isAdmin(login);

  return (
    <header className="border-b border-neutral-200 dark:border-neutral-800">
      <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-3">
        <Link
          href="/"
          className="focus-ring rounded font-mono text-sm font-medium tracking-tight"
        >
          transformer-explainer
        </Link>
        <nav className="flex items-center gap-3 text-sm">
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
            <form action={signInAction}>
              <button
                type="submit"
                className="focus-ring rounded bg-accent px-3 py-1 text-accent-fg hover:opacity-90"
              >
                Sign in with GitHub
              </button>
            </form>
          )}
        </nav>
      </div>
    </header>
  );
}
