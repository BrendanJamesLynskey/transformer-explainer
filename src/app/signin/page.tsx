/**
 * /signin — the site's own sign-in page, replacing Auth.js's default one.
 *
 * Server Component. GitHub is the only provider, so the page is one button
 * that runs a Server Action calling `signIn("github")`. Auth.js sends users
 * here (via `pages.signIn` / `pages.error` in `lib/auth/config.ts`) whenever
 * it needs a sign-in or a sign-in fails, with `?callbackUrl=` and
 * `?error=` in the query string.
 */
import type { Metadata } from "next";
import Link from "next/link";
import { redirect } from "next/navigation";

import { getSession, signIn } from "@/lib/auth";
import { safeCallbackUrl, signInErrorMessage } from "@/lib/auth/helpers";
import { runOrFallback } from "@/lib/db-fallback";

export const metadata: Metadata = {
  title: "Sign in",
  robots: { index: false, follow: false },
};

type SearchParams = {
  callbackUrl?: string | string[];
  error?: string | string[];
};

export default async function SignInPage({
  searchParams,
}: {
  searchParams: SearchParams;
}): Promise<JSX.Element> {
  const callbackUrl = safeCallbackUrl(searchParams.callbackUrl);
  const error = signInErrorMessage(searchParams.error);
  const errorCode = Array.isArray(searchParams.error)
    ? searchParams.error[0]
    : searchParams.error;

  // Already signed in? Nothing to do here.
  const session = await runOrFallback("signin:auth", getSession, null);
  if (session?.user) redirect(callbackUrl as never);

  async function signInWithGitHub() {
    "use server";
    await signIn("github", { redirectTo: callbackUrl });
  }

  return (
    <main className="mx-auto max-w-md px-6 py-16 sm:py-24">
      <p className="font-mono text-xs uppercase tracking-widest text-accent dark:text-indigo-300">
        Sign in
      </p>
      <h1 className="mt-3 text-3xl font-semibold tracking-tight">
        Save, share and discuss
      </h1>
      <p className="mt-4 text-neutral-600 dark:text-neutral-300">
        Everything on this site works without an account. Signing in lets you
        save playground configurations as experiments, fork other readers&rsquo;
        experiments, comment on a section, and keep track of the sections
        you&rsquo;ve finished.
      </p>

      {error ? (
        <div
          role="alert"
          className="mt-6 rounded border border-red-300 bg-red-50 p-3 text-sm text-red-800 dark:border-red-900 dark:bg-red-950 dark:text-red-200"
        >
          {error}
          {errorCode ? (
            <span className="ml-1 font-mono text-xs opacity-75">
              ({errorCode})
            </span>
          ) : null}
        </div>
      ) : null}

      <form action={signInWithGitHub} className="mt-8">
        <button
          type="submit"
          className="focus-ring flex w-full items-center justify-center gap-3 rounded bg-neutral-900 px-4 py-3 text-sm font-medium text-white hover:bg-neutral-700 dark:bg-white dark:text-neutral-900 dark:hover:bg-neutral-200"
        >
          <svg
            aria-hidden="true"
            viewBox="0 0 16 16"
            width="18"
            height="18"
            fill="currentColor"
          >
            <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8Z" />
          </svg>
          Sign in with GitHub
        </button>
      </form>

      <p className="mt-6 text-xs text-neutral-600 dark:text-neutral-400">
        GitHub shares your public profile (name, login, avatar) and email
        address with this site. Your GitHub password never passes through it.{" "}
        <Link
          href="/about"
          className="focus-ring rounded underline underline-offset-2"
        >
          About this project
        </Link>
      </p>
    </main>
  );
}
