/**
 * /account — current user's experiments + a sign-out shortcut.
 *
 * Server Component. Redirects to the home page if the user isn't signed in.
 */
import Link from "next/link";
import { redirect } from "next/navigation";

import { getSession } from "@/lib/auth";
import { listByOwner } from "@/lib/experiments";

export const dynamic = "force-dynamic";

export const metadata = {
  title: "Account",
  description: "Your saved experiments.",
};

export default async function AccountPage(): Promise<JSX.Element> {
  const session = await getSession();
  if (!session?.user) redirect("/");
  const userId = (session.user as { id?: string }).id;
  if (!userId) redirect("/");

  const items = await listByOwner(userId);
  const login = (session.user as { githubLogin?: string | null }).githubLogin;

  return (
    <main className="mx-auto max-w-3xl px-6 py-10">
      <p className="font-mono text-xs uppercase tracking-widest text-accent">
        /account
      </p>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">
        {login ? `@${login}` : "Your experiments"}
      </h1>
      <p className="mt-2 text-sm text-neutral-500">
        Saved on this account, newest first. Public ones also show up at{" "}
        <Link href="/experiments" className="underline">
          /experiments
        </Link>
        .
      </p>

      {items.length === 0 ? (
        <p className="mt-12 text-sm text-neutral-500">
          You haven&apos;t saved anything yet. Try the{" "}
          <Link href="/playground" className="underline">
            playground
          </Link>{" "}
          and click &ldquo;Save as experiment&rdquo;.
        </p>
      ) : (
        <ul className="mt-10 divide-y divide-neutral-200 dark:divide-neutral-800">
          {items.map((e) => (
            <li key={e.id} className="py-3">
              <Link
                href={`/experiments/${e.slug}`}
                className="focus-ring rounded text-base font-medium text-neutral-900 hover:text-accent dark:text-neutral-100"
              >
                {e.name}
              </Link>
              <p className="mt-1 text-xs text-neutral-500">
                {e.visibility} · created{" "}
                {new Date(e.createdAt).toLocaleDateString()} · {e.viewCount}{" "}
                views
              </p>
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}
