/**
 * Auth.js v5 (NextAuth) configuration.
 *
 * Single source of truth for the auth runtime. The pure helpers
 * (`mapGitHubProfile`, `buildE2EUser`, `enrichSession`, `isAdmin`) live in
 * `./helpers.ts` so unit tests can exercise them without dragging in the
 * DB client.
 *
 * Per RUNBOOK.md §3 ("Blocker: GitHub OAuth not configured"), we support a
 * Credentials-provider escape hatch gated on `E2E_TEST_AUTH=true`. This is
 * for Playwright; production never enables it because the env var is never
 * set there.
 */
import { DrizzleAdapter } from "@auth/drizzle-adapter";
import { eq } from "drizzle-orm";
import type { NextAuthConfig } from "next-auth";
import Credentials from "next-auth/providers/credentials";
import GitHub from "next-auth/providers/github";

import { db } from "@/lib/db/client";
import { accounts, sessions, users, verificationTokens } from "@/lib/db/schema";

import { buildE2EUser, enrichSession, mapGitHubProfile } from "./helpers";

const e2eAuthEnabled = process.env.E2E_TEST_AUTH === "true";

/**
 * For the e2e Credentials path: ensure a real `users` row exists (id =
 * UUID) for the synthetic username. Returns the row, so downstream
 * inserts (experiments, comments, …) have a valid FK to point at.
 */
async function upsertE2EUser(username: string): Promise<{
  id: string;
  name: string;
  email: string;
  image: null;
  githubId: string;
  githubLogin: string;
}> {
  const githubLogin = username.toLowerCase();
  const synthEmail = `${githubLogin}@example.invalid`;
  const existing = await db
    .select()
    .from(users)
    .where(eq(users.githubLogin, githubLogin))
    .limit(1);
  if (existing[0]) {
    return {
      id: existing[0].id,
      name: existing[0].name ?? githubLogin,
      email: existing[0].email ?? synthEmail,
      image: null,
      githubId: existing[0].githubId ?? `e2e:${githubLogin}`,
      githubLogin,
    };
  }
  const inserted = await db
    .insert(users)
    .values({
      name: githubLogin,
      email: synthEmail,
      githubLogin,
      githubId: `e2e:${githubLogin}`,
    })
    .returning();
  const row = inserted[0]!;
  return {
    id: row.id,
    name: row.name ?? githubLogin,
    email: row.email ?? synthEmail,
    image: null,
    githubId: row.githubId ?? `e2e:${githubLogin}`,
    githubLogin,
  };
}

const githubProvider = GitHub({
  clientId: process.env.AUTH_GITHUB_ID,
  clientSecret: process.env.AUTH_GITHUB_SECRET,
  profile: mapGitHubProfile,
});

const e2eCredentialsProvider = Credentials({
  id: "e2e",
  name: "E2E Test Login",
  credentials: { username: { label: "Username", type: "text" } },
  async authorize(credentials) {
    if (!buildE2EUser("anything", e2eAuthEnabled)) return null;
    const username = String(
      (credentials?.username as string | undefined) ?? "e2e-user",
    );
    return (await upsertE2EUser(username)) as never;
  },
});

export const authConfig: NextAuthConfig = {
  adapter: DrizzleAdapter(db, {
    usersTable: users,
    accountsTable: accounts,
    sessionsTable: sessions,
    verificationTokensTable: verificationTokens,
  }),
  providers: e2eAuthEnabled
    ? [githubProvider, e2eCredentialsProvider]
    : [githubProvider],
  // Auth.js v5: Credentials provider only supports JWT sessions, so the E2E
  // build flips strategy. Production stays on database sessions so revoking
  // a session row immediately logs the user out.
  session: { strategy: e2eAuthEnabled ? "jwt" : "database" },
  pages: { signIn: "/api/auth/signin" },
  callbacks: {
    /** Stamp the user with GitHub metadata on first sign-in. */
    async signIn({ user, profile }) {
      if (profile && "login" in profile) {
        Object.assign(user, {
          githubLogin: profile.login as string,
          githubId: String(profile.id),
        });
      }
      return true;
    },
    /**
     * JWT-mode only (E2E build): copy GitHub metadata onto the token so it
     * reaches `session()` below. Production uses database sessions and
     * skips this callback entirely.
     */
    async jwt({ token, user }) {
      if (user) {
        const u = user as {
          githubLogin?: string | null;
          githubId?: string | null;
        };
        token.githubLogin = u.githubLogin ?? null;
        token.githubId = u.githubId ?? null;
      }
      return token;
    },
    /** Add `githubLogin` to the session so the UI / admin gate can read it. */
    async session({ session, user, token }) {
      // database-strategy: adapter hands us the full User row.
      // jwt-strategy:      we read from the token populated by `jwt()`.
      const source = user
        ? (user as {
            githubLogin?: string | null;
            githubId?: string | null;
          })
        : {
            githubLogin:
              (token?.githubLogin as string | null | undefined) ?? null,
            githubId: (token?.githubId as string | null | undefined) ?? null,
          };
      const enriched = enrichSession(session, source);
      // JWT strategy doesn't auto-populate session.user.id — copy it from
      // the token (Auth.js sets token.sub = user.id by default).
      if (!user && token?.sub) {
        enriched.user = {
          ...enriched.user,
          id: token.sub,
        } as typeof enriched.user;
      }
      return enriched;
    },
  },
};
