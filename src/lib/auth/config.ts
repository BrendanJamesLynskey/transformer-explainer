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
import type { NextAuthConfig } from "next-auth";
import Credentials from "next-auth/providers/credentials";
import GitHub from "next-auth/providers/github";

import { db } from "@/lib/db/client";
import { accounts, sessions, users, verificationTokens } from "@/lib/db/schema";

import { buildE2EUser, enrichSession, mapGitHubProfile } from "./helpers";

const e2eAuthEnabled = process.env.E2E_TEST_AUTH === "true";

const githubProvider = GitHub({
  clientId: process.env.AUTH_GITHUB_ID,
  clientSecret: process.env.AUTH_GITHUB_SECRET,
  profile: mapGitHubProfile,
});

const e2eCredentialsProvider = Credentials({
  id: "e2e",
  name: "E2E Test Login",
  credentials: { username: { label: "Username", type: "text" } },
  authorize(credentials) {
    return buildE2EUser(
      credentials?.username as string | undefined,
      e2eAuthEnabled,
    ) as never;
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
      return enrichSession(session, source);
    },
  },
};
