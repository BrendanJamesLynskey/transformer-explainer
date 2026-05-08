/**
 * Auth.js v5 entrypoint.
 *
 * `NextAuth(config)` returns the runtime helpers. Importing from here keeps
 * the rest of the codebase one short import away from `auth()` / `signIn()`
 * / `signOut()`. The handlers (`GET`/`POST`) are re-exported from
 * `src/app/api/auth/[...nextauth]/route.ts`.
 */
import NextAuth from "next-auth";

import { adminLogins } from "@/lib/env";

import { authConfig } from "./config";
import { isAdmin as isAdminFn } from "./helpers";

export const { auth, handlers, signIn, signOut } = NextAuth(authConfig);

/**
 * Server-side helper: returns the current session, or null. Thin shim around
 * `auth()` so callers don't need to know about Auth.js v5 specifically.
 */
export async function getSession() {
  return auth();
}

/**
 * Returns true if the signed-in user is in the admin allow-list. Logins are
 * normalised to lowercase so case differences don't break the gate.
 */
export function isAdmin(login: string | null | undefined): boolean {
  return isAdminFn(login, adminLogins);
}
