/**
 * Pure auth helpers.
 *
 * Kept in their own file so unit tests can import them without dragging in
 * the DB client (which `config.ts` imports transitively via the Drizzle
 * adapter).
 */

/**
 * Map a GitHub OAuth profile into our user shape. Persists both the
 * numeric `id` (immutable, primary key) and the visible `login` (mutable
 * but used by the admin allow-list).
 */
export function mapGitHubProfile(profile: {
  id: number | string;
  login: string;
  name?: string | null;
  email?: string | null;
  avatar_url?: string | null;
}): {
  name: string | null;
  email: string | null;
  image: string | null;
  githubId: string;
  githubLogin: string;
} {
  // Note: we deliberately do NOT return `id`. Our `users.id` column is a
  // Postgres `uuid` with `defaultRandom()` — letting the DB generate it
  // gives us a valid UUID. Returning GitHub's numeric id (e.g. "12345")
  // makes Postgres reject the insert as invalid UUID syntax, which surfaces
  // in production as an OAuthCallbackError on the GitHub sign-in flow.
  return {
    name: profile.name ?? profile.login,
    email: profile.email ?? null,
    image: profile.avatar_url ?? null,
    githubId: String(profile.id),
    githubLogin: profile.login,
  };
}

/**
 * Build the synthetic user the E2E Credentials provider returns. Returns
 * `null` if E2E auth isn't enabled — the provider should reject in that case.
 */
export function buildE2EUser(
  rawUsername: string | undefined,
  enabled: boolean,
): {
  id: string;
  name: string;
  email: string;
  image: null;
  githubId: null;
  githubLogin: string;
} | null {
  if (!enabled) return null;
  const username = String(rawUsername ?? "e2e-user").toLowerCase();
  return {
    id: `e2e:${username}`,
    name: username,
    email: `${username}@example.invalid`,
    image: null,
    githubId: null,
    githubLogin: username,
  };
}

/**
 * Inject our extra fields onto the Auth.js session payload. The DB row that
 * Auth.js hydrates is already typed against `users`, so the cast in
 * `config.ts` reads fields TypeScript doesn't know are there at this seam.
 */
export function enrichSession<S extends { user: object }>(
  session: S,
  user: { githubLogin?: string | null; githubId?: string | null },
): S {
  return {
    ...session,
    user: {
      ...session.user,
      githubLogin: user.githubLogin ?? null,
      githubId: user.githubId ?? null,
    },
  };
}

/**
 * Returns true if the signed-in user is in the admin allow-list. Logins are
 * normalised to lowercase so case differences don't break the gate.
 */
export function isAdmin(
  login: string | null | undefined,
  allowList: ReadonlySet<string>,
): boolean {
  if (!login) return false;
  return allowList.has(login.toLowerCase());
}

/**
 * Where to send the user after sign-in, from the `callbackUrl` query
 * parameter. Only same-site paths are allowed: anything else (an absolute
 * URL, a protocol-relative `//evil.example`, a backslash trick) falls back
 * to `/`, so the sign-in page can't be used as an open redirect. Auth.js
 * applies its own same-origin check later; this keeps the page honest too.
 */
export function safeCallbackUrl(raw: string | string[] | undefined): string {
  const value = Array.isArray(raw) ? raw[0] : raw;
  if (!value || !value.startsWith("/")) return "/";
  if (value.startsWith("//") || value.includes("\\")) return "/";
  return value;
}

/**
 * Human-readable text for an Auth.js `?error=` code on the sign-in page.
 * Returns null when there is no error. Unknown codes get a generic line;
 * the code itself is shown alongside so a bug report can quote it.
 */
export function signInErrorMessage(
  code: string | string[] | undefined,
): string | null {
  const value = Array.isArray(code) ? code[0] : code;
  if (!value) return null;
  switch (value) {
    case "OAuthAccountNotLinked":
      return "That email address is already linked to a different sign-in. Sign in the way you did before.";
    case "AccessDenied":
      return "Sign-in was cancelled or refused.";
    case "Configuration":
      return "The server's sign-in configuration has a problem. Please try again later.";
    default:
      return "Sign-in didn't complete. Please try again.";
  }
}
