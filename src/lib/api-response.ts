/**
 * Client-side reader for this app's API responses.
 *
 * Every route returns `{ ok: true, data } | { ok: false, error }` (CLAUDE.md
 * §5 → Errors). A crashed route doesn't: Vercel answers a 500 with a plain
 * text or HTML page, and `await res.json()` then throws. That is how a
 * comment POST once "did nothing" in production — the comment was stored,
 * the response was a non-JSON 500, and the UI showed no error, so the owner
 * posted three times. `readApiResponse` turns every outcome into the same
 * shape, so callers always have a message to show.
 */

/** The `{ ok, data } | { ok, error }` shape every API route uses. */
export type ApiResult<T> = { ok: true; data: T } | { ok: false; error: string };

/**
 * Read a `fetch` response into an {@link ApiResult}. Never throws.
 *
 * - 2xx with `{ ok: true, data }` → that result.
 * - Any status with `{ ok: false, error }` → that error (the route's message).
 * - Anything else (non-JSON body, unexpected JSON, a non-2xx claiming
 *   `ok: true`) → a generic error naming the HTTP status.
 *
 * @param res     The response from `fetch`.
 * @param action  What the user was doing, for the message ("post your comment").
 */
export async function readApiResponse<T>(
  res: Response,
  action: string,
): Promise<ApiResult<T>> {
  const fallback: ApiResult<T> = {
    ok: false,
    error: `Couldn't ${action} (server error ${res.status}). Please try again.`,
  };
  let json: unknown;
  try {
    json = await res.json();
  } catch {
    return fallback;
  }
  if (typeof json !== "object" || json === null || !("ok" in json)) {
    return fallback;
  }
  const body = json as { ok: unknown; data?: unknown; error?: unknown };
  if (body.ok === true && res.ok && "data" in body) {
    return { ok: true, data: body.data as T };
  }
  if (body.ok === false && typeof body.error === "string" && body.error) {
    return { ok: false, error: body.error };
  }
  return fallback;
}

/**
 * The message for a request that never got a response (offline, DNS, CORS).
 *
 * @param action  What the user was doing, as for {@link readApiResponse}.
 */
export function networkErrorMessage(action: string): string {
  return `Couldn't ${action}: the network request failed. Check your connection and try again.`;
}
