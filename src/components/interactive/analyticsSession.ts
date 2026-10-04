/**
 * The browser's anonymous analytics session id.
 *
 * A UUID persisted in `localStorage`, so one visitor's events tie together
 * across pages without needing an account. `EventTracker` sends it in its
 * event batches; `CommentSection` and `ProgressTracker` send it in the
 * {@link SESSION_HEADER} header so the events the server records for them
 * join the same session.
 */
import { SESSION_HEADER } from "@/lib/analytics-constants";

const SESSION_KEY = "te.sessionId";

/** Read the session id from `localStorage`, creating one on first use. */
export function getOrCreateSessionId(): string {
  try {
    const existing = window.localStorage.getItem(SESSION_KEY);
    if (existing) return existing;
  } catch {
    /* localStorage may be disabled — fall through */
  }
  const fresh =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `s_${Math.random().toString(36).slice(2)}_${Date.now()}`;
  try {
    window.localStorage.setItem(SESSION_KEY, fresh);
  } catch {
    /* ignore — the id still works for this page load */
  }
  return fresh;
}

/** Headers for a JSON POST that carries the analytics session id. */
export function jsonHeadersWithSession(): Record<string, string> {
  return {
    "content-type": "application/json",
    [SESSION_HEADER]: getOrCreateSessionId(),
  };
}
