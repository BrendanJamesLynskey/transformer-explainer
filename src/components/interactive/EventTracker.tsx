"use client";

/**
 * Fires analytics events for a section / page.
 *
 *   - On mount, posts a `page_view` event.
 *   - On the first input/click anywhere on the page, posts one
 *     `widget_interact` event.
 *   - When the page reaches 80% scroll, posts a single `section_complete`.
 *
 * Each event carries an anon `sessionId` (UUID v4) persisted in
 * localStorage so a single visitor's clicks tie together across pages.
 *
 * Beacons are batched into a single POST per page: the three events fire
 * at most once each, so this is at most three network calls per visit.
 */
import { useEffect } from "react";

const SESSION_KEY = "te.sessionId";

function getOrCreateSessionId(): string {
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
    /* ignore — sessionId still works for this tab via the closure below */
  }
  return fresh;
}

type Kind = "page_view" | "widget_interact" | "section_complete" | "exp_view";

async function postEvent(
  sessionId: string,
  kind: Kind,
  sectionSlug: string | null,
  meta?: Record<string, unknown>,
) {
  await fetch("/api/events", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      sessionId,
      events: [
        {
          kind,
          ...(sectionSlug ? { sectionSlug } : {}),
          ...(meta ? { meta } : {}),
        },
      ],
    }),
    keepalive: true,
  }).catch(() => undefined);
}

export function EventTracker({
  sectionSlug = null,
  meta,
  pageKind = "page_view",
}: {
  sectionSlug?: string | null;
  meta?: Record<string, unknown>;
  /** Override the on-mount kind. Use `exp_view` on /experiments/[slug]. */
  pageKind?: Kind;
}): JSX.Element | null {
  useEffect(() => {
    const sessionId = getOrCreateSessionId();

    void postEvent(sessionId, pageKind, sectionSlug, meta);

    let interacted = false;
    let completed = false;

    function onInteract() {
      if (interacted) return;
      interacted = true;
      void postEvent(sessionId, "widget_interact", sectionSlug);
    }
    function onScroll() {
      if (completed) return;
      const top = window.scrollY;
      const total = document.documentElement.scrollHeight - window.innerHeight;
      if (total <= 0) return;
      if (top / total >= 0.8) {
        completed = true;
        void postEvent(sessionId, "section_complete", sectionSlug);
      }
    }

    window.addEventListener("input", onInteract, true);
    window.addEventListener("click", onInteract, true);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      window.removeEventListener("input", onInteract, true);
      window.removeEventListener("click", onInteract, true);
      window.removeEventListener("scroll", onScroll);
    };
    // sectionSlug change → it's a new page, treat as a fresh visit.
    // meta is intentionally not in the deps; we read it once on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sectionSlug, pageKind]);

  return null;
}
