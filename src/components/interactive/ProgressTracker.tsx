"use client";

/**
 * Per-section progress auto-tracking. Mounts on /learn/[slug] for
 * signed-in users; signed-out visits are no-ops.
 *
 * Rules (SPEC §6.5):
 *   - any scroll past the viewport entry → in_progress
 *   - 80%+ scroll *and* at least one widget interaction (input, click on
 *     buttons inside .te-widget) → completed
 *
 * The /api/progress upsert is monotonic, so a flicker can't downgrade a
 * completed section.
 */
import { useEffect } from "react";

async function postProgress(
  sectionSlug: string,
  status: "in_progress" | "completed",
) {
  await fetch("/api/progress", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ sectionSlug, status }),
  }).catch(() => undefined);
}

export function ProgressTracker({
  sectionSlug,
  signedIn,
}: {
  sectionSlug: string;
  signedIn: boolean;
}): JSX.Element | null {
  useEffect(() => {
    if (!signedIn) return;

    let interacted = false;
    let scrolledFar = false;
    let inProgressSent = false;
    let completedSent = false;

    function maybeComplete() {
      if (interacted && scrolledFar && !completedSent) {
        completedSent = true;
        void postProgress(sectionSlug, "completed");
      }
    }
    function markInProgress() {
      if (!inProgressSent) {
        inProgressSent = true;
        void postProgress(sectionSlug, "in_progress");
      }
    }

    function onScroll() {
      const top = window.scrollY;
      const total = document.documentElement.scrollHeight - window.innerHeight;
      if (total <= 0) return;
      const pct = top / total;
      if (pct > 0 && !inProgressSent) markInProgress();
      if (pct >= 0.8) {
        scrolledFar = true;
        maybeComplete();
      }
    }

    function onInteraction() {
      interacted = true;
      markInProgress();
      maybeComplete();
    }

    // Fire `in_progress` on mount — landing on the page counts as engagement.
    markInProgress();

    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("input", onInteraction, true);
    window.addEventListener("click", onInteraction, true);
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("input", onInteraction, true);
      window.removeEventListener("click", onInteraction, true);
    };
  }, [sectionSlug, signedIn]);

  return null;
}
