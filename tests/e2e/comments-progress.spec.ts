/**
 * Phase-8 e2e: signed-in user posts a comment on a section page,
 * navigates back to /learn, and sees an "in progress" badge on the
 * section they just visited.
 */
import { expect, test } from "@playwright/test";

test.skip(
  process.env.E2E_TEST_AUTH !== "true",
  "Set E2E_TEST_AUTH=true to exercise the comments flow.",
);

async function signInAs(
  page: import("@playwright/test").Page,
  username: string,
) {
  const csrfResp = await page.request.get("/api/auth/csrf");
  const { csrfToken } = (await csrfResp.json()) as { csrfToken: string };
  await page.request.post("/api/auth/callback/e2e", {
    form: { csrfToken, username, callbackUrl: "/" },
    maxRedirects: 0,
  });
}

test.describe("comments + progress", () => {
  test("signed-in user can post a comment and progress is tracked", async ({
    page,
  }) => {
    await signInAs(page, "carol");

    const section = "01-overview";
    // Wait for the ProgressTracker's first POST to /api/progress so the
    // in_progress upsert lands before we navigate away.
    const progressDone = page.waitForResponse(
      (r) => r.url().includes("/api/progress") && r.request().method() === "POST",
    );
    await page.goto(`/learn/${section}`);
    await progressDone;

    const body = `Hello from Phase 8 — ${Date.now()}`;
    await page.getByLabel(/comment body/i).fill(body);
    await page.getByRole("button", { name: /^Post$/ }).click();

    // Posted comment renders.
    await expect(page.getByText(body)).toBeVisible({ timeout: 5_000 });

    // /learn server-renders a progress badge for the section we visited.
    // Either "in progress" or "✓ done" is acceptable — short sections can
    // complete on first visit (scroll past 80% + interaction).
    await page.goto("/learn");
    await expect(
      page.getByText(/in progress|✓ done/i).first(),
    ).toBeVisible({ timeout: 5_000 });
  });
});
