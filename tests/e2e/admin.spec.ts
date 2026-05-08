/**
 * Phase-9 e2e: events flow on visits + admin dashboard / 403 gate.
 *
 *   1. signed-in admin (`testadmin`) visits a section, then /admin →
 *      sees a non-empty funnel row for that section.
 *   2. /api/admin/metrics returns 403 to a non-admin.
 *   3. /admin returns 404 to a non-admin (we use `notFound()` rather
 *      than 403 so the route's existence isn't leaked).
 */
import { expect, test } from "@playwright/test";

test.skip(
  process.env.E2E_TEST_AUTH !== "true",
  "Set E2E_TEST_AUTH=true for the admin flow.",
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

test.describe("admin", () => {
  test("admin sees funnel rows after a section visit", async ({ page }) => {
    await signInAs(page, "testadmin");

    // Visit a section + wait for the page_view event to land.
    const eventLanded = page.waitForResponse(
      (r) => r.url().includes("/api/events") && r.request().method() === "POST",
    );
    await page.goto("/learn/02-embeddings");
    await eventLanded;

    // Hit the metrics endpoint directly — non-zero counts prove ingestion.
    const res = await page.request.get("/api/admin/metrics");
    expect(res.status()).toBe(200);
    const json = (await res.json()) as {
      ok: true;
      data: {
        funnel: { sectionSlug: string; views: number }[];
      };
    };
    const row = json.data.funnel.find((r) => r.sectionSlug === "02-embeddings");
    expect(row?.views ?? 0).toBeGreaterThan(0);

    // Dashboard renders the section funnel card. `domcontentloaded` is enough
    // — we don't need every keep-alive `/api/events` to settle before
    // asserting on the SSR HTML.
    await page.goto("/admin", { waitUntil: "domcontentloaded" });
    await expect(page.getByTestId("dau-card")).toBeVisible();
    await expect(page.getByTestId("funnel-card")).toBeVisible();
    await expect(page.getByText("02-embeddings").first()).toBeVisible();
  });

  test("non-admin gets 403 on /api/admin/metrics", async ({ page }) => {
    await signInAs(page, "alice");
    const res = await page.request.get("/api/admin/metrics");
    expect(res.status()).toBe(403);
  });

  test("non-admin gets 404 on /admin", async ({ page }) => {
    await signInAs(page, "alice");
    const res = await page.goto("/admin");
    expect(res?.status()).toBe(404);
  });
});
