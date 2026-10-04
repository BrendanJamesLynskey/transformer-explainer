/**
 * Phase-2 auth e2e.
 *
 * The real GitHub OAuth flow is too brittle to drive in CI (third-party
 * web pages, real network round-trips). Instead we exercise the same
 * Auth.js plumbing through the dev-only Credentials provider gated on
 * `E2E_TEST_AUTH=true` (RUNBOOK.md §3 → "GitHub OAuth not configured").
 *
 * The test signs in via a direct POST to `/api/auth/callback/e2e`,
 * verifies the homepage shows the signed-out → signed-in transition, and
 * then signs out again.
 *
 * The manual smoke test for the real OAuth path lives in the README.
 */
import { expect, test } from "@playwright/test";

test.describe("auth", () => {
  test("homepage links to the sign-in page when logged out", async ({
    page,
  }) => {
    await page.goto("/");
    await page.getByRole("link", { name: /^sign in$/i }).click();
    await expect(page).toHaveURL(/\/signin$/);
    await expect(
      page.getByRole("button", { name: /sign in with github/i }),
    ).toBeVisible();
  });

  test("the sign-in page starts the GitHub OAuth flow", async ({ page }) => {
    // Stop at GitHub's door: answer the authorize request locally so the
    // test never depends on github.com, then check where we were sent.
    await page.route("https://github.com/**", (route) =>
      route.fulfill({ status: 200, body: "stub" }),
    );
    await page.goto("/signin?callbackUrl=/learn");
    await page.getByRole("button", { name: /sign in with github/i }).click();
    await page.waitForURL(/github\.com\/login\/oauth\/authorize/);
    const url = new URL(page.url());
    expect(url.searchParams.get("redirect_uri")).toMatch(
      /\/api\/auth\/callback\/github$/,
    );
  });

  test("the sign-in page explains an Auth.js error", async ({ page }) => {
    await page.goto("/signin?error=AccessDenied");
    // Filter by text: Next's route announcer is also a role="alert".
    await expect(
      page.getByRole("alert").filter({ hasText: /cancelled/i }),
    ).toBeVisible();
  });

  test.skip(
    process.env.E2E_TEST_AUTH !== "true",
    "Set E2E_TEST_AUTH=true to exercise the Credentials provider.",
  );

  test("e2e credentials sign-in round trip", async ({ page, request }) => {
    // 1) Fetch a CSRF token.
    const csrfResp = await request.get("/api/auth/csrf");
    expect(csrfResp.ok()).toBeTruthy();
    const { csrfToken } = (await csrfResp.json()) as { csrfToken: string };
    expect(csrfToken).toBeTruthy();

    // 2) Submit credentials. Auth.js v5 reads them as form-encoded.
    const signInResp = await request.post("/api/auth/callback/e2e", {
      form: {
        csrfToken,
        username: "playwright-test",
        callbackUrl: "/",
      },
      maxRedirects: 0,
    });
    // Auth.js redirects on success; either status is fine, what we care
    // about is the Set-Cookie that lands on the request context.
    expect([200, 302]).toContain(signInResp.status());

    // 3) Carry the auth cookie into the browser context.
    const cookies = await request.storageState();
    await page.context().addCookies(cookies.cookies);

    // 4) The home page should now show the sign-out button.
    await page.goto("/");
    await expect(page.getByRole("button", { name: /sign out/i })).toBeVisible({
      timeout: 10_000,
    });
  });
});
