/**
 * Phase-7 e2e: signed-in user saves a public experiment, signs out,
 * confirms the experiment is reachable as a guest, then signs back in
 * as a different user and forks.
 */
import { expect, test } from "@playwright/test";

test.describe.configure({ mode: "serial" });

test.skip(
  process.env.E2E_TEST_AUTH !== "true",
  "Set E2E_TEST_AUTH=true to exercise the experiments flow.",
);

async function signInAs(
  page: import("@playwright/test").Page,
  username: string,
) {
  // Hit Auth.js's CSRF endpoint via the page's request context (so cookies
  // land in the page's storage), then POST credentials to /api/auth/callback/e2e.
  const csrfResp = await page.request.get("/api/auth/csrf");
  const { csrfToken } = (await csrfResp.json()) as { csrfToken: string };
  await page.request.post("/api/auth/callback/e2e", {
    form: { csrfToken, username, callbackUrl: "/" },
    maxRedirects: 0,
  });
}

const NAME = `Run-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;

test.describe("experiments", () => {
  let savedSlug = "";

  test("signed-in user can save a public experiment from the playground", async ({
    page,
  }) => {
    await signInAs(page, "alice");

    await page.goto("/playground");
    await page.getByRole("button", { name: /save as experiment/i }).click();

    const nameInput = page.getByLabel(/experiment name/i);
    await nameInput.fill(NAME);
    await page.getByLabel(/experiment visibility/i).selectOption("public");
    await page.getByRole("button", { name: /^save$/i }).click();

    // Save redirects to /experiments/<slug>; capture the slug from URL.
    await page.waitForURL(/\/experiments\/.+/);
    const url = page.url();
    const match = url.match(/\/experiments\/(.+)$/);
    expect(match).not.toBeNull();
    savedSlug = match![1]!;

    await expect(
      page.getByRole("heading", { name: new RegExp(`^${NAME}$`) }),
    ).toBeVisible();
  });

  test("a signed-out visitor can read the saved public experiment", async ({
    page,
  }) => {
    await page.context().clearCookies();
    await page.goto(`/experiments/${savedSlug}`);
    await expect(
      page.getByRole("heading", { name: new RegExp(`^${NAME}$`) }),
    ).toBeVisible();
    // Public experiments also surface on the index.
    await page.goto("/experiments");
    await expect(
      page.getByRole("link", { name: new RegExp(`^${NAME}$`) }),
    ).toBeVisible();
  });

  test("a different signed-in user can fork it to their account", async ({
    page,
  }) => {
    await signInAs(page, "bob");
    await page.goto(`/experiments/${savedSlug}`);
    await page.getByRole("button", { name: /fork to my account/i }).click();
    await page.waitForURL(/\/experiments\/.+-/);
    await expect(
      page.getByRole("heading", { name: new RegExp(`${NAME}.*\\(fork\\)`) }),
    ).toBeVisible();
  });
});
