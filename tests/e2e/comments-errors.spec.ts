/**
 * Brief 08B e2e: comment failures are visible and lose nothing.
 *
 * In production a comment POST once returned a non-JSON 500 (the comment
 * was stored, the renderer then crashed) and the UI showed nothing, so the
 * owner posted three times. These tests fake that response with
 * `page.route` and check the UI explains it, keeps the draft, and that a
 * failing list says so instead of looking like an empty section.
 */
import { expect, test } from "@playwright/test";

const SECTION = "02-embeddings";
const COMMENTS_API = `**/api/sections/${SECTION}/comments`;

test("a failing comment list shows an error, not an empty section", async ({
  page,
}) => {
  await page.route(COMMENTS_API, (route) =>
    route.request().method() === "GET"
      ? route.fulfill({ status: 500, body: "Internal Server Error" })
      : route.continue(),
  );
  await page.goto(`/learn/${SECTION}`);
  const alert = page.getByTestId("comments-load-error");
  await expect(alert).toBeVisible({ timeout: 10_000 });
  await expect(alert).toContainText(/load the comments/);
  await expect(page.getByText(/Be the first to leave a comment/)).toHaveCount(
    0,
  );
});

test.describe("posting", () => {
  test.skip(
    process.env.E2E_TEST_AUTH !== "true",
    "Set E2E_TEST_AUTH=true to exercise the signed-in comment flow.",
  );

  test("a non-JSON 500 on POST shows an error and keeps the draft", async ({
    page,
  }) => {
    const csrf = await page.request.get("/api/auth/csrf");
    const { csrfToken } = (await csrf.json()) as { csrfToken: string };
    await page.request.post("/api/auth/callback/e2e", {
      form: { csrfToken, username: "dave", callbackUrl: "/" },
      maxRedirects: 0,
    });

    let posts = 0;
    await page.route(COMMENTS_API, (route) => {
      if (route.request().method() !== "POST") return route.continue();
      posts += 1;
      return route.fulfill({ status: 500, body: "Internal Server Error" });
    });

    await page.goto(`/learn/${SECTION}`);
    const draft = `A draft that must survive — ${Date.now()}`;
    const textarea = page.getByLabel(/comment body/i);
    await textarea.fill(draft);
    await page.getByRole("button", { name: /^Post$/ }).click();

    const alert = page.getByTestId("comment-post-error");
    await expect(alert).toBeVisible({ timeout: 10_000 });
    await expect(alert).toContainText(/post your comment/);
    await expect(alert).toContainText(/500/);
    await expect(textarea).toHaveValue(draft);
    // The failed post didn't add a phantom comment to the list.
    await expect(
      page.getByTestId("comments-list").getByText(draft),
    ).toHaveCount(0);
    expect(posts).toBe(1);
  });
});
