/**
 * Phase-3 e2e: navigate /learn → /learn/02-embeddings, toggle the layers,
 * and confirm the embedding widget hits the API and renders.
 */
import { expect, test } from "@playwright/test";

test.describe("learn", () => {
  test("index lists sections and links to ready ones", async ({ page }) => {
    await page.goto("/learn");
    await expect(
      page.getByRole("heading", { name: /the decoder, step by step/i }),
    ).toBeVisible();

    // Phase 3 ships sections 01 and 02; the rest are "coming soon".
    await expect(page.getByRole("link", { name: /^Overview$/ })).toBeVisible();
    await expect(
      page.getByRole("link", { name: /^Embeddings$/ }),
    ).toBeVisible();
    await expect(page.getByText("coming soon").first()).toBeVisible();
  });

  test("section page renders MDX, the layer toggle, and the embedding widget", async ({
    page,
  }) => {
    await page.goto("/learn/02-embeddings");

    // MDX rendered:
    await expect(
      page.getByRole("heading", { name: /^Embeddings$/ }),
    ).toBeVisible();

    // Layer toggle present with the three switches.
    const concept = page.getByRole("switch", { name: /concept/i });
    const maths = page.getByRole("switch", { name: /maths/i });
    const code = page.getByRole("switch", { name: /code/i });
    await expect(concept).toHaveAttribute("aria-checked", "true");
    await expect(maths).toHaveAttribute("aria-checked", "false");
    await expect(code).toHaveAttribute("aria-checked", "false");

    // Toggling maths flips the data-layer-maths attribute on <html>.
    await maths.click();
    await expect(page.locator("html")).toHaveAttribute(
      "data-layer-maths",
      "on",
    );
    await expect(maths).toHaveAttribute("aria-checked", "true");

    // Embedding widget renders, reacts to input, and produces heatmaps.
    const input = page.getByLabel(/embedding widget input text/i);
    await expect(input).toBeVisible();
    await input.fill("abc");

    // Wait for the response — the heatmap SVG appears once data lands.
    await expect(
      page
        .locator("svg")
        .filter({ has: page.locator("rect") })
        .first(),
    ).toBeVisible({ timeout: 5_000 });

    // The visible token row should reflect the typed input.
    await expect(page.getByText(/^a$/).first()).toBeVisible();
  });
});
