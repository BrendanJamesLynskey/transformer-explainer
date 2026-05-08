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

    // Sections 01 through 07 should all be shipped and clickable as links.
    for (const name of [
      "Overview",
      "Embeddings",
      "Attention",
      "Feed-forward",
      "LayerNorm & residuals",
      "Stacking blocks",
      "Sampling",
    ]) {
      await expect(
        page.getByRole("link", { name: new RegExp(`^${name}$`) }),
      ).toBeVisible();
    }
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

  test("attention page exposes head selector and hover-row highlight", async ({
    page,
  }) => {
    await page.goto("/learn/03-attention");

    await expect(
      page.getByRole("heading", { name: /^Attention$/ }),
    ).toBeVisible();

    // Wait for the live trace to land — the panel headers are static, but
    // the matrices only render once /api/compute/attention responds.
    await expect(
      page.getByRole("heading", { name: /softmax weights/i }),
    ).toBeVisible();
    const matrices = page.locator("svg").filter({ has: page.locator("rect") });
    await expect(matrices.first()).toBeVisible({ timeout: 10_000 });

    // Multi-head selector renders both heads and toggles the pressed state.
    const head0 = page.getByRole("button", { name: /^0$/ });
    const head1 = page.getByRole("button", { name: /^1$/ });
    await expect(head0).toHaveAttribute("aria-pressed", "true");
    await head1.click();
    await expect(head1).toHaveAttribute("aria-pressed", "true");
    await expect(head0).toHaveAttribute("aria-pressed", "false");

    // Hovering the first row of the scores matrix surfaces the
    // "attends to" strip below the panels.
    const scoresPanel = page
      .getByRole("heading", { name: /Q\s*·\s*Kᵀ/ })
      .locator("xpath=ancestor::div[1]");
    const firstRow = scoresPanel.locator("svg g").nth(0);
    await firstRow.hover();
    await expect(page.getByText(/attends to:/i)).toBeVisible();
  });

  test("FFN page renders the position picker and bar charts", async ({
    page,
  }) => {
    await page.goto("/learn/04-ffn");

    await expect(
      page.getByRole("heading", { name: /^Feed-forward$/i }),
    ).toBeVisible();

    // Wait for the live trace.
    await expect(
      page.getByRole("heading", { name: /Pre-activation/ }),
    ).toBeVisible();
    await expect(
      page
        .locator("svg")
        .filter({ has: page.locator("rect") })
        .first(),
    ).toBeVisible({ timeout: 10_000 });

    // Position picker exposes one button per token.
    const buttons = page
      .getByRole("group", { name: /pick a sequence position/i })
      .getByRole("button");
    await expect(buttons.first()).toHaveAttribute("aria-pressed", "true");
    await buttons.nth(2).click();
    await expect(buttons.nth(2)).toHaveAttribute("aria-pressed", "true");
  });

  test("LayerNorm page renders both sliders and the explanatory copy", async ({
    page,
  }) => {
    await page.goto("/learn/05-layernorm-residuals");

    await expect(
      page.getByRole("heading", { name: /^LayerNorm & residuals$/ }),
    ).toBeVisible();

    // Both sliders are present and accessible.
    await expect(page.getByLabel(/γ \(scale\)/)).toBeVisible();
    await expect(page.getByLabel(/β \(shift\)/)).toBeVisible();

    // MDX rendered: the residual-stream paragraph is in the concept layer.
    await expect(page.getByText(/residual stream/i).first()).toBeVisible();
  });

  test("playground page mounts every widget and the preset strip", async ({
    page,
  }) => {
    await page.goto("/playground");

    await expect(
      page.getByRole("heading", { name: /end-to-end playground/i }),
    ).toBeVisible();

    // Preset strip is visible with its three named buttons.
    await expect(page.getByRole("button", { name: /^Hello$/ })).toBeVisible();
    await expect(
      page.getByRole("button", { name: /^Recent-bias$/ }),
    ).toBeVisible();

    // Sampling section button is present (one of the harder-to-reach widgets).
    await expect(
      page.getByRole("button", { name: /sample the next token/i }),
    ).toBeVisible();

    // Clicking a preset emits the te:preset event; one of the inputs should
    // pick up the new text.
    await page.getByRole("button", { name: /^Recent-bias$/ }).click();
    await expect(page.getByLabel(/embedding widget input text/i)).toHaveValue(
      "the cat",
    );
  });
});
