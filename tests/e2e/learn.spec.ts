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
    // Generous timeout because the dev server may be first-compiling
    // /api/compute/embed on this run.
    await expect(
      page
        .locator("svg")
        .filter({ has: page.locator("rect") })
        .first(),
    ).toBeVisible({ timeout: 10_000 });

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

test.describe("companion site", () => {
  test("the two-group site switch: a toggle and a row on desktop, a dropdown below lg", async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto("/");
    const full = page.locator("[data-site-switch='full']");
    await expect(full).toBeVisible();
    const llm = full.locator("nav[data-site-group='llm']");
    const agents = full.locator("nav[data-site-group='agents']");
    // starts on this site's group
    await expect(llm).toBeVisible();
    await expect(agents).toBeHidden();
    await expect(llm.getByRole("link", { name: "Decoder" })).toHaveAttribute(
      "aria-current",
      "true",
    );
    for (const [name, href] of [
      ["Decoder", "https://transformer-decoder-explained.vercel.app"],
      ["Inference", "https://llm-inference-explained.vercel.app"],
      ["Architectures", "https://llm-architectures-explained.vercel.app"],
      ["Kernels", "https://gpu-kernels-explained.vercel.app"],
      ["Numerics", "https://numerics-explained.vercel.app"],
      ["Silicon", "https://systolic-arrays-explained.vercel.app"],
      ["Trade-offs", "https://inference-tradeoffs-explained.vercel.app"],
    ] as const)
      await expect(llm.getByRole("link", { name })).toHaveAttribute(
        "href",
        href,
      );
    // the toggle shows the agent sites (CSS only)
    await full.getByText("Agents", { exact: true }).click();
    await expect(agents).toBeVisible();
    await expect(llm).toBeHidden();
    await expect(
      agents.getByRole("link", { name: "Harnesses" }),
    ).toHaveAttribute("href", "https://agent-harnesses-explained.vercel.app");
    await expect(
      agents.getByRole("link", { name: "Protocols" }),
    ).toHaveAttribute("href", "https://agent-protocols-explained.vercel.app");
    await expect(agents.getByRole("link", { name: "Context" })).toHaveAttribute(
      "href",
      "https://agent-context-explained.vercel.app",
    );
    for (const soon of ["Orchestration", "Evals", "Security"]) {
      await expect(agents.getByText(soon)).toBeVisible();
      await expect(agents.getByRole("link", { name: soon })).toHaveCount(0);
    }
    // keyboard: the toggle is a pair of radio buttons
    await page
      .getByRole("radio", { name: "Show the LLM systems sites" })
      .focus();
    await page.keyboard.press("Space");
    await expect(llm).toBeVisible();

    // below lg (not sm): the dropdown
    await page.setViewportSize({ width: 768, height: 800 });
    await expect(full).toBeHidden();
    const compact = page.locator("[data-site-switch='compact']");
    await expect(compact).toBeVisible();
    await page.setViewportSize({ width: 390, height: 800 });
    await expect(compact).toBeVisible();
    await compact.locator("summary").click();
    await expect(compact.getByText("LLM systems")).toBeVisible();
    await expect(compact.getByText("Agents", { exact: true })).toBeVisible();
    await expect(
      compact.getByRole("link", { name: "Decoder" }),
    ).toHaveAttribute("aria-current", "true");
    await expect(
      compact.getByRole("link", { name: "Harnesses" }),
    ).toHaveAttribute("href", "https://agent-harnesses-explained.vercel.app");
    await expect(
      compact.getByRole("link", { name: "Protocols" }),
    ).toHaveAttribute("href", "https://agent-protocols-explained.vercel.app");
    await expect(
      compact.getByRole("link", { name: "Context" }),
    ).toHaveAttribute("href", "https://agent-context-explained.vercel.app");
    await expect(compact.getByText("Security")).toBeVisible();
    await expect(compact.getByRole("link", { name: "Security" })).toHaveCount(
      0,
    );
    const box = await compact
      .getByRole("link", { name: "Decoder" })
      .boundingBox();
    expect(box!.height).toBeGreaterThanOrEqual(44);
    const overflow = await page.evaluate(
      () =>
        document.scrollingElement!.scrollWidth -
        document.scrollingElement!.clientWidth,
    );
    expect(overflow).toBeLessThanOrEqual(0);
  });

  test("the sampling chapter ends with a link to inference time", async ({
    page,
  }) => {
    await page.goto("/learn/07-sampling");
    await expect(
      page.getByRole("link", { name: "What happens at inference time →" }),
    ).toHaveAttribute(
      "href",
      "https://llm-inference-explained.vercel.app/learn/01-generation-loop",
    );
  });
});
