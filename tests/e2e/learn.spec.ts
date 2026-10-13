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
  test("the site switch: four tracks, and the button and header say where you are", async ({
    page,
  }) => {
    const menu = page.locator("[data-site-switch='menu']");
    const panel = menu.locator("summary + div");
    const site = (key: string) => panel.locator(`[data-site-key='${key}']`);
    const brand = page.locator("header [data-site-brand]");
    for (const width of [1280, 1024, 390]) {
      await page.setViewportSize({ width, height: 800 });
      await page.goto("/");
      // closed: the button reads "<Track> › <Site>", and the header shows the same place
      await expect(menu).toBeVisible();
      await expect(menu.locator("summary")).toHaveText(
        /^Foundations\W+Decoder\W*$/,
      );
      await expect(brand).toHaveText(/^Foundations\W+Decoder\W*$/);
      expect(await brand.locator("[data-site-place]").textContent()).toBe(
        await menu.locator("summary [data-site-place]").textContent(),
      );
      if (width === 1280)
        expect((await brand.boundingBox())!.height).toBeLessThan(30);
      await expect(panel).toBeHidden();
      // one click opens every track, in the order Foundations, Hardware, Inference, Agents
      await menu.locator("summary").click();
      await expect(panel).toBeVisible();
      await expect(menu.locator("nav > p:first-child")).toHaveText([
        "Foundations",
        "Hardware",
        "Inference",
        "Agents",
      ]);
      await expect(menu.locator("nav > p:nth-child(2)")).toHaveText([
        "One small model, end to end",
        "The machines underneath, chip to rack",
        "Serving models at scale",
        "Building on models",
      ]);
      // every site, in order, with its name, its one-line description and its address
      const SITES = [
        [
          "decoder",
          "Decoder",
          "How it runs, one forward pass",
          "https://transformer-decoder-explained.vercel.app",
        ],
        [
          "training",
          "Training",
          "How it learns, from one step to many GPUs",
          "https://transformer-training-explained.vercel.app",
        ],
        [
          "kernels",
          "Kernels",
          "How a GPU executes the maths",
          "https://gpu-kernels-explained.vercel.app",
        ],
        [
          "silicon",
          "Silicon",
          "Inside a matrix engine",
          "https://systolic-arrays-explained.vercel.app",
        ],
        [
          "racks",
          "Racks",
          "GPUs, networks, power, cooling at scale",
          "https://ai-racks-explained.vercel.app",
        ],
        [
          "inference",
          "Serving",
          "How real systems generate text",
          "https://llm-inference-explained.vercel.app",
        ],
        [
          "architectures",
          "Architectures",
          "How model designs differ",
          "https://llm-architectures-explained.vercel.app",
        ],
        [
          "numerics",
          "Numerics",
          "Number formats and quantisation",
          "https://numerics-explained.vercel.app",
        ],
        [
          "tradeoffs",
          "Trade-offs",
          "Which lever helps which metric",
          "https://inference-tradeoffs-explained.vercel.app",
        ],
        [
          "harnesses",
          "Harnesses",
          "The loop between a model and the world",
          "https://agent-harnesses-explained.vercel.app",
        ],
        [
          "protocols",
          "Protocols",
          "MCP, transports, auth and A2A",
          "https://agent-protocols-explained.vercel.app",
        ],
        [
          "context",
          "Context",
          "Retrieval, memory and context engineering",
          "https://agent-context-explained.vercel.app",
        ],
        [
          "orchestration",
          "Orchestration",
          "Graphs, multi-agent patterns, durability",
          "https://agent-orchestration-explained.vercel.app",
        ],
        ["evals", "Evals", "Measuring agents", ""],
        ["security", "Security", "Threats and defences", ""],
      ] as const;
      expect(
        await panel
          .locator("[data-site-key]")
          .evaluateAll((els) =>
            els.map((e) => e.getAttribute("data-site-key")),
          ),
      ).toEqual(SITES.map((s) => s[0]));
      for (const [key, name, blurb, href] of SITES) {
        await expect(site(key)).toContainText(name);
        await expect(site(key)).toContainText(blurb);
        if (href) await expect(site(key)).toHaveAttribute("href", href);
        // not live yet: shown, marked "(soon)", not a link
        else {
          await expect(site(key)).toContainText("(soon)");
          await expect(site(key)).not.toHaveAttribute("href", /./);
        }
      }
      // each track holds its own sites
      for (const [group, n] of [
        ["foundations", 2],
        ["hardware", 3],
        ["inference", 4],
        ["agents", 6],
      ] as const)
        await expect(
          menu.locator(`nav[data-site-group='${group}'] [data-site-key]`),
        ).toHaveCount(n);
      // no site is named after a track
      await expect(
        panel.getByRole("link", {
          name: /^(Foundations|Hardware|Inference|Agents)\b/,
        }),
      ).toHaveCount(0);
      // this site is highlighted
      await expect(site("decoder")).toHaveAttribute("aria-current", "true");
      await expect(panel.locator("[aria-current]")).toHaveCount(1);
      await expect(panel.getByText("Compute in transit")).toHaveCount(0);
      // every entry is a 44 px touch target
      expect(
        (await site("decoder").boundingBox())!.height,
      ).toBeGreaterThanOrEqual(44);
      // the tracks sit side by side on desktop and stack on a phone
      const f = (await menu
        .locator("nav[data-site-group='foundations']")
        .boundingBox())!;
      const a = (await menu
        .locator("nav[data-site-group='agents']")
        .boundingBox())!;
      if (width >= 1024) expect(Math.round(a.y)).toBe(Math.round(f.y));
      else expect(a.y).toBeGreaterThan(f.y);
      // the open panel stays on screen and the page never scrolls sideways
      const p = (await panel.boundingBox())!;
      expect(p.x).toBeGreaterThanOrEqual(0);
      expect(p.x + p.width).toBeLessThanOrEqual(width);
      expect(p.y + p.height).toBeLessThanOrEqual(800);
      const overflow = await page.evaluate(
        () =>
          document.scrollingElement!.scrollWidth -
          document.scrollingElement!.clientWidth,
      );
      expect(overflow).toBeLessThanOrEqual(0);
    }
    // keyboard: Enter on the button opens the menu (a native <details>, no JavaScript),
    // and Tab goes on to the first site
    await page.goto("/");
    await menu.locator("summary").focus();
    await page.keyboard.press("Enter");
    await expect(panel).toBeVisible();
    await page.keyboard.press("Tab");
    await expect(site("decoder")).toBeFocused();
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
