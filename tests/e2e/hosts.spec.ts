/**
 * The site's two addresses (brief 29E, src/lib/site/hosts.ts). Locally,
 * `training.localhost:<port>` plays the training host and `localhost:<port>`
 * its decoder host (Chromium sends `*.localhost` to the loopback), so these
 * tests run the same rules production uses, minus the production names.
 */
import { expect, test, type Page } from "@playwright/test";

const BASE = new URL(
  process.env.PLAYWRIGHT_BASE_URL ?? "http://localhost:3000",
);
const DECODER = `${BASE.protocol}//localhost:${BASE.port}`;
const TRAINING = `${BASE.protocol}//training.localhost:${BASE.port}`;
// the training stand-in's decoder host (see the note in src/lib/site/hosts.ts)
const TRAINING_DECODER = `${BASE.protocol}//decoder.localhost:${BASE.port}`;
const TRAINING_PROD = "https://transformer-training-explained.vercel.app";

/** The /api/events rate limit (60/min) is shared by the whole run: stub it. */
async function stubEvents(page: Page): Promise<void> {
  await page.route("**/api/events", (r) =>
    r.fulfill({
      status: 200,
      contentType: "application/json",
      body: '{"ok":true}',
    }),
  );
}

test.describe("the training host", () => {
  test.beforeEach(async ({ page }) => stubEvents(page));

  test("its home page is /training, and the header names the Training site", async ({
    page,
  }) => {
    await page.goto(`${TRAINING}/`);
    await expect(page).toHaveURL(`${TRAINING}/training`);
    const header = page.locator("header");
    await expect(header.locator("[data-site-name]")).toHaveText(
      "transformer-training",
    );
    await expect(header.locator("[data-site-name]")).toHaveAttribute(
      "href",
      "/training",
    );
    const menu = header.locator("[data-site-switch='menu']");
    await expect(menu.locator("summary")).toHaveText(/Site:\s*Training/);
    // Part 2 is the current part; Part 1 is on the decoder host
    await expect(
      header.getByRole("link", { name: "Part 2: Training" }),
    ).toHaveAttribute("aria-current", "page");
    await expect(
      header.getByRole("link", { name: "Part 1: Inference" }),
    ).toHaveAttribute("href", `${TRAINING_DECODER}/learn`);
    await expect(header.getByRole("link", { name: "About" })).toHaveAttribute(
      "href",
      `${TRAINING_DECODER}/about`,
    );
    await expect(header.getByRole("link", { name: "Playground" })).toHaveCount(
      0,
    );
    // the menu highlights Training
    await menu.locator("summary").click();
    await expect(
      menu.getByRole("link", { name: "Training", exact: true }),
    ).toHaveAttribute("aria-current", "true");
    await expect(
      menu.getByRole("link", { name: "Training", exact: true }),
    ).toHaveAttribute("href", TRAINING_PROD);
  });

  test("chapter links stay on the training host; Part 1 links go to the decoder host", async ({
    page,
  }) => {
    await page.goto(`${TRAINING}/training/01-training-loop`);
    await expect(
      page.getByRole("link", { name: "← /training" }),
    ).toHaveAttribute("href", "/training");
    await expect(
      page.getByRole("link", { name: /Data and tokenisation →/ }),
    ).toHaveAttribute("href", "/training/02-data-and-tokens");
    // a Part 1 link inside the chapter text
    await expect(
      page.locator("article a", { hasText: "the overview" }),
    ).toHaveAttribute("href", `${TRAINING_DECODER}/learn/01-overview`);
  });

  test("Part 1 pages asked for on the training host move to the decoder host", async ({
    page,
  }) => {
    await page.goto(`${TRAINING}/learn/03-attention`);
    await expect(page).toHaveURL(`${TRAINING_DECODER}/learn/03-attention`);
    await expect(page.locator("header [data-site-name]")).toHaveText(
      "transformer-explainer",
    );
    const res = await page.request.get(`${TRAINING}/about?x=1`, {
      maxRedirects: 0,
    });
    expect(res.status()).toBe(308);
    expect(res.headers()["location"]).toBe(`${TRAINING_DECODER}/about?x=1`);
  });

  test("sign-in and the API answer on the training host itself", async ({
    page,
  }) => {
    const signin = await page.request.get(`${TRAINING}/signin`, {
      maxRedirects: 0,
    });
    expect(signin.status()).toBe(200);
    const health = await page.request.get(`${TRAINING}/api/health`, {
      maxRedirects: 0,
    });
    expect(health.status()).toBe(200);
    const comments = await page.request.get(
      `${TRAINING}/api/sections/training-01-training-loop/comments`,
      { maxRedirects: 0 },
    );
    expect(comments.status()).toBe(200);
  });

  test("its sitemap lists Part 2 at the training host", async ({ page }) => {
    const res = await page.request.get(`${TRAINING}/sitemap.xml`);
    const xml = await res.text();
    expect(xml).toContain(`<loc>${TRAINING_PROD}/training/13-slow-links</loc>`);
    expect(xml).not.toContain("/learn");
    const robots = await (
      await page.request.get(`${TRAINING}/robots.txt`)
    ).text();
    expect(robots).toContain(`Sitemap: ${TRAINING_PROD}/sitemap.xml`);
  });
});

test.describe("the decoder host", () => {
  test.beforeEach(async ({ page }) => stubEvents(page));

  test("names the Decoder site on Part 1 pages", async ({ page }) => {
    await page.goto(`${DECODER}/learn`);
    const header = page.locator("header");
    await expect(header.locator("[data-site-name]")).toHaveText(
      "transformer-explainer",
    );
    await expect(
      header.locator("[data-site-switch='menu'] summary"),
    ).toHaveText(/Site:\s*Decoder/);
    await expect(
      header.getByRole("link", { name: "Playground" }),
    ).toBeVisible();
  });

  test("serves Training pages locally (no redirect off production), named Training", async ({
    page,
  }) => {
    await page.goto(`${DECODER}/training/13-slow-links`);
    await expect(page).toHaveURL(`${DECODER}/training/13-slow-links`);
    await expect(
      page.locator("header [data-site-switch='menu'] summary"),
    ).toHaveText(/Site:\s*Training/);
    // canonical: the training host, wherever the page is served
    await expect(page.locator("link[rel='canonical']")).toHaveAttribute(
      "href",
      `${TRAINING_PROD}/training/13-slow-links`,
    );
    // the chapter's slide deck
    await expect(page.locator("a[data-chapter-deck]")).toHaveAttribute(
      "href",
      "https://brendanjameslynskey.github.io/Transformer_Training_12_Training_over_Slow_Links/",
    );
  });

  test("its sitemap lists Part 1 at the decoder host and no Training page", async ({
    page,
  }) => {
    const xml = await (await page.request.get(`${DECODER}/sitemap.xml`)).text();
    expect(xml).toContain(
      "<loc>https://transformer-decoder-explained.vercel.app/learn/03-attention</loc>",
    );
    expect(xml).not.toContain("/training");
  });
});
