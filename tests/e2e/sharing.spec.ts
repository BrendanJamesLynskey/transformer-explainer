/**
 * Sharing and first impression (brief 39), on both addresses of this deployment: each landing
 * page's start button, its "How this is checked" strip (whose link lands on a real About section)
 * and its hero figure above the fold on a phone; link-preview tags on each landing page and a
 * chapter of each site, with cards that are 1200 × 630 PNGs; and each site's own favicon and
 * Apple icon. (The sitemaps and robots.txt are checked in hosts.spec.ts.)
 */
import {
  expect,
  test,
  type APIRequestContext,
  type Page,
} from "@playwright/test";

import { SECTIONS, TRAINING_SECTIONS } from "@/lib/mdx/sections";

const SITES = [
  {
    home: "/",
    first: `/learn/${SECTIONS[0]!.slug}`,
    icon: /^\/icon\.svg/,
    other: ["/about"],
  },
  {
    home: "/training",
    first: `/training/${TRAINING_SECTIONS[0]!.slug}`,
    icon: /^\/training\/icon\.svg/,
    other: [],
  },
] as const;

async function meta(page: Page, key: string): Promise<string | null> {
  return page
    .locator(`meta[property="${key}"], meta[name="${key}"]`)
    .first()
    .getAttribute("content");
}

/** The path (and query) of an absolute URL, to fetch from the server under test. */
const local = (u: string) => {
  const x = new URL(u);
  return x.pathname + x.search;
};

async function pngSize(request: APIRequestContext, u: string) {
  const res = await request.get(local(u));
  expect(res.status()).toBe(200);
  expect(res.headers()["content-type"]).toBe("image/png");
  const b = await res.body();
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

for (const s of SITES) {
  test(`${s.home}: start button, checked strip, figure above the fold on a phone`, async ({
    page,
  }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto(s.home);
    const start = page.locator("[data-start-button]");
    await expect(start).toHaveText("Start with chapter 1 →");
    await expect(start).toHaveAttribute("href", s.first);
    const fig = page.locator("[data-hero-figure]").first();
    await expect(fig).toBeVisible();
    const box = (await fig.boundingBox())!;
    expect(box.y, "hero figure starts on the first screen").toBeLessThan(
      844 - 120,
    );
    const strip = page.locator("[data-checked-strip]");
    await expect(strip).toContainText("How this is checked");
    const href = (await strip
      .locator("[data-checked-link]")
      .getAttribute("href"))!;
    const [path, id] = href.split("#");
    expect(id, "the strip links to a section").toBeTruthy();
    await page.goto(path!);
    await expect(page.locator(`[id="${id}"]`)).toHaveCount(1);
  });

  for (const path of [s.home, s.first, ...s.other]) {
    test(`${path} carries Open Graph and Twitter cards`, async ({
      page,
      request,
    }) => {
      await page.goto(path);
      expect(await meta(page, "og:title")).toBeTruthy();
      expect(await meta(page, "og:description")).toBeTruthy();
      expect(await meta(page, "og:site_name")).toBeTruthy();
      expect(new URL((await meta(page, "og:url"))!).pathname).toBe(path);
      expect(await meta(page, "twitter:card")).toBe("summary_large_image");
      for (const key of ["og:image", "twitter:image"]) {
        const u = (await meta(page, key))!;
        expect(u, key).toMatch(/^https?:\/\//);
        expect(await pngSize(request, u)).toEqual({ w: 1200, h: 630 });
        // a chapter has its own card
        if (path === s.first && /^\/(learn|training)\//.test(s.first))
          expect(new URL(u).pathname).toContain(s.first);
      }
    });
  }

  test(`${s.home}: the site's own favicon and Apple icon`, async ({
    page,
    request,
  }) => {
    await page.goto(s.home);
    const icon = (await page
      .locator('link[rel="icon"]')
      .last()
      .getAttribute("href"))!;
    expect(icon).toMatch(s.icon);
    const svg = await request.get(icon);
    expect(svg.headers()["content-type"]).toContain("image/svg+xml");
    const apple = (await page
      .locator('link[rel="apple-touch-icon"]')
      .last()
      .getAttribute("href"))!;
    expect(await pngSize(request, `https://x${apple}`)).toEqual({
      w: 180,
      h: 180,
    });
  });
}

test("the training site's card is its own", async ({ page }) => {
  await page.goto("/training");
  expect(await meta(page, "og:site_name")).toBe(
    "Transformer Training Explained",
  );
  expect(await meta(page, "og:image")).toContain("/training/opengraph-image");
  await page.goto("/");
  expect(await meta(page, "og:site_name")).toBe(
    "Transformer Decoder Explained",
  );
});
