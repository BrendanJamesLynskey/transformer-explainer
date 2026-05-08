/**
 * Captures the README screenshots. Manual run, output committed.
 *
 *   pnpm dev          # in another shell
 *   pnpm screenshots  # fires up a headless Chromium and writes PNGs
 *
 * The intent is "stable, identifiable, no clutter": light theme, fixed
 * viewport, deterministic seeds (the widgets default to seed=42), and
 * the playground doesn't need a sign-in for any of the captures.
 */
import { mkdirSync } from "node:fs";
import path from "node:path";

import { chromium } from "@playwright/test";

const OUT = path.join(process.cwd(), "docs", "screenshots");
const BASE = process.env.SCREENSHOT_BASE_URL ?? "http://localhost:3000";

type Shot = {
  name: string;
  path: string;
  /** Optional CSS selector to scroll into view + clip to. */
  clip?: string;
};

const SHOTS: Shot[] = [
  { name: "01-landing", path: "/" },
  { name: "02-learn-index", path: "/learn" },
  { name: "03-attention", path: "/learn/03-attention" },
  { name: "04-playground", path: "/playground" },
  { name: "05-experiments", path: "/experiments" },
  { name: "06-about", path: "/about" },
];

async function main() {
  mkdirSync(OUT, { recursive: true });

  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    deviceScaleFactor: 2,
    colorScheme: "light",
  });

  for (const shot of SHOTS) {
    const page = await context.newPage();
    const url = `${BASE}${shot.path}`;
    // eslint-disable-next-line no-console
    console.log(`-> ${url}`);
    await page.goto(url, { waitUntil: "networkidle" });
    // give D3 transitions time to settle
    await page.waitForTimeout(300);

    const out = path.join(OUT, `${shot.name}.png`);
    await page.screenshot({ path: out, fullPage: true });
    await page.close();
  }

  await context.close();
  await browser.close();
  // eslint-disable-next-line no-console
  console.log(`wrote ${SHOTS.length} PNGs to ${OUT}`);
}

main().catch((err) => {
  // eslint-disable-next-line no-console
  console.error(err);
  process.exit(1);
});
