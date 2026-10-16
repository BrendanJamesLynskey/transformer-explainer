/**
 * Brief 41: the landing hero shows the trained model's four heads, drawn
 * from the same numbers scripts/reference.py computes in PyTorch
 * (tests/unit/fixtures/animations_trained.json), and the same head at
 * random init for contrast.
 */
import { expect, test } from "@playwright/test";

import fx from "../unit/fixtures/animations_trained.json";

test("the landing hero is the trained heads, cell for cell", async ({
  page,
}) => {
  await page.goto("/");
  const svg = page.locator("[data-hero-figure] [data-trained-heads]");
  await expect(svg).toBeVisible();
  // four trained heads and one random-init head
  await expect(svg.locator("[data-head]")).toHaveCount(5);
  const show = (c: string) => (c === " " ? "␣" : c);
  const text = fx.preview.text;
  for (const [k, h] of fx.preview.heads.entries()) {
    const grid = svg.locator("[data-head]").nth(k);
    await expect(grid).toHaveAttribute("data-head", `${h.block}-${h.head}`);
    // the last row's cells carry the reference's weights
    const last = text.length - 1;
    for (const j of [0, last - 1, last]) {
      const w = h.weights[last]![j]!;
      // cells are drawn row by row, one per (query, key)
      await expect(
        grid.locator("rect title").nth(last * text.length + j),
      ).toHaveText(
        `"${show(text[last]!)}" → "${show(text[j]!)}": ${w.toFixed(3)}`,
      );
    }
  }
  await expect(page.locator("[data-hero-figure] figcaption")).toContainText(
    "windows of the book",
  );
});
