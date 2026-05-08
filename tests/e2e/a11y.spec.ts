/**
 * axe-core a11y smoke tests on the three pages SPEC §10 calls out as the
 * Lighthouse targets. Fails on `serious` and `critical` violations only —
 * `minor` / `moderate` issues come through as console output but don't
 * block CI (they tend to flag colour-contrast on neutrals that we tune
 * deliberately).
 */
import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

const PAGES = ["/", "/learn/03-attention", "/playground"] as const;

for (const path of PAGES) {
  test(`a11y: ${path} has no serious or critical violations`, async ({
    page,
  }) => {
    await page.goto(path);
    // Give D3 viz a beat to mount; axe scans the live DOM.
    await page.waitForLoadState("networkidle");

    const results = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
      .analyze();

    const blocking = results.violations.filter((v) =>
      ["serious", "critical"].includes(v.impact ?? ""),
    );
    if (blocking.length > 0) {
      // eslint-disable-next-line no-console
      console.log(
        "axe blocking violations on",
        path,
        JSON.stringify(blocking, null, 2),
      );
    }
    expect(blocking).toEqual([]);
  });
}
