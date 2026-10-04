/**
 * axe-core a11y smoke tests on the three pages SPEC §10 calls out as the
 * Lighthouse targets. Fails on `serious` and `critical` violations only —
 * `minor` / `moderate` issues come through as console output but don't
 * block CI (they tend to flag colour-contrast on neutrals that we tune
 * deliberately). Every page is scanned in light and in dark mode
 * (`darkMode: "media"`, so Playwright's colorScheme switches it): the dark
 * palette needs its own contrast check.
 */
import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

const PAGES = [
  "/",
  "/learn",
  "/learn/03-attention",
  "/playground",
  "/signin",
] as const;

for (const scheme of ["light", "dark"] as const) {
  test.describe(scheme, () => {
    test.use({ colorScheme: scheme });
    for (const path of PAGES) {
      test(`a11y (${scheme}): ${path} has no serious or critical violations`, async ({
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
            scheme,
            JSON.stringify(blocking, null, 2),
          );
        }
        expect(blocking).toEqual([]);
      });
    }
  });
}
