/**
 * Brief 44, the learning layer: on one chapter of each site a self-check
 * waits for a committed answer, then reveals the model's value and says
 * whether the guess was right; with no JavaScript the answer is still one
 * click away (a native <details>); Training chapter 5's race has its
 * Nesterov line; the Foundations start page lists the reading order and a
 * glossary. axe runs on a chapter with its answers revealed.
 */
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

import { formatValue, lookup } from "@/components/training/R";
import { learnValue } from "@/lib/learn/values";

async function check(page: Page, id: string) {
  const box = page.locator(`[data-self-check="${id}"]`);
  await expect(box).toBeVisible();
  return box;
}

test.describe("self-checks", () => {
  test("Decoder ch 3: the reveal waits for an answer, then shows the model's", async ({
    page,
  }) => {
    await page.goto("/learn/03-attention");
    await expect(page.locator("[data-check-yourself-link]")).toHaveAttribute(
      "href",
      "#check-yourself",
    );
    const box = await check(page, "d03-q1");
    const summary = box.locator("summary");
    const details = box.locator("details");
    // not committed yet: the click is refused and the reader is told why
    await summary.click();
    await expect(details).not.toHaveAttribute("open", "");
    await expect(box).toContainText("Commit to an answer first");
    // a wrong guess, then the reveal
    const want = String(learnValue("decoder.maskedPerHead"));
    await box.getByRole("spinbutton").fill("32");
    await summary.click();
    await expect(details).toHaveAttribute("open", "");
    await expect(box.locator("[data-answer]")).toHaveText(want);
    await expect(box.locator("[data-verdict]")).toHaveAttribute(
      "data-verdict",
      "wrong",
    );
    // the right one
    await box.getByRole("spinbutton").fill(want);
    await expect(box.locator("[data-verdict]")).toHaveAttribute(
      "data-verdict",
      "right",
    );
    // a choice question, by keyboard
    const q2 = await check(page, "d03-q3");
    await q2.getByRole("slider").focus();
    await page.keyboard.press("End");
    await q2.locator("summary").press("Enter");
    await expect(q2.locator("details")).toHaveAttribute("open", "");
    await expect(q2.locator("[data-answer]")).toHaveText(
      `${formatValue(learnValue("decoder.b1h1CurrentPrev"), { pct: true, d: 0 }).replace(/%$/, "")}%`,
    );
  });

  test("Training ch 5: Nesterov in the race, and its self-check", async ({
    page,
  }) => {
    await page.goto("/training/05-optimisers");
    const legend = page.locator(
      '[data-testid="optim2d-animation"] [data-legend="nesterov"]',
    );
    await expect(legend).toHaveCount(1);
    await expect(legend).toContainText("Nesterov");
    const box = await check(page, "t05-q1");
    await box.getByLabel("Nesterov momentum").check();
    await box.locator("summary").click();
    await expect(box.locator("[data-verdict]")).toHaveAttribute(
      "data-verdict",
      "right",
    );
    await expect(box.locator("[data-self-check-answer]")).toContainText(
      formatValue(lookup("optim2d.byKind.nesterov.overshootX"), { d: 3 }),
    );
    const q2 = await check(page, "t05-q2");
    await q2.getByRole("spinbutton").fill("0.04");
    await q2.locator("summary").click();
    await expect(q2.locator("[data-answer]")).toHaveText(
      formatValue(learnValue("training.valleySgdLimit"), { d: 2 }),
    );
    await expect(q2.locator("[data-verdict]")).toHaveAttribute(
      "data-verdict",
      "right",
    );
  });

  test("axe: no serious violations with the answers revealed (390 px, dark)", async ({
    browser,
  }) => {
    const ctx = await browser.newContext({
      colorScheme: "dark",
      viewport: { width: 390, height: 900 },
    });
    const page = await ctx.newPage();
    await page.goto("/training/12-model-parallelism");
    for (const id of ["t12-q1", "t12-q2", "t12-q3"]) {
      const box = await check(page, id);
      const slider = box.getByRole("slider");
      if (await slider.count()) {
        await slider.focus();
        await page.keyboard.press("ArrowRight");
      } else if (await box.getByRole("spinbutton").count()) {
        await box.getByRole("spinbutton").fill("10");
      } else {
        await box.getByRole("radio").first().check();
      }
      await box.locator("summary").click();
      await expect(box.locator("details")).toHaveAttribute("open", "");
    }
    const overflow = await page.evaluate(
      () =>
        document.documentElement.scrollWidth >
        document.documentElement.clientWidth,
    );
    expect(overflow).toBe(false);
    const results = await new AxeBuilder({ page })
      .include("#check-yourself")
      .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
      .analyze();
    expect(
      results.violations.filter((v) =>
        ["serious", "critical"].includes(v.impact ?? ""),
      ),
    ).toEqual([]);
    await ctx.close();
  });
});

test.describe("no JavaScript", () => {
  test.use({ javaScriptEnabled: false });
  test("the question and the answer are in the page; the answer opens with a click", async ({
    page,
  }) => {
    await page.goto("/learn/07-sampling");
    const box = page.locator('[data-self-check="d07-q2"]');
    await expect(box).toContainText("how many of the three tokens survive");
    await expect(box.locator("[data-answer]")).toHaveText(
      String(learnValue("decoder.topP09Kept")),
    );
    await box.locator("summary").click();
    await expect(box.locator("details")).toHaveAttribute("open", "");
    await expect(box.locator("[data-answer]")).toBeVisible();
    await expect(page.locator("[data-remember]")).toContainText(
      "What to remember",
    );
  });
});

test("the Foundations start page: reading order, prerequisites, glossary", async ({
  page,
}) => {
  await page.goto("/");
  await page.locator("[data-foundations-link]").first().click();
  await expect(page).toHaveURL(/\/foundations$/);
  await expect(page.getByTestId("reading-order")).toContainText("Decoder");
  await expect(page.getByTestId("reading-order")).toContainText("Training");
  await expect(
    page.getByRole("heading", { name: "What you need first" }),
  ).toBeVisible();
  const terms = page.locator("[data-foundations-glossary] a");
  expect(await terms.count()).toBeGreaterThanOrEqual(30);
  await expect(
    page.locator(
      '[data-foundations-glossary] a[href$="/training/05-optimisers#t-nesterov"]',
    ),
  ).toHaveCount(1);
});
