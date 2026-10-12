/**
 * Part 2 (Training, brief 29A): the /training index and chapters, the
 * header's Inference / Training sections, the live in-browser trainer, and
 * frame tests: key frames of each Part 2 animation show the caption built
 * from scripts/training's state for that frame.
 */
import { expect, test, type Locator, type Page } from "@playwright/test";

import bpx from "../unit/fixtures/training/anim_backprop.json";
import bx from "../unit/fixtures/training/anim_bpe.json";
import hx from "../unit/fixtures/training/anim_hero.json";
import ox from "../unit/fixtures/training/anim_objective.json";
import qx from "../unit/fixtures/training/anim_optim2d.json";
import tx from "../unit/fixtures/training/train_run.json";

import { bpeCaption, type BpeState } from "@/lib/anim/training/bpe-steps";
import { heroCaption, type HeroState } from "@/lib/anim/training/hero-steps";
import {
  objectiveCaption,
  type ObjectiveState,
} from "@/lib/anim/training/objective-steps";
import {
  optim2dCaption,
  type Optim2dState,
} from "@/lib/anim/training/optim2d-steps";
import { ALPHABET } from "@/lib/transformer/tokenizer";

function errors(page: Page): string[] {
  const out: string[] = [];
  page.on("pageerror", (e) => out.push(`pageerror: ${e.message}`));
  page.on("console", (m) => {
    if (m.type() === "error") out.push(`console: ${m.text()}`);
  });
  return out;
}

const SHIPPED = [
  ["01-training-loop", "The training loop"],
  ["02-data-and-tokens", "Data and tokenisation"],
  ["03-objective", "The objective"],
  ["04-backprop", "Backprop through the model"],
  ["05-optimisers", "Optimisers"],
] as const;

test.describe("Part 2 pages", () => {
  test("the index lists 16 chapters; the shipped ones link", async ({
    page,
  }) => {
    await page.goto("/training");
    await expect(
      page.getByRole("heading", { name: "Training the decoder" }),
    ).toBeVisible();
    await expect(page.locator("main ol > li")).toHaveCount(16);
    for (const [slug, title] of SHIPPED) {
      await expect(
        page.getByRole("link", { name: title, exact: true }),
      ).toHaveAttribute("href", `/training/${slug}`);
    }
    await expect(page.getByText("coming soon")).toHaveCount(11);
    await expect(
      page.getByRole("link", { name: "Training over slow links" }),
    ).toHaveCount(0);
  });

  test("the header has an Inference and a Training section, marked by route", async ({
    page,
  }) => {
    await page.goto("/training/03-objective");
    const header = page.locator("header");
    await expect(
      header.getByRole("link", { name: "Training", exact: true }),
    ).toHaveAttribute("aria-current", "page");
    await header.getByRole("link", { name: "Inference", exact: true }).click();
    await expect(page).toHaveURL(/\/learn$/);
    await expect(
      header.getByRole("link", { name: "Inference", exact: true }),
    ).toHaveAttribute("aria-current", "page");
    await expect(
      header.getByRole("link", { name: "Training", exact: true }),
    ).not.toHaveAttribute("aria-current", "page");
  });

  test("chapters chain with previous / next links and a part label", async ({
    page,
  }) => {
    await page.goto("/training/02-data-and-tokens");
    await expect(
      page.getByText("Part 2 · Training · chapter 2 of 16"),
    ).toBeVisible();
    await expect(
      page.getByRole("link", { name: "← The training loop" }),
    ).toHaveAttribute("href", "/training/01-training-loop");
    await page.goto("/training/05-optimisers");
    // chapter 6 has not shipped: no next link
    await expect(
      page.getByRole("link", { name: /Learning rate, batch size/ }),
    ).toHaveCount(0);
    expect(
      (await page.goto("/training/06-learning-rate-and-schedules"))!.status(),
    ).toBe(404);
    expect((await page.goto("/training/not-a-chapter"))!.status()).toBe(404);
  });

  test("a Training chapter's comment list answers (section key training-<slug>)", async ({
    request,
  }) => {
    const ok = await request.get(
      "/api/sections/training-01-training-loop/comments",
    );
    expect(ok.status()).toBe(200);
    const bad = await request.get("/api/sections/training-99-nope/comments");
    expect(bad.status()).toBe(404);
  });

  test("the home and about pages frame both parts", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByRole("heading", { level: 1 })).toHaveText(
      "How a decoder-only transformer runs, and how it is trained.",
    );
    await expect(
      page.getByRole("link", { name: "Part 2: Training →" }),
    ).toHaveAttribute("href", "/training");
    await page.goto("/about");
    await expect(
      page.getByRole("link", { name: "Part 2: Training" }),
    ).toBeVisible();
  });
});

test.describe("the live trainer", () => {
  test("trains Part 1's model for 200 steps in a worker and matches the reference", async ({
    page,
  }) => {
    test.setTimeout(120_000);
    const errs = errors(page);
    await page.route("**/api/events", (r) =>
      r.fulfill({ json: { ok: true, data: { written: 0 } } }),
    );
    await page.goto("/training/01-training-loop");
    const t = page.getByTestId("live-trainer");
    await t.scrollIntoViewIfNeeded();
    await t.getByRole("combobox", { name: "Steps" }).selectOption("200");
    await t.getByTestId("train-start").click();
    await expect(t).toHaveAttribute("data-state", "done", { timeout: 90_000 });
    await expect(t).toHaveAttribute("data-steps-done", "200");
    // validation every 100 steps: 0, 100, 200
    await expect(t.getByTestId("val-dot")).toHaveCount(3);
    // the browser's validation loss at step 200 is the Python reference's
    const want = tx.evals[2]!.val.toFixed(3);
    await expect(t.getByTestId("train-status")).toContainText(
      `validation ${want}`,
    );
    await expect(t.getByTestId("samples")).toContainText("step 200:");
    const overflow = await page.evaluate(
      () =>
        document.scrollingElement!.scrollWidth -
        document.scrollingElement!.clientWidth,
    );
    expect(overflow).toBeLessThanOrEqual(0);
    expect(errs).toEqual([]);
  });

  test("pauses and resumes", async ({ page }) => {
    await page.goto("/training/01-training-loop");
    const t = page.getByTestId("live-trainer");
    await t.scrollIntoViewIfNeeded();
    await t.getByTestId("train-start").click();
    await expect
      .poll(async () => Number(await t.getAttribute("data-steps-done")))
      .toBeGreaterThan(20);
    await t.getByTestId("train-pause").click();
    await expect(t).toHaveAttribute("data-state", "paused");
    const n = Number(await t.getAttribute("data-steps-done"));
    await page.waitForTimeout(800);
    // at most the batch in flight arrives after the pause
    expect(Number(await t.getAttribute("data-steps-done"))).toBeLessThanOrEqual(
      n + 400,
    );
    await t.getByTestId("train-resume").click();
    await expect(t).toHaveAttribute("data-state", "running");
    await t.getByRole("button", { name: "Reset" }).click();
    await expect(t).toHaveAttribute("data-state", "idle");
  });
});

test.describe("frame tests: the page's frames are scripts/training's", () => {
  test.use({ contextOptions: { reducedMotion: "reduce" } });

  async function open(page: Page, path: string, id: string): Promise<Locator> {
    await page.goto(path);
    await expect(page.locator("[data-pending-widget]")).toHaveCount(0, {
      timeout: 30_000,
    });
    const fig = page.getByTestId(id);
    await expect(fig).toBeVisible();
    return fig;
  }
  async function show(fig: Locator, s: number): Promise<void> {
    await fig.getByTestId("scrub").fill(String(s));
    await expect(fig).toHaveAttribute("data-step", String(s));
  }
  function keyFrames(n: number): number[] {
    const out = new Set([0, 1, 2, n - 1]);
    for (let k = 1; k < 6; k++) out.add(Math.floor((k * (n - 1)) / 6));
    return [...out].sort((a, b) => a - b);
  }

  test("chapter 1 hero", async ({ page }) => {
    const fig = await open(page, "/training/01-training-loop", "training-hero");
    const want = hx.states as unknown as HeroState[];
    await expect(fig.getByTestId("scrub")).toHaveAttribute(
      "max",
      String(want.length - 1),
    );
    for (const k of keyFrames(want.length)) {
      await show(fig, k);
      await expect(fig.getByTestId("caption")).toHaveText(
        heroCaption(want[k]!),
      );
      const reached = await fig
        .locator("[data-testid^='group-'][data-reached='true']")
        .count();
      expect(reached).toBe(want[k]!.gradNorms.filter((g) => g !== null).length);
    }
  });

  test("chapter 2 BPE", async ({ page }) => {
    const fig = await open(
      page,
      "/training/02-data-and-tokens",
      "bpe-animation",
    );
    const want = bx.states as unknown as BpeState[];
    const vocab = (id: number) =>
      id < 64 ? ALPHABET[id]! : want[id - 64 + 1]!.token!;
    await expect(fig.getByTestId("scrub")).toHaveAttribute(
      "max",
      String(want.length - 1),
    );
    for (const k of keyFrames(want.length)) {
      await show(fig, k);
      await expect(fig.getByTestId("caption")).toHaveText(
        bpeCaption(want[k]!, vocab),
      );
    }
  });

  test("chapter 3 objective, trained and initial weights", async ({ page }) => {
    const fig = await open(
      page,
      "/training/03-objective",
      "objective-animation",
    );
    for (const kind of ["trained", "init"] as const) {
      if (kind === "init") {
        const before = (await fig.getAttribute("data-key")) ?? "";
        await fig
          .getByRole("combobox", { name: "Cross-entropy animation weights" })
          .selectOption("init");
        await expect(fig).not.toHaveAttribute("data-key", before);
      }
      const want = ox[kind] as unknown as ObjectiveState[];
      for (const k of keyFrames(want.length)) {
        await show(fig, k);
        await expect(fig.getByTestId("caption")).toHaveText(
          objectiveCaption(want[k]!),
        );
      }
    }
  });

  test("chapter 4 backprop: every stop, in order", async ({ page }) => {
    const fig = await open(page, "/training/04-backprop", "backprop-animation");
    const names = bpx.trained.record.map((r) => r.name);
    await expect(fig.getByTestId("scrub")).toHaveAttribute(
      "max",
      String(names.length),
    );
    for (const k of [0, 3, 7, names.length - 1, names.length]) {
      await show(fig, k);
      await expect(fig.getByTestId("caption")).toHaveText(
        new RegExp(`^Stop ${k + 1}/${names.length + 1}:`),
      );
    }
    await show(fig, names.length);
    await expect(
      fig.locator("[data-testid='op-emb'][data-current='true']"),
    ).toHaveCount(1);
  });

  test("chapter 5 optimisers", async ({ page }) => {
    const fig = await open(
      page,
      "/training/05-optimisers",
      "optim2d-animation",
    );
    const want = qx.states as unknown as Optim2dState[];
    for (const k of keyFrames(want.length)) {
      await show(fig, k);
      await expect(fig.getByTestId("caption")).toHaveText(
        optim2dCaption(want[k]!),
      );
    }
    await expect(page.getByTestId("optimizer-runs")).toBeVisible();
  });
});
