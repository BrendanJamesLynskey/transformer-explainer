/**
 * Frame tests on the page (visual standard §4): set key frames of every
 * animation with its scrub bar and require the caption on screen to be the
 * caption built from `scripts/reference.py`'s state for that frame
 * (tests/unit/fixtures/animations.json), and the drawing to match the
 * state where it can be read off the page.
 */
import { expect, test, type Locator } from "@playwright/test";

import fx from "../unit/fixtures/animations.json";

import { attentionCaption, type AttnState } from "@/lib/anim/attention-steps";
import { embedCaption, type EmbedState } from "@/lib/anim/embed-steps";
import { ffnCaption, type FfnState } from "@/lib/anim/ffn-steps";
import { overviewCaption, type OverviewState } from "@/lib/anim/overview-steps";

// No autoplay (reduced motion): the test sets each frame itself.
test.use({ contextOptions: { reducedMotion: "reduce" } });

async function open(
  page: import("@playwright/test").Page,
  path: string,
  id: string,
) {
  await page.goto(path);
  await expect(page.locator("[data-pending-widget]")).toHaveCount(0);
  const fig = page.getByTestId(id);
  await expect(fig).toBeVisible({ timeout: 30_000 });
  return fig;
}

async function show(fig: Locator, s: number): Promise<void> {
  await fig.getByTestId("scrub").fill(String(s));
  await expect(fig).toHaveAttribute("data-step", String(s));
  await expect(fig).toHaveAttribute("data-playing", "false");
}

/** A handful of frames spread over the animation, plus first and last. */
function keyFrames(n: number): number[] {
  const out = new Set([0, 1, 2, n - 1]);
  for (let k = 1; k < 6; k++) out.add(Math.floor((k * (n - 1)) / 6));
  return [...out].sort((a, b) => a - b);
}

const TOKENS = ["h", "e", "l", "l", "o", "!", "a", "a"];

test("embeddings: the page's frames are the reference's", async ({ page }) => {
  const fig = await open(page, "/learn/02-embeddings", "embedding-animation");
  const want = fx.embed.states as unknown as EmbedState[];
  await expect(fig.getByTestId("scrub")).toHaveAttribute(
    "max",
    String(want.length - 1),
  );
  for (const k of keyFrames(want.length)) {
    await show(fig, k);
    await expect(fig.getByTestId("caption")).toHaveText(embedCaption(want[k]!));
    // the rows of X drawn as done are the reference's
    const done = await fig
      .locator("[data-testid^='x-row-'][data-done]")
      .count();
    expect(done).toBe(want[k]!.done.length);
  }
});

test("attention: the page's frames are the reference's, one head and all heads", async ({
  page,
}) => {
  const fig = await open(page, "/learn/03-attention", "attention-animation");
  for (const run of fx.attention) {
    const before = (await fig.getAttribute("data-key")) ?? "";
    await fig
      .getByRole("combobox", { name: "Attention head" })
      .selectOption(String(run.head));
    await fig
      .getByRole("combobox", { name: "First query row" })
      .selectOption(String(run.row));
    if (run.head !== 0 || run.row !== 3)
      await expect(fig).not.toHaveAttribute("data-key", before);
    const want = run.states as unknown as AttnState[];
    await expect(fig.getByTestId("scrub")).toHaveAttribute(
      "max",
      String(want.length - 1),
    );
    for (const k of keyFrames(want.length)) {
      await show(fig, k);
      const s = want[k]!;
      await expect(fig.getByTestId("caption")).toHaveText(
        attentionCaption(s, TOKENS),
      );
      if (s.phase !== "concat" && s.phase !== "project") {
        // the masked cells drawn in the current row are the reference's
        const masked = await fig
          .locator("[data-testid^='cell-'][data-masked]")
          .count();
        const want =
          s.scores && s.phase !== "scale"
            ? s.scores.filter((v) => v === null).length
            : 0;
        expect(masked).toBe(want);
      }
    }
  }
});

test("feed-forward: the page's frames are the reference's", async ({
  page,
}) => {
  const fig = await open(page, "/learn/04-ffn", "ffn-animation");
  const want = fx.ffn.states as unknown as FfnState[];
  for (const k of keyFrames(want.length)) {
    await show(fig, k);
    await expect(fig.getByTestId("caption")).toHaveText(ffnCaption(want[k]!));
  }
  // the neurons marked as fired are the reference's
  const fired = await fig
    .locator("[data-fired='true']")
    .evaluateAll((els) =>
      els.map((e) => Number(e.getAttribute("data-testid")!.split("-")[1])),
    );
  expect(fired).toEqual(want[want.length - 1]!.fired);
});

test("overview hero: the page's frames are the reference's", async ({
  page,
}) => {
  const fig = await open(page, "/learn/01-overview", "overview-hero");
  const want = fx.overview.states as unknown as OverviewState[];
  await expect(fig.getByTestId("scrub")).toHaveAttribute(
    "max",
    String(want.length - 1),
  );
  for (const k of keyFrames(want.length)) {
    await show(fig, k);
    await expect(fig.getByTestId("caption")).toHaveText(
      overviewCaption(want[k]!),
    );
  }
  // after the last append the context holds the prompt plus three samples
  await show(fig, want.length - 1);
  const last = want[want.length - 1]!;
  expect(last.context).toHaveLength(8);
  await expect(fig.getByTestId("ctx-7").locator("text")).toHaveCount(1);
});
