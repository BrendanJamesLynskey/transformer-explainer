/**
 * Frame tests on the page for the brief 27B animations (visual standard
 * §4): set key frames with the scrub bar and require the caption on screen
 * to be the caption built from `scripts/reference.py`'s state for that
 * frame (tests/unit/fixtures/animations_b.json), and the drawing to match
 * the state where it can be read off the page.
 */
import { expect, test, type Locator, type Page } from "@playwright/test";

import fx from "../unit/fixtures/animations_b.json";

import { genCaption, type GenState } from "@/lib/anim/gen-steps";
import { lnCaption, type LnState } from "@/lib/anim/ln-steps";
import { pct, showChar } from "@/lib/anim/format";
import type { TopToken } from "@/lib/anim/overview-steps";
import { stackCaption, type StackState } from "@/lib/anim/stack-steps";
import { ALPHABET } from "@/lib/transformer/tokenizer";

// No autoplay (reduced motion): the test sets each frame itself.
test.use({ contextOptions: { reducedMotion: "reduce" } });
// many frames and parameter changes per test
test.describe.configure({ timeout: 120_000 });

const TOKENS = ["h", "e", "l", "l", "o", "!", "a", "a"];

async function open(page: Page, path: string, id: string): Promise<Locator> {
  await page.route("**/api/events", (r) =>
    r.fulfill({ json: { ok: true, data: { written: 0 } } }),
  );
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

/** Change a parameter and wait until the animation has restarted for it. */
async function setParam(fig: Locator, change: () => Promise<unknown>) {
  const before = (await fig.getAttribute("data-key")) ?? "";
  await change();
  await expect(fig).not.toHaveAttribute("data-key", before);
}

test("layernorm: the page's frames are the reference's", async ({ page }) => {
  const fig = await open(
    page,
    "/learn/05-layernorm-residuals",
    "layernorm-animation",
  );
  for (const [r, run] of fx.layernorm.entries()) {
    if (r > 0) {
      await setParam(fig, () =>
        fig
          .getByRole("combobox", { name: "LayerNorm position" })
          .selectOption(String(run.pos)),
      );
      await setParam(fig, () =>
        fig
          .getByRole("combobox", { name: "LayerNorm gain" })
          .selectOption(String(run.gamma)),
      );
      await setParam(fig, () =>
        fig
          .getByRole("combobox", { name: "LayerNorm bias" })
          .selectOption(String(run.beta)),
      );
    }
    const want = run.states as unknown as LnState[];
    await expect(fig.getByTestId("scrub")).toHaveAttribute(
      "max",
      String(want.length - 1),
    );
    for (const k of keyFrames(want.length)) {
      await show(fig, k);
      const s = want[k]!;
      await expect(fig.getByTestId("caption")).toHaveText(lnCaption(s, TOKENS));
      // the strips drawn are the ones the state has
      await expect(fig.getByTestId("ln-out")).toHaveCount(s.ln ? 1 : 0);
      await expect(fig.getByTestId("ln-delta")).toHaveCount(s.delta ? 1 : 0);
      await expect(fig.getByTestId("ln-sum")).toHaveCount(s.out ? 1 : 0);
      // the mean line appears with μ and goes once LN(x) is formed
      const meanShown = ["mean", "var", "centre", "scale", "affine"].includes(
        s.phase,
      );
      await expect(fig.getByTestId("ln-mean")).toHaveCount(meanShown ? 1 : 0);
    }
  }
});

const top3 = (top: TopToken[]) =>
  top
    .slice(0, 3)
    .map((t) => `'${showChar(ALPHABET[t.id] ?? "?")}' ${pct(t.prob)}`)
    .join("  ");

test("stacking: the page's frames are the reference's", async ({ page }) => {
  const fig = await open(page, "/learn/06-stacking", "stacking-animation");
  for (const [r, run] of fx.stacking.entries()) {
    if (r > 0) {
      await setParam(fig, () =>
        fig
          .getByRole("combobox", { name: "Number of blocks" })
          .selectOption(String(run.nBlocks)),
      );
      await setParam(fig, () =>
        fig
          .getByRole("combobox", { name: "Stacking position" })
          .selectOption(String(run.pos)),
      );
    }
    const want = run.states as unknown as StackState[];
    await expect(fig.getByTestId("scrub")).toHaveAttribute(
      "max",
      String(want.length - 1),
    );
    for (const k of keyFrames(want.length)) {
      await show(fig, k);
      const s = want[k]!;
      await expect(fig.getByTestId("caption")).toHaveText(
        stackCaption(s, TOKENS),
      );
      // one row per finished layer, plus the block in progress
      await expect(fig.locator("[data-testid^='layer-']")).toHaveCount(
        s.rows.length + (s.kind === "attn" ? 1 : 0),
      );
      // the logit lens printed for each finished layer is the reference's
      for (const row of s.rows)
        await expect(fig.getByTestId(`lens-${row.layer}`)).toHaveText(
          top3(row.lens),
        );
    }
  }
});

test("generation loop: the page's frames are the reference's", async ({
  page,
}) => {
  const fig = await open(page, "/learn/07-sampling", "generation-animation");
  for (const [r, run] of fx.generation.entries()) {
    if (r > 0) {
      const o = run.opts;
      if (run.text !== "hello")
        await setParam(fig, () =>
          fig
            .getByRole("textbox", { name: "Generation prompt" })
            .fill(run.text),
        );
      if (run.seed !== 42)
        await setParam(fig, () =>
          fig
            .getByRole("spinbutton", { name: "Generation seed" })
            .fill(String(run.seed)),
        );
      await setParam(fig, () =>
        fig
          .getByRole("combobox", { name: "Sampling rule" })
          .selectOption(o.mode),
      );
      // τ and p re-run the model without restarting: wait on the caption
      await fig
        .getByRole("slider", { name: "Temperature" })
        .fill(String(o.temperature));
      if (o.mode === "top-p")
        await fig
          .getByRole("combobox", { name: "Top-p" })
          .selectOption(String(o.p));
    }
    const want = run.states as unknown as GenState[];
    await expect(fig.getByTestId("scrub")).toHaveAttribute(
      "max",
      String(want.length - 1),
    );
    const frames = new Set(keyFrames(want.length));
    // every phase of the first round, too
    for (let k = 0; k < Math.min(7, want.length); k++) frames.add(k);
    for (const k of [...frames].sort((a, b) => a - b)) {
      await show(fig, k);
      const s = want[k]!;
      await expect(fig.getByTestId("caption")).toHaveText(genCaption(s));
      // the dropped (hatched) bars are the ones the mask dropped
      if (s.kept && (s.phase === "truncate" || s.phase === "draw"))
        await expect(fig.locator("[data-dropped]")).toHaveCount(
          64 - s.kept.length,
        );
      // the KV cache holds one column per cached token in every grid
      await expect(fig.locator("[data-testid^='kv-0-K-col-']")).toHaveCount(
        s.cache[0]!.K.length,
      );
    }
  }
});

test("playground: step through generation runs the same loop", async ({
  page,
}) => {
  const fig = await open(page, "/playground", "playground-generation");
  const want = fx.generation[0]!.states as unknown as GenState[];
  await expect(fig.getByTestId("scrub")).toHaveAttribute(
    "max",
    String(want.length - 1),
  );
  for (const k of keyFrames(want.length)) {
    await show(fig, k);
    await expect(fig.getByTestId("caption")).toHaveText(genCaption(want[k]!));
  }
  // the presets drive it: "Recent-bias" is "the cat", seed 7
  await setParam(fig, () =>
    page.getByRole("button", { name: "Recent-bias" }).click(),
  );
  await expect(
    fig.getByRole("textbox", { name: "Generation prompt" }),
  ).toHaveValue("the cat");
});
