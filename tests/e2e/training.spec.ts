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
import ax from "../unit/fixtures/training/anim_attention.json";
import dx from "../unit/fixtures/training/anim_depth.json";
import mx from "../unit/fixtures/training/anim_memory.json";
import rx from "../unit/fixtures/training/anim_rounding.json";
import sx from "../unit/fixtures/training/anim_schedules.json";

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
import {
  attentionGrowthCaption,
  type AttentionGrowthState,
} from "@/lib/anim/training/attention-growth-steps";
import { depthCaption, type DepthState } from "@/lib/anim/training/depth-steps";
import { memoryCaption } from "@/lib/anim/training/memory-steps";
import {
  roundingCaption,
  type RoundingState,
} from "@/lib/anim/training/rounding-steps";
import {
  scheduleCaption,
  type ScheduleState,
} from "@/lib/anim/training/schedule-steps";
import { LLAMA_7B, type MemEvent } from "@/lib/training/memory";
import { demoCaption, demoStates } from "@/lib/anim/training/demo-steps";
import {
  goodputCaption,
  goodputStates,
} from "@/lib/anim/training/goodput-steps";
import {
  isoflopCaption,
  type IsoRow,
  type IsoState,
} from "@/lib/anim/training/isoflop-steps";
import {
  pipelineCaption,
  pipelineStates,
} from "@/lib/anim/training/pipeline-steps";
import { ringCaption, type RingState } from "@/lib/anim/training/ring-steps";
import { sftCaption, sftStates } from "@/lib/anim/training/sft-steps";
import type { DemoRecord } from "@/lib/training/slowlinks";
import type { PipeResult, Segment } from "@/lib/training/systems";
import demox from "../unit/fixtures/training/anim_demo.json";
import goodx from "../unit/fixtures/training/anim_goodput.json";
import isox from "../unit/fixtures/training/anim_isoflop.json";
import pipex from "../unit/fixtures/training/anim_pipeline.json";
import ringx from "../unit/fixtures/training/anim_ring.json";
import { ALPHABET } from "@/lib/transformer/tokenizer";

import results from "@/data/training/results.json";

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
  ["06-learning-rate-and-schedules", "Learning rate, batch size and schedules"],
  ["07-init-norm-mup", "Initialisation, normalisation and µP"],
  ["08-stability", "Stability"],
  ["09-mixed-precision", "Mixed precision"],
  ["10-memory", "Memory"],
  ["11-data-parallelism", "Parallelism I: data"],
  ["12-model-parallelism", "Parallelism II: model"],
  ["13-slow-links", "Training over slow links"],
  ["14-scaling-laws", "Scaling laws"],
  ["15-running-a-real-job", "Running a real job"],
  ["16-after-pretraining", "After pretraining"],
] as const;

test.describe("Training pages", () => {
  test("the index lists 16 chapters, every one linked", async ({ page }) => {
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
    await expect(page.getByText("coming soon")).toHaveCount(0);
  });

  test("the header names the site by route, with that site's own nav", async ({
    page,
  }) => {
    await page.goto("/training/03-objective");
    const header = page.locator("header");
    await expect(header.locator("[data-site-brand]")).toHaveText(
      /^Foundations\W+Training$/,
    );
    await expect(header.locator("a[data-site-chapters]")).toHaveAttribute(
      "href",
      "/training",
    );
    await expect(header.getByRole("link", { name: "Playground" })).toHaveCount(
      0,
    );
    await expect(header.getByText(/Part [12]/)).toHaveCount(0);
    // the chapter index leads back to the decoder
    await page.goto("/training");
    await page.getByRole("link", { name: "← The decoder" }).click();
    await expect(page).toHaveURL(/\/learn$/);
    await expect(header.locator("[data-site-brand]")).toHaveText(
      /^Foundations\W+Decoder$/,
    );
    await expect(header.locator("a[data-site-chapters]")).toHaveAttribute(
      "href",
      "/learn",
    );
    await expect(
      header.getByRole("link", { name: "Playground" }),
    ).toBeVisible();
    // and the decoder's index (and its last chapter) go on to Training
    await expect(
      page.getByRole("link", { name: "Next: Training →" }),
    ).toHaveAttribute("href", "/training");
    await page.goto("/learn/07-sampling");
    await expect(
      page
        .locator("article nav")
        .getByRole("link", { name: "Next: Training →" }),
    ).toHaveAttribute("href", "/training");
  });

  test("chapters chain with previous / next links and a chapter label", async ({
    page,
  }) => {
    await page.goto("/training/02-data-and-tokens");
    await expect(
      page.getByText("Chapter 2 of 16", { exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole("link", { name: "← The training loop" }),
    ).toHaveAttribute("href", "/training/01-training-loop");
    await page.goto("/training/05-optimisers");
    await expect(
      page.getByRole("link", { name: /Learning rate, batch size/ }),
    ).toHaveAttribute("href", "/training/06-learning-rate-and-schedules");
    await page.goto("/training/10-memory");
    await expect(
      page.getByRole("link", { name: "Parallelism I: data →" }),
    ).toHaveAttribute("href", "/training/11-data-parallelism");
    await page.goto("/training/16-after-pretraining");
    await expect(
      page.getByText("Chapter 16 of 16", { exact: true }),
    ).toBeVisible();
    // the last chapter has no next link
    await expect(page.locator("article nav a")).toHaveCount(1);
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

  test("the home and about pages frame both Foundations sites, by name", async ({
    page,
  }) => {
    await page.goto("/");
    await expect(page.getByRole("heading", { level: 1 })).toHaveText(
      "How a decoder-only transformer runs, and how it is trained.",
    );
    await expect(
      page.getByRole("link", { name: "Next: Training →" }),
    ).toHaveAttribute("href", "/training");
    await expect(page.locator("main").getByText(/Part [12]/)).toHaveCount(0);
    await page.goto("/about");
    await expect(
      page.locator("main").getByRole("link", { name: "Training", exact: true }),
    ).toBeVisible();
    await expect(page.locator("main").getByText(/Part [12]/)).toHaveCount(0);
  });
});

test.describe("the live trainer", () => {
  test("trains the Decoder's model for 200 steps in a worker and matches the reference", async ({
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
  test("chapter 6 schedules", async ({ page }) => {
    const fig = await open(
      page,
      "/training/06-learning-rate-and-schedules",
      "schedule-animation",
    );
    const want = sx.states as unknown as ScheduleState[];
    const names = results.schedules.rows.map((r) => r.name);
    await expect(fig.getByTestId("scrub")).toHaveAttribute(
      "max",
      String(want.length - 1),
    );
    for (const k of keyFrames(want.length)) {
      await show(fig, k);
      await expect(fig.getByTestId("caption")).toHaveText(
        scheduleCaption(want[k]!, names),
      );
    }
    for (const id of ["lr-sweep-chart", "batch-size-chart"])
      await expect(page.getByTestId(id)).toBeVisible();
  });

  test("chapter 7 depth", async ({ page }) => {
    const fig = await open(
      page,
      "/training/07-init-norm-mup",
      "depth-animation",
    );
    for (const std of ["0.25", "0.02"]) {
      await fig
        .getByRole("combobox", { name: /Initial standard deviation/ })
        .selectOption(std);
      const want = (dx.stds as Record<string, unknown[]>)[std] as DepthState[];
      for (const k of keyFrames(want.length)) {
        await show(fig, k);
        await expect(fig.getByTestId("caption")).toHaveText(
          depthCaption(want[k]!),
        );
      }
    }
    for (const id of ["pre-post-chart", "coord-check-chart", "mup-chart"])
      await expect(page.getByTestId(id)).toBeVisible();
  });

  test("chapter 8 attention growth", async ({ page }) => {
    const fig = await open(
      page,
      "/training/08-stability",
      "attention-growth-animation",
    );
    const want = ax.states as unknown as AttentionGrowthState[];
    for (const k of keyFrames(want.length)) {
      await show(fig, k);
      await expect(fig.getByTestId("caption")).toHaveText(
        attentionGrowthCaption(want[k]!),
      );
    }
    await expect(page.getByTestId("bad-batch-chart")).toBeVisible();
  });

  test("chapter 9 rounding", async ({ page }) => {
    const fig = await open(
      page,
      "/training/09-mixed-precision",
      "rounding-animation",
    );
    const want = rx.states as unknown as RoundingState[];
    for (const k of keyFrames(want.length)) {
      await show(fig, k);
      await expect(fig.getByTestId("caption")).toHaveText(
        roundingCaption(want[k]!),
      );
    }
    await expect(page.getByTestId("precision-chart")).toBeVisible();
  });

  test("chapter 10 memory", async ({ page }) => {
    const fig = await open(page, "/training/10-memory", "memory-animation");
    const c = mx.cases.find(
      (x) => x.arch.name === "LLaMA 7B" && x.recompute === "none",
    )!;
    const want = c.states as unknown as MemEvent[];
    for (const k of keyFrames(want.length)) {
      await show(fig, k);
      await expect(fig.getByTestId("caption")).toHaveText(
        memoryCaption(want[k]!, LLAMA_7B),
      );
    }
    await fig
      .getByRole("combobox", { name: "Activation recomputation" })
      .selectOption("full");
    const full = mx.cases.find(
      (x) => x.arch.name === "LLaMA 7B" && x.recompute === "full",
    )!.states as unknown as MemEvent[];
    await expect(fig.getByTestId("scrub")).toHaveAttribute(
      "max",
      String(full.length - 1),
    );
  });
  // brief 29C
  test("chapter 11 ring all-reduce", async ({ page }) => {
    const fig = await open(
      page,
      "/training/11-data-parallelism",
      "ring-animation",
    );
    const want = ringx.states as unknown as RingState[];
    await expect(fig.getByTestId("scrub")).toHaveAttribute(
      "max",
      String(want.length - 1),
    );
    for (const k of keyFrames(want.length)) {
      await show(fig, k);
      await expect(fig.getByTestId("caption")).toHaveText(
        ringCaption(want[k]!, 4, 16),
      );
    }
    await expect(page.getByTestId("dp-calculator")).toBeVisible();
  });

  test("chapter 12 pipeline", async ({ page }) => {
    const fig = await open(
      page,
      "/training/12-model-parallelism",
      "pipeline-animation",
    );
    const r = (pipex as Record<string, unknown>)["1f1b"] as PipeResult;
    const want = pipelineStates(r);
    await expect(fig.getByTestId("scrub")).toHaveAttribute(
      "max",
      String(want.length - 1),
    );
    for (const k of keyFrames(want.length)) {
      await show(fig, k);
      await expect(fig.getByTestId("caption")).toHaveText(
        pipelineCaption(want[k]!, r),
      );
    }
    for (const id of ["layout-explorer", "validation-chart"])
      await expect(page.getByTestId(id)).toBeVisible();
  });

  test("chapter 13 DeMo sync", async ({ page }) => {
    const fig = await open(page, "/training/13-slow-links", "demo-animation");
    const recs = demox.states as unknown as DemoRecord[];
    const want = demoStates(recs);
    for (const k of keyFrames(want.length)) {
      await show(fig, k);
      await expect(fig.getByTestId("caption")).toHaveText(
        demoCaption(want[k]!, recs),
      );
    }
    for (const id of ["slow-links-charts", "wan-widget"])
      await expect(page.getByTestId(id)).toBeVisible();
  });

  test("chapter 14 IsoFLOP", async ({ page }) => {
    const fig = await open(
      page,
      "/training/14-scaling-laws",
      "isoflop-animation",
    );
    const want = isox.states as unknown as IsoState[];
    const rows = results.isoflop.rows as unknown as IsoRow[];
    for (const k of keyFrames(want.length)) {
      await show(fig, k);
      await expect(fig.getByTestId("caption")).toHaveText(
        isoflopCaption(want[k]!, rows),
      );
    }
  });

  test("chapter 15 goodput", async ({ page }) => {
    const fig = await open(
      page,
      "/training/15-running-a-real-job",
      "goodput-animation",
    );
    const want = goodputStates(
      goodx.sim.segments as Segment[],
      goodx.params.horizon,
    );
    await expect(fig.getByTestId("scrub")).toHaveAttribute(
      "max",
      String(want.length - 1),
    );
    for (const k of keyFrames(want.length)) {
      await show(fig, k);
      await expect(fig.getByTestId("caption")).toHaveText(
        goodputCaption(want[k]!),
      );
    }
    for (const id of ["monitor-charts", "resume-chart", "goodput-chart"])
      await expect(page.getByTestId(id)).toBeVisible();
  });

  test("chapter 16 SFT", async ({ page }) => {
    const fig = await open(
      page,
      "/training/16-after-pretraining",
      "sft-animation",
    );
    const x = results.sft;
    const d = {
      promptTokens: x.promptTokens,
      responseTokens: x.responseTokens,
      chosenTokens: x.pref.chosenTokens,
      rejectedTokens: x.pref.rejectedTokens,
    };
    const want = sftStates(d);
    for (const k of keyFrames(want.length)) {
      await show(fig, k);
      await expect(fig.getByTestId("caption")).toHaveText(
        sftCaption(want[k]!, d),
      );
    }
  });
});
