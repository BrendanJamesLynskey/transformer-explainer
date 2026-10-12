/**
 * Records the README's animated GIFs (and WebM videos) of the animations
 * (copied from Systolic Arrays Explained). Manual run, output committed; a heavy job, so run it alone:
 *
 *   pnpm build && pnpm start   # in another shell
 *   pnpm animations            # writes docs/media/*.gif and *.webm
 *
 * Every frame is a model state set by the animation's scrub bar (reduced
 * motion, so nothing plays by itself), screenshotted, then joined by
 * ffmpeg: the recordings are reproducible frame for frame. Needs ffmpeg on
 * the PATH.
 */
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

import { chromium } from "@playwright/test";

const OUT = path.join(process.cwd(), "docs", "media");
const BASE = process.env.SCREENSHOT_BASE_URL ?? "http://localhost:3000";

type Clip = {
  name: string;
  path: string;
  widget: string;
  /** frames per second of the output */
  fps: number;
  /** a radio to press first (a parameter), if any */
  radio?: string;
  /** at most this many frames, evenly spaced over the steps (long animations) */
  maxFrames?: number;
};

const CLIPS: Clip[] = [
  {
    name: "overview-hero",
    path: "/learn/01-overview",
    widget: "overview-hero",
    fps: 2,
  },
  {
    name: "embeddings",
    path: "/learn/02-embeddings",
    widget: "embedding-animation",
    fps: 3,
  },
  {
    name: "attention",
    path: "/learn/03-attention",
    widget: "attention-animation",
    fps: 5,
  },
  { name: "ffn", path: "/learn/04-ffn", widget: "ffn-animation", fps: 4 },
  // brief 27B
  {
    name: "layernorm",
    path: "/learn/05-layernorm-residuals",
    widget: "layernorm-animation",
    fps: 2,
  },
  {
    name: "stacking",
    path: "/learn/06-stacking",
    widget: "stacking-animation",
    fps: 2,
  },
  {
    name: "generation",
    path: "/learn/07-sampling",
    widget: "generation-animation",
    fps: 2,
  },
  // Part 2, Training (briefs 29A-29C)
  ...(
    [
      ["training-hero", "/training/01-training-loop", "training-hero", 3],
      ["bpe", "/training/02-data-and-tokens", "bpe-animation", 3],
      ["objective", "/training/03-objective", "objective-animation", 2],
      ["backprop", "/training/04-backprop", "backprop-animation", 2],
      ["optimisers", "/training/05-optimisers", "optim2d-animation", 6],
      [
        "schedules",
        "/training/06-learning-rate-and-schedules",
        "schedule-animation",
        6,
      ],
      ["depth", "/training/07-init-norm-mup", "depth-animation", 2],
      [
        "attention-growth",
        "/training/08-stability",
        "attention-growth-animation",
        2,
      ],
      ["rounding", "/training/09-mixed-precision", "rounding-animation", 2],
      ["memory", "/training/10-memory", "memory-animation", 4],
      ["ring-all-reduce", "/training/11-data-parallelism", "ring-animation", 1],
      ["pipeline", "/training/12-model-parallelism", "pipeline-animation", 4],
      ["demo-sync", "/training/13-slow-links", "demo-animation", 1],
      ["isoflop", "/training/14-scaling-laws", "isoflop-animation", 3],
      ["goodput", "/training/15-running-a-real-job", "goodput-animation", 8],
      ["sft", "/training/16-after-pretraining", "sft-animation", 2],
    ] as const
  ).map(
    ([name, p, widget, fps]): Clip => ({
      name,
      path: p,
      widget,
      fps,
      maxFrames: 60,
    }),
  ),
];

// `pnpm animations attention ffn` records only the named clips
const ONLY = process.argv.slice(2);

async function main(): Promise<void> {
  mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1280, height: 900 },
    colorScheme: "light",
    reducedMotion: "reduce",
  });
  const page = await context.newPage();
  for (const c of CLIPS) {
    if (ONLY.length > 0 && !ONLY.includes(c.name)) continue;
    const dir = mkdtempSync(path.join(tmpdir(), `te-${c.name}-`));
    await page.goto(BASE + c.path, { waitUntil: "networkidle" });
    const fig = page.getByTestId(c.widget);
    await fig.waitFor();
    if (c.radio) await fig.getByRole("radio", { name: c.radio }).click();
    const scrub = fig.getByTestId("scrub");
    const max = Number(await scrub.getAttribute("max"));
    const visual = fig.getByTestId("visual");
    const n = Math.min(max + 1, c.maxFrames ?? max + 1);
    const steps = Array.from({ length: n }, (_, i) =>
      n === 1 ? 0 : Math.round((i * max) / (n - 1)),
    );
    for (let i = 0; i < steps.length; i++) {
      await scrub.fill(String(steps[i]));
      await visual.screenshot({
        path: path.join(dir, `f${String(i).padStart(4, "0")}.png`),
      });
    }
    const input = path.join(dir, "f%04d.png");
    const gif = path.join(OUT, `${c.name}.gif`);
    const webm = path.join(OUT, `${c.name}.webm`);
    // even dimensions for the video encoder; a shared palette for the GIF
    const scale = "scale=560:-2:flags=lanczos";
    execFileSync("ffmpeg", [
      "-y",
      "-loglevel",
      "error",
      "-framerate",
      String(c.fps),
      "-i",
      input,
      "-vf",
      `${scale},split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=none`,
      "-loop",
      "0",
      gif,
    ]);
    execFileSync("ffmpeg", [
      "-y",
      "-loglevel",
      "error",
      "-framerate",
      String(c.fps),
      "-i",
      input,
      "-vf",
      scale,
      "-c:v",
      "libvpx-vp9",
      "-b:v",
      "0",
      "-crf",
      "40",
      "-pix_fmt",
      "yuv420p",
      webm,
    ]);
    rmSync(dir, { recursive: true, force: true });
    console.log(
      `wrote ${path.relative(process.cwd(), gif)} and .webm (${steps.length} frames)`,
    );
  }
  await browser.close();
}

void main();
