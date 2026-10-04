import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 30_000,
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL ?? "http://localhost:3000",
    trace: "on-first-retry",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: process.env.CI
    ? {
        // `--no-experimental-require-module` makes Node refuse to require()
        // an ES module, as Vercel's function runtime does. Plain Node 20.19+
        // / 22.12+ allows it, which is how a server dependency that crashed
        // every comment list in production (jsdom → ESM-only @exodus/bytes,
        // 2026-10-04) passed this suite. With the flag, such a dependency
        // fails here instead.
        command:
          "pnpm build && node --no-experimental-require-module node_modules/next/dist/bin/next start -p 3000",
        url: "http://localhost:3000",
        reuseExistingServer: false,
        timeout: 120_000,
      }
    : {
        command: "pnpm dev",
        url: "http://localhost:3000",
        reuseExistingServer: true,
        timeout: 120_000,
      },
});
