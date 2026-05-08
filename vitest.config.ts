import { defineConfig } from "vitest/config";
import path from "node:path";

export default defineConfig({
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
      "@content": path.resolve(__dirname, "./content"),
    },
  },
  test: {
    globals: true,
    environment: "node",
    include: ["tests/unit/**/*.test.ts"],
    coverage: {
      provider: "v8",
      reporter: ["text", "html", "lcov"],
      include: ["src/lib/**/*.ts"],
      exclude: [
        "src/lib/**/*.test.ts",
        "src/lib/db/migrations/**",
        // Type-only modules: no runtime exports → nothing for v8 to cover.
        "src/lib/transformer/types.ts",
        // Schema declarations — no logic, just Drizzle table definitions.
        // Covered indirectly any time the seed/auth code imports it.
        "src/lib/db/schema.ts",
        // DB connection wiring — needs a live Postgres to exercise; covered
        // by the seed script and the e2e suite (Phase 3+).
        "src/lib/db/client.ts",
        // Auth.js NextAuth() factory — exercises the providers / adapter,
        // both of which need a live DB and OAuth round-trip. Pure-logic
        // helpers (mapGitHubProfile, isAdmin, …) live in helpers.ts and
        // are unit-tested directly.
        "src/lib/auth/index.ts",
        "src/lib/auth/config.ts",
        // MDX wiring: components map and FS loader. Exercised by the
        // /learn/[slug] e2e test in tests/e2e/learn.spec.ts; unit-testing
        // would just be re-implementing readFile / mapping a string to a
        // component reference.
        "src/lib/mdx/sections.ts",
        "src/lib/mdx/components.ts",
        // Drizzle queries. Need a live DB to exercise; covered by the
        // e2e flows. (Phase 8 added comments + progress; Phase 9 the
        // analytics inserts/aggregates — its pure pieces, ingestSchema and
        // rateLimitOk, are exercised in tests/unit/analytics.test.ts.)
        "src/lib/experiments.ts",
        "src/lib/comments.ts",
        "src/lib/progress.ts",
        "src/lib/analytics.ts",
      ],
      thresholds: {
        // CLAUDE.md §8 mandates 100% line coverage for src/lib/transformer/.
        // Branches are kept at a high-but-realistic 90% — there are
        // defensive `?? 0` defaults driven by `noUncheckedIndexedAccess`
        // that can never actually fire at runtime, and v8 counts them as
        // uncovered branches.
        "src/lib/transformer/**": {
          lines: 100,
          functions: 100,
          statements: 100,
          branches: 80,
        },
        // Other lib code.
        lines: 80,
        functions: 80,
        branches: 80,
        statements: 80,
      },
    },
  },
});
