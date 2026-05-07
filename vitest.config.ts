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
        "src/lib/**/index.ts",
        "src/lib/**/*.test.ts",
        "src/lib/db/migrations/**",
        // Type-only modules: no runtime exports → nothing for v8 to cover.
        "src/lib/transformer/types.ts",
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
