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
      ],
      thresholds: {
        // Per CLAUDE.md §8.
        "src/lib/transformer/**": {
          lines: 100,
          functions: 100,
          branches: 100,
          statements: 100,
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
