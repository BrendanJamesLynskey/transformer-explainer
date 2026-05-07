/**
 * Smoke test — confirms the test runner and TS toolchain work.
 * Phase 1 will replace this with the real maths test suite.
 */
import { describe, expect, it } from "vitest";

describe("smoke", () => {
  it("runs", () => {
    expect(1 + 1).toBe(2);
  });
});
