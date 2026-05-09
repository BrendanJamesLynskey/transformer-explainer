import { describe, expect, it, vi } from "vitest";

import { runOrFallback } from "@/lib/db-fallback";

describe("runOrFallback", () => {
  it("returns the resolved value when fn succeeds", async () => {
    const result = await runOrFallback("ok", async () => 42, 0);
    expect(result).toBe(42);
  });

  it("returns the fallback when fn throws", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const result = await runOrFallback(
      `throw-${Date.now()}-${Math.random()}`,
      async () => {
        throw new Error("boom");
      },
      "fallback",
    );
    expect(result).toBe("fallback");
    expect(warn).toHaveBeenCalledOnce();
    warn.mockRestore();
  });

  it("throttles repeated warnings for the same key", async () => {
    const key = `throttle-${Date.now()}-${Math.random()}`;
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    for (let i = 0; i < 5; i++) {
      await runOrFallback(
        key,
        async () => {
          throw new Error("boom");
        },
        null,
      );
    }
    expect(warn).toHaveBeenCalledOnce();
    warn.mockRestore();
  });
});
