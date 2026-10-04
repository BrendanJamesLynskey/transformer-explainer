import { describe, expect, it, vi } from "vitest";

import {
  _resetFallbackStatsForTest,
  fallbackStats,
  runOrFallback,
} from "@/lib/db-fallback";

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

describe("fallbackStats", () => {
  it("counts every fallback per key, even when the warning is throttled", async () => {
    _resetFallbackStatsForTest();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const fail = async (): Promise<number> => {
      throw new Error("db down");
    };
    await runOrFallback("stats:a", fail, 0);
    await runOrFallback("stats:a", fail, 0);
    await runOrFallback("stats:b", fail, 0);
    await runOrFallback("stats:ok", async () => 1, 0);
    warn.mockRestore();

    const stats = fallbackStats();
    expect(stats.map((s) => [s.key, s.count]).sort()).toEqual([
      ["stats:a", 2],
      ["stats:b", 1],
    ]);
    for (const s of stats) expect(() => new Date(s.lastAt)).not.toThrow();
  });

  it("is empty when nothing has fallen back", () => {
    _resetFallbackStatsForTest();
    expect(fallbackStats()).toEqual([]);
  });
});
