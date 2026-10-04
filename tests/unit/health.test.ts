import { describe, expect, it } from "vitest";

import journal from "../../drizzle/meta/_journal.json";

import {
  buildHealthReport,
  compareSchema,
  safeErrorCode,
  type JournalEntry,
} from "@/lib/health-shared";

const J: JournalEntry[] = [
  { idx: 0, when: 1000, tag: "0000_first" },
  { idx: 1, when: 2000, tag: "0001_second" },
];

describe("compareSchema", () => {
  it("is current when the newest applied migration is the newest shipped", () => {
    expect(compareSchema([1000, 2000], J)).toEqual({
      status: "current",
      applied: 2,
      expected: 2,
      latestExpected: "0001_second",
      latestApplied: "0001_second",
    });
  });

  it("is behind when the build ships a migration the DB hasn't run", () => {
    const r = compareSchema([1000], J);
    expect(r.status).toBe("behind");
    expect(r.latestApplied).toBe("0000_first");
  });

  it("is ahead when the DB has a migration this build doesn't know", () => {
    const r = compareSchema([1000, 2000, 3000], J);
    expect(r.status).toBe("ahead");
    expect(r.latestApplied).toBeNull();
  });

  it("is none when the migrations table is missing or empty", () => {
    expect(compareSchema(null, J).status).toBe("none");
    expect(compareSchema([], J).status).toBe("none");
  });

  it("treats an empty journal with no migrations as current", () => {
    expect(compareSchema(null, []).status).toBe("current");
  });

  it("does not depend on journal order", () => {
    expect(compareSchema([2000, 1000], [...J].reverse()).status).toBe(
      "current",
    );
  });

  it("reads the real journal shipped in drizzle/", () => {
    const entries = journal.entries as JournalEntry[];
    const whens = entries.map((e) => e.when);
    expect(compareSchema(whens, entries).status).toBe("current");
  });
});

describe("safeErrorCode", () => {
  it("passes through SQLSTATE and socket codes", () => {
    expect(
      safeErrorCode(Object.assign(new Error("x"), { code: "28P01" })),
    ).toBe("28P01");
    expect(safeErrorCode({ code: "ECONNREFUSED" })).toBe("ECONNREFUSED");
  });

  it("maps the timeout sentinel", () => {
    expect(safeErrorCode(new Error("timeout"))).toBe("TIMEOUT");
  });

  it("never returns a message or a code that could carry free text", () => {
    expect(
      safeErrorCode(new Error('password failed for user "neondb_owner"')),
    ).toBe("UNKNOWN");
    expect(safeErrorCode({ code: "host db.example.com refused" })).toBe(
      "UNKNOWN",
    );
    expect(safeErrorCode("boom")).toBe("UNKNOWN");
    expect(safeErrorCode(null)).toBe("UNKNOWN");
  });
});

describe("buildHealthReport", () => {
  const now = new Date("2026-10-04T12:00:00Z");

  it("is healthy when the DB is up and current", () => {
    const r = buildHealthReport(
      { reachable: true, schema: compareSchema([1000, 2000], J) },
      now,
    );
    expect(r).toMatchObject({ healthy: true, db: "up", dbError: null });
    expect(r.checkedAt).toBe("2026-10-04T12:00:00.000Z");
  });

  it("stays healthy when the DB is ahead", () => {
    const r = buildHealthReport(
      { reachable: true, schema: compareSchema([3000], J) },
      now,
    );
    expect(r.healthy).toBe(true);
  });

  it("is unhealthy when the schema is behind or missing", () => {
    for (const applied of [[1000], null]) {
      const r = buildHealthReport(
        { reachable: true, schema: compareSchema(applied, J) },
        now,
      );
      expect(r.healthy).toBe(false);
      expect(r.db).toBe("up");
    }
  });

  it("is unhealthy with a code when the DB is down", () => {
    const r = buildHealthReport(
      { reachable: false, error: { code: "ENOTFOUND" } },
      now,
    );
    expect(r).toMatchObject({
      healthy: false,
      db: "down",
      dbError: "ENOTFOUND",
      schema: null,
    });
  });

  it("uses the current time by default", () => {
    const r = buildHealthReport({ reachable: false, error: null });
    expect(Number.isNaN(Date.parse(r.checkedAt))).toBe(false);
  });
});
