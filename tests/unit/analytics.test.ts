/**
 * Unit tests for `lib/analytics`. The DB-touching helpers
 * (`recordEvents`, `dailyActive`, `sectionFunnel`, `topExperiments`,
 * `recentComments`) need a live Postgres and are exercised by the e2e
 * admin flow. These tests cover the pure pieces.
 */
import { afterEach, describe, expect, it } from "vitest";

import {
  EVENT_KIND,
  SESSION_HEADER,
  _resetRateLimitForTest,
  buildServerEvent,
  ingestSchema,
  rateLimitOk,
  serverEventSessionId,
} from "@/lib/analytics-shared";

afterEach(() => {
  _resetRateLimitForTest();
});

describe("ingestSchema", () => {
  it("accepts a valid batch", () => {
    const r = ingestSchema.safeParse({
      sessionId: "abcd-1234-efgh",
      events: [
        { kind: "page_view", sectionSlug: "01-overview" },
        { kind: "widget_interact", sectionSlug: "01-overview" },
      ],
    });
    expect(r.success).toBe(true);
  });

  it("rejects an unknown kind", () => {
    const r = ingestSchema.safeParse({
      sessionId: "abcd-1234-efgh",
      events: [{ kind: "exfiltrate" }],
    });
    expect(r.success).toBe(false);
  });

  it("rejects a too-short sessionId", () => {
    const r = ingestSchema.safeParse({
      sessionId: "short",
      events: [{ kind: "page_view" }],
    });
    expect(r.success).toBe(false);
  });

  it("caps batch size at 50", () => {
    const r = ingestSchema.safeParse({
      sessionId: "abcd-1234-efgh",
      events: Array.from({ length: 51 }, () => ({ kind: "page_view" })),
    });
    expect(r.success).toBe(false);
  });

  it("EVENT_KIND covers every kind the schema accepts", () => {
    for (const kind of EVENT_KIND) {
      const r = ingestSchema.safeParse({
        sessionId: "abcd-1234-efgh",
        events: [{ kind }],
      });
      expect(r.success).toBe(true);
    }
  });
});

describe("rateLimitOk", () => {
  it("allows up to capacity then rejects", () => {
    // 3 tokens, refilling 1/sec — fixed `now` so no refill happens
    const now = 1_000_000;
    expect(rateLimitOk("k", 3, 1, now)).toBe(true);
    expect(rateLimitOk("k", 3, 1, now)).toBe(true);
    expect(rateLimitOk("k", 3, 1, now)).toBe(true);
    expect(rateLimitOk("k", 3, 1, now)).toBe(false);
  });

  it("refills over time", () => {
    const t0 = 1_000_000;
    expect(rateLimitOk("k", 1, 1, t0)).toBe(true);
    expect(rateLimitOk("k", 1, 1, t0)).toBe(false);
    // 1.5s later → bucket has refilled enough for one more
    expect(rateLimitOk("k", 1, 1, t0 + 1500)).toBe(true);
  });

  it("buckets are independent per key", () => {
    const now = 1_000_000;
    expect(rateLimitOk("a", 1, 1, now)).toBe(true);
    expect(rateLimitOk("b", 1, 1, now)).toBe(true);
    expect(rateLimitOk("a", 1, 1, now)).toBe(false);
    expect(rateLimitOk("b", 1, 1, now)).toBe(false);
  });
});

describe("server-recorded events", () => {
  it("includes the comment and progress kinds", () => {
    expect(EVENT_KIND).toContain("comment_post");
    expect(EVENT_KIND).toContain("progress_update");
    expect(SESSION_HEADER).toBe("x-te-session");
  });

  it("uses the client's session id when the header carries a valid one", () => {
    const id = "3f0c8a5e-1b2d-4c3e-9f00-123456789abc";
    expect(serverEventSessionId(id, "u1")).toBe(id);
  });

  it("falls back to a per-user id for a missing or malformed header", () => {
    expect(serverEventSessionId(null, "u1")).toBe("user:u1_");
    expect(serverEventSessionId("short", "u1")).toBe("user:u1_");
    expect(serverEventSessionId("has spaces in it!", "u1")).toBe("user:u1_");
    expect(serverEventSessionId("x".repeat(65), "u1")).toBe("user:u1_");
  });

  it("caps the fallback id at the schema's 64 characters", () => {
    const id = serverEventSessionId(null, "u".repeat(100));
    expect(id).toHaveLength(64);
    expect(() =>
      buildServerEvent(id, "comment_post", "01-overview"),
    ).not.toThrow();
  });

  it("builds a one-event batch that passes the ingest schema", () => {
    const batch = buildServerEvent(
      "user:u1_",
      "progress_update",
      "03-attention",
      {
        status: "completed",
        stored: "completed",
      },
    );
    expect(batch).toEqual({
      sessionId: "user:u1_",
      events: [
        {
          kind: "progress_update",
          sectionSlug: "03-attention",
          meta: { status: "completed", stored: "completed" },
        },
      ],
    });
    expect(buildServerEvent("user:u1_", "comment_post", "01-overview")).toEqual(
      {
        sessionId: "user:u1_",
        events: [{ kind: "comment_post", sectionSlug: "01-overview" }],
      },
    );
  });

  it("rejects what the ingest schema rejects", () => {
    expect(() =>
      buildServerEvent("user:u1_", "comment_post", "x".repeat(81)),
    ).toThrow();
  });
});
