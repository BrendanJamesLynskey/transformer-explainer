/**
 * Tests for src/lib/env.ts.
 *
 * We exercise `makeEnvSchema` and `parseAdminLogins` directly so we don't
 * have to wrestle with module-load-time evaluation of process.env. The
 * `env` export itself is implicitly covered every time the test runner
 * imports a module that imports it (so that path is exercised by `pnpm
 * dev`, the smoke test, and CI's process.env).
 */
import { describe, it, expect } from "vitest";

import { makeEnvSchema, parseAdminLogins } from "@/lib/env";

const validBase = {
  NODE_ENV: "test",
  DATABASE_URL: "postgresql://u:p@host.example.com/db?sslmode=require",
  AUTH_SECRET: "x".repeat(32),
  AUTH_GITHUB_ID: "id",
  AUTH_GITHUB_SECRET: "secret",
  NEXT_PUBLIC_SITE_URL: "http://localhost:3000",
  ADMIN_GITHUB_LOGINS: "alice, BOB ,charlie",
  MAX_SEQ_LEN: "16",
  MAX_D_MODEL: "64",
  MAX_BLOCKS: "4",
  NEXT_PUBLIC_DEV_DEMOS_ENABLED: "true",
};

describe("makeEnvSchema (non-prod)", () => {
  const schema = makeEnvSchema(false);

  it("parses a fully-populated env", () => {
    const parsed = schema.parse(validBase);
    expect(parsed.MAX_SEQ_LEN).toBe(16);
    expect(parsed.MAX_D_MODEL).toBe(64);
    expect(parsed.MAX_BLOCKS).toBe(4);
    expect(parsed.NEXT_PUBLIC_DEV_DEMOS_ENABLED).toBe(true);
    expect(parsed.NODE_ENV).toBe("test");
  });

  it("applies defaults when optional fields are absent", () => {
    const parsed = schema.parse({});
    expect(parsed.NODE_ENV).toBe("development");
    expect(parsed.NEXT_PUBLIC_SITE_URL).toBe("http://localhost:3000");
    expect(parsed.ADMIN_GITHUB_LOGINS).toBe("");
    expect(parsed.MAX_SEQ_LEN).toBe(16);
    expect(parsed.MAX_D_MODEL).toBe(64);
    expect(parsed.MAX_BLOCKS).toBe(4);
    expect(parsed.NEXT_PUBLIC_DEV_DEMOS_ENABLED).toBe(false);
  });

  it("tolerates missing AUTH_* and DATABASE_URL", () => {
    expect(() => schema.parse({ NODE_ENV: "development" })).not.toThrow();
  });

  it("rejects a non-postgres DATABASE_URL", () => {
    expect(() =>
      schema.parse({ ...validBase, DATABASE_URL: "https://example.com" }),
    ).toThrow(/postgres/);
  });

  it("accepts both postgres:// and postgresql:// schemes", () => {
    expect(() =>
      schema.parse({
        ...validBase,
        DATABASE_URL: "postgres://u:p@host/db",
      }),
    ).not.toThrow();
    expect(() =>
      schema.parse({
        ...validBase,
        DATABASE_URL: "postgresql://u:p@host/db",
      }),
    ).not.toThrow();
  });

  it("rejects MAX_SEQ_LEN larger than the hard cap", () => {
    expect(() => schema.parse({ ...validBase, MAX_SEQ_LEN: "999" })).toThrow();
  });

  it("rejects non-numeric compute limits", () => {
    expect(() => schema.parse({ ...validBase, MAX_BLOCKS: "lots" })).toThrow();
  });

  it("rejects an unknown NODE_ENV", () => {
    expect(() => schema.parse({ ...validBase, NODE_ENV: "staging" })).toThrow();
  });

  it("rejects an unknown NEXT_PUBLIC_DEV_DEMOS_ENABLED", () => {
    expect(() =>
      schema.parse({ ...validBase, NEXT_PUBLIC_DEV_DEMOS_ENABLED: "yes" }),
    ).toThrow();
  });
});

describe("makeEnvSchema (prod)", () => {
  const schema = makeEnvSchema(true);

  it("requires AUTH_SECRET, AUTH_GITHUB_*, and DATABASE_URL", () => {
    const result = schema.safeParse({ NODE_ENV: "production" });
    expect(result.success).toBe(false);
    if (!result.success) {
      const messages = result.error.issues.map((i) => i.message).join("\n");
      expect(messages).toMatch(/AUTH_SECRET is required in production/);
      expect(messages).toMatch(/AUTH_GITHUB_ID is required in production/);
      expect(messages).toMatch(/AUTH_GITHUB_SECRET is required in production/);
      expect(messages).toMatch(/DATABASE_URL is required in production/);
    }
  });

  it("treats empty strings as missing in production", () => {
    const result = schema.safeParse({
      ...validBase,
      NODE_ENV: "production",
      AUTH_SECRET: "",
    });
    expect(result.success).toBe(false);
    if (!result.success) {
      expect(
        result.error.issues.some((i) =>
          i.message.includes("AUTH_SECRET is required"),
        ),
      ).toBe(true);
    }
  });

  it("accepts a fully-populated production env", () => {
    expect(() =>
      schema.parse({ ...validBase, NODE_ENV: "production" }),
    ).not.toThrow();
  });
});

describe("parseAdminLogins", () => {
  it("trims, lowercases, and dedupes entries", () => {
    const set = parseAdminLogins("Alice, BOB , alice ,charlie,");
    expect([...set].sort()).toEqual(["alice", "bob", "charlie"]);
  });

  it("returns an empty set for an empty string", () => {
    expect(parseAdminLogins("").size).toBe(0);
  });

  it("returns an empty set for whitespace-only input", () => {
    expect(parseAdminLogins("   ,  ,").size).toBe(0);
  });
});
