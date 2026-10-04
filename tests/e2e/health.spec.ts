/**
 * Phase-11 e2e: `/api/health` against the real (migrated) test database.
 *
 * The unit tests cover the comparison logic; this proves the SQL works
 * against Postgres and that the reply carries no connection details.
 */
import { expect, test } from "@playwright/test";

test("health reports a reachable, current database @smoke", async ({
  request,
}) => {
  const res = await request.get("/api/health");
  expect(res.status()).toBe(200);
  expect(res.headers()["cache-control"]).toContain("no-store");
  const json = (await res.json()) as {
    ok: boolean;
    data: {
      healthy: boolean;
      db: string;
      schema: { status: string; applied: number; expected: number };
    };
  };
  expect(json.ok).toBe(true);
  expect(json.data.healthy).toBe(true);
  expect(json.data.db).toBe("up");
  expect(json.data.schema.status).toBe("current");
  expect(json.data.schema.applied).toBe(json.data.schema.expected);

  const raw = JSON.stringify(json);
  expect(raw).not.toMatch(/postgres(ql)?:\/\//);
  expect(raw).not.toMatch(/password/i);
});
