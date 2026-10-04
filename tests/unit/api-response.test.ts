import { describe, expect, it } from "vitest";

import { networkErrorMessage, readApiResponse } from "@/lib/api-response";

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

describe("readApiResponse", () => {
  it("returns data from a 2xx { ok: true } body", async () => {
    const r = await readApiResponse<number[]>(
      json({ ok: true, data: [1, 2] }, 201),
      "post",
    );
    expect(r).toEqual({ ok: true, data: [1, 2] });
  });

  it("passes the route's own error message through", async () => {
    const r = await readApiResponse(
      json({ ok: false, error: "Sign in." }, 401),
      "post your comment",
    );
    expect(r).toEqual({ ok: false, error: "Sign in." });
  });

  it("turns a non-JSON 500 (the production failure) into a visible error", async () => {
    const res = new Response("Internal Server Error", { status: 500 });
    const r = await readApiResponse(res, "post your comment");
    expect(r.ok).toBe(false);
    if (!r.ok) {
      expect(r.error).toMatch(/post your comment/);
      expect(r.error).toMatch(/500/);
    }
  });

  it("treats an HTML error page as an error", async () => {
    const res = new Response("<!doctype html><h1>500</h1>", {
      status: 500,
      headers: { "content-type": "text/html" },
    });
    expect((await readApiResponse(res, "load comments")).ok).toBe(false);
  });

  it("treats unexpected JSON shapes as errors", async () => {
    expect((await readApiResponse(json([1, 2]), "x")).ok).toBe(false);
    expect((await readApiResponse(json(null), "x")).ok).toBe(false);
    expect((await readApiResponse(json({ ok: false }), "x")).ok).toBe(false);
    expect((await readApiResponse(json({ ok: true }), "x")).ok).toBe(false);
  });

  it("doesn't trust { ok: true } on a non-2xx status", async () => {
    const r = await readApiResponse(json({ ok: true, data: 1 }, 500), "x");
    expect(r.ok).toBe(false);
  });
});

describe("networkErrorMessage", () => {
  it("names the action", () => {
    expect(networkErrorMessage("load comments")).toMatch(/load comments/);
  });
});
