/**
 * Sign-in on the training host (brief 29E): the GitHub OAuth app has one
 * callback URL, on the decoder host, so `src/lib/auth/config.ts` sets
 * Auth.js's `redirectProxyUrl` to the decoder host in production. This runs
 * the installed Auth.js through both legs with real Request objects (no
 * database, no GitHub): the sign-in that starts on the training host asks
 * GitHub to return to the decoder host, and the decoder host's callback
 * forwards the code and state to the training host's own callback.
 */
import NextAuth from "next-auth";
import GitHub from "next-auth/providers/github";
import { NextRequest } from "next/server";
import { describe, expect, it } from "vitest";

const DECODER = "https://transformer-decoder-explained.vercel.app";
const TRAINING = "https://transformer-training-explained.vercel.app";

const { handlers } = NextAuth({
  secret: "test-secret-do-not-use-in-production-test-secret",
  trustHost: true,
  redirectProxyUrl: `${DECODER}/api/auth`,
  providers: [
    GitHub({
      clientId: "id",
      clientSecret: "secret",
      issuer: "https://github.com/login/oauth",
    }),
  ],
  session: { strategy: "jwt" },
});

/** A request as Vercel delivers it: the reader's host in the Host header. */
function req(
  url: string,
  init: {
    method?: string;
    headers?: Record<string, string>;
    body?: string;
  } = {},
): NextRequest {
  const headers = new Headers(init.headers);
  headers.set("host", new URL(url).host);
  headers.set("x-forwarded-proto", "https");
  return new NextRequest(url, {
    method: init.method,
    body: init.body,
    headers,
  });
}

/** Cookie header from a response's Set-Cookie lines. */
function cookiesOf(res: Response): string {
  return res.headers
    .getSetCookie()
    .map((c) => c.split(";")[0])
    .join("; ");
}

describe("Auth.js redirect proxy across the two hosts", () => {
  it("a training-host sign-in returns through the decoder host to the training host", async () => {
    // 1. CSRF token and cookie on the training host
    const csrfRes = await handlers.GET(req(`${TRAINING}/api/auth/csrf`));
    const { csrfToken } = (await csrfRes.json()) as { csrfToken: string };
    // 2. start the GitHub sign-in on the training host
    const signin = await handlers.POST(
      req(`${TRAINING}/api/auth/signin/github`, {
        method: "POST",
        headers: {
          "content-type": "application/x-www-form-urlencoded",
          cookie: cookiesOf(csrfRes),
        },
        body: new URLSearchParams({
          csrfToken,
          callbackUrl: `${TRAINING}/training/03-objective`,
        }).toString(),
      }),
    );
    const github = new URL(signin.headers.get("location")!);
    expect(github.origin + github.pathname).toBe(
      "https://github.com/login/oauth/authorize",
    );
    // GitHub is asked to return to the one registered callback: the decoder's
    expect(github.searchParams.get("redirect_uri")).toBe(
      `${DECODER}/api/auth/callback/github`,
    );
    const state = github.searchParams.get("state")!;
    expect(state).toBeTruthy();
    // the state cookie is set on the training host, where the flow finishes
    expect(cookiesOf(signin)).toContain("authjs.state");

    // 3. GitHub returns to the decoder host, which forwards to the training host
    const back = await handlers.GET(
      req(
        `${DECODER}/api/auth/callback/github?code=abc&state=${encodeURIComponent(state)}`,
      ),
    );
    const fwd = new URL(back.headers.get("location")!);
    expect(fwd.origin + fwd.pathname).toBe(
      `${TRAINING}/api/auth/callback/github`,
    );
    expect(fwd.searchParams.get("code")).toBe("abc");
    expect(fwd.searchParams.get("state")).toBe(state);
  });

  it("a decoder-host sign-in is unchanged: GitHub returns straight to the decoder host", async () => {
    const csrfRes = await handlers.GET(req(`${DECODER}/api/auth/csrf`));
    const { csrfToken } = (await csrfRes.json()) as { csrfToken: string };
    const signin = await handlers.POST(
      req(`${DECODER}/api/auth/signin/github`, {
        method: "POST",
        headers: {
          "content-type": "application/x-www-form-urlencoded",
          cookie: cookiesOf(csrfRes),
        },
        body: new URLSearchParams({ csrfToken, callbackUrl: "/" }).toString(),
      }),
    );
    const github = new URL(signin.headers.get("location")!);
    expect(github.searchParams.get("redirect_uri")).toBe(
      `${DECODER}/api/auth/callback/github`,
    );
  });
});
