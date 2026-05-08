/**
 * Unit tests for the pure auth helpers (no DB, no Auth.js wiring).
 */
import { describe, expect, it } from "vitest";

import {
  buildE2EUser,
  enrichSession,
  isAdmin,
  mapGitHubProfile,
} from "@/lib/auth/helpers";

describe("mapGitHubProfile", () => {
  it("stringifies the numeric GitHub id and copies login/email/avatar", () => {
    const out = mapGitHubProfile({
      id: 12345,
      login: "alice",
      name: "Alice Smith",
      email: "alice@example.com",
      avatar_url: "https://avatars.example.com/alice.png",
    });
    expect(out).toEqual({
      id: "12345",
      name: "Alice Smith",
      email: "alice@example.com",
      image: "https://avatars.example.com/alice.png",
      githubId: "12345",
      githubLogin: "alice",
    });
  });

  it("falls back to login when name is missing", () => {
    const out = mapGitHubProfile({ id: 1, login: "bob" });
    expect(out.name).toBe("bob");
    expect(out.email).toBeNull();
    expect(out.image).toBeNull();
  });
});

describe("buildE2EUser", () => {
  it("returns null when E2E auth is disabled", () => {
    expect(buildE2EUser("alice", false)).toBeNull();
  });

  it("returns a synthetic user when E2E auth is enabled", () => {
    const u = buildE2EUser("Alice", true);
    expect(u).toMatchObject({
      id: "e2e:alice",
      name: "alice",
      email: "alice@example.invalid",
      image: null,
      githubId: null,
      githubLogin: "alice",
    });
  });

  it("uses 'e2e-user' when no username is provided", () => {
    expect(buildE2EUser(undefined, true)?.githubLogin).toBe("e2e-user");
  });
});

describe("enrichSession", () => {
  it("merges githubLogin/githubId onto session.user", () => {
    const session = { user: { name: "Alice", email: "a@example.com" } };
    const out = enrichSession(session, {
      githubLogin: "alice",
      githubId: "12345",
    });
    expect(out.user).toMatchObject({
      name: "Alice",
      email: "a@example.com",
      githubLogin: "alice",
      githubId: "12345",
    });
  });

  it("emits null when the user has no GitHub fields", () => {
    const out = enrichSession({ user: {} }, {});
    expect(out.user).toEqual({ githubLogin: null, githubId: null });
  });
});

describe("isAdmin", () => {
  const allow = new Set(["alice", "bob"]);

  it("matches case-insensitively", () => {
    expect(isAdmin("Alice", allow)).toBe(true);
    expect(isAdmin("ALICE", allow)).toBe(true);
  });

  it("rejects non-admins and missing logins", () => {
    expect(isAdmin("charlie", allow)).toBe(false);
    expect(isAdmin(null, allow)).toBe(false);
    expect(isAdmin(undefined, allow)).toBe(false);
    expect(isAdmin("", allow)).toBe(false);
  });
});
