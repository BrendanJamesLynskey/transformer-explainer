import { describe, expect, it } from "vitest";

import { experimentSlug, slugify } from "@/lib/slug";

describe("slugify", () => {
  it("lowercases and dasherises whitespace + punctuation", () => {
    expect(slugify("My Experiment!")).toBe("my-experiment");
    expect(slugify("hello, world")).toBe("hello-world");
  });

  it("collapses runs of separators", () => {
    expect(slugify("foo   bar---baz")).toBe("foo-bar-baz");
  });

  it("trims leading and trailing separators", () => {
    expect(slugify("  -hi-  ")).toBe("hi");
  });

  it("falls back to 'experiment' on empty / weird inputs", () => {
    expect(slugify("")).toBe("experiment");
    expect(slugify("!!!")).toBe("experiment");
    expect(slugify("   ")).toBe("experiment");
  });

  it("caps length at 32 characters", () => {
    expect(slugify("a".repeat(80)).length).toBe(32);
  });
});

describe("experimentSlug", () => {
  it("appends a 6-char alphanumeric suffix", () => {
    const s = experimentSlug("Hello");
    expect(s).toMatch(/^hello-[a-z0-9]{6}$/);
  });

  it("yields different suffixes on each call", () => {
    const a = experimentSlug("x");
    const b = experimentSlug("x");
    expect(a).not.toBe(b);
  });
});
