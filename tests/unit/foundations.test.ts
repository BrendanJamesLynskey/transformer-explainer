/**
 * Brief 44: the Foundations start page's glossary links each term to the
 * chapter section that defines it; every chapter and anchor must exist.
 */
import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

import { FOUNDATIONS_GLOSSARY } from "@/lib/learn/foundations";
import { SECTIONS, TRAINING_SECTIONS } from "@/lib/mdx/sections";

const CONTENT = path.resolve(__dirname, "../../content");

describe("/foundations glossary", () => {
  it("has a term for each Decoder chapter and most Training chapters", () => {
    const pages = new Set(
      FOUNDATIONS_GLOSSARY.map((g) => g.href.split("#")[0]),
    );
    for (const s of SECTIONS) expect(pages).toContain(`/learn/${s.slug}`);
    const training = TRAINING_SECTIONS.filter((s) =>
      pages.has(`/training/${s.slug}`),
    );
    expect(training.length).toBeGreaterThanOrEqual(14);
  });

  it.each(FOUNDATIONS_GLOSSARY.map((g) => [g.term, g.href]))(
    "%s → %s exists",
    (_term, href) => {
      const m = /^\/(learn|training)\/([\w-]+)(?:#([\w-]+))?$/.exec(href);
      expect(m).not.toBeNull();
      const dir = m![1] === "learn" ? "decoder" : "training";
      const source = readFileSync(
        path.join(CONTENT, dir, `${m![2]}.mdx`),
        "utf-8",
      );
      if (m![3]) expect(source).toContain(`id="${m![3]}"`);
    },
  );
});
