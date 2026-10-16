import { describe, expect, it } from "vitest";

import { SHARE, TRAINING_SHARE, withShare } from "@/lib/share";
import { DECODER_ORIGIN as SITE_URL, TRAINING_ORIGIN } from "@/lib/site/hosts";

describe("share: link previews", () => {
  it("names the site, its colour and its address", () => {
    expect(SHARE.name).toBe("Transformer Decoder Explained");
    expect(SHARE.url).toBe(SITE_URL);
    expect(SHARE.color).toMatch(/^#[0-9a-f]{6}$/);
    expect(SHARE.claim.length).toBeLessThanOrEqual(80);
    expect(SHARE.checkedShort.length).toBeLessThanOrEqual(110);
  });

  it("gives the home page the site's own card", () => {
    const m = withShare("/", { description: "d" });
    expect(m.openGraph).toMatchObject({
      title: SHARE.name,
      url: "/",
      siteName: SHARE.name,
      type: "website",
      description: "d",
    });
    expect(m.twitter).toMatchObject({ card: "summary_large_image" });
  });

  it("names a page by its own title and path", () => {
    const m = withShare("/about", { title: "About" });
    expect(m.title).toBe("About");
    expect(m.openGraph).toMatchObject({
      title: `About · ${SHARE.name}`,
      url: "/about",
      type: "article",
      description: SHARE.description,
    });
    const a = withShare("/x", { title: { absolute: "X" } });
    expect(a.openGraph).toMatchObject({ title: `X · ${SHARE.name}` });
    const n = withShare("/y", { title: { default: "Y", template: "%s" } });
    expect(n.openGraph).toMatchObject({ title: SHARE.name });
  });

  it("gives the Training site, at its own address, its own card", () => {
    expect(TRAINING_SHARE.url).toBe(TRAINING_ORIGIN);
    expect(TRAINING_SHARE.name).toBe("Transformer Training Explained");
    const m = withShare(
      "/training/03-objective",
      { title: { absolute: "The objective · Transformer Training Explained" } },
      TRAINING_SHARE,
      true,
    );
    expect(m.openGraph).toMatchObject({
      title: "The objective · Transformer Training Explained",
      siteName: "Transformer Training Explained",
      url: "/training/03-objective",
      description: TRAINING_SHARE.description,
    });
    expect(m.openGraph).not.toHaveProperty("images");
    expect(withShare("/about", { title: "About" }).openGraph).toMatchObject({
      images: [{ url: "/opengraph-image" }],
    });
    expect(
      withShare("/training", { title: "T" }, TRAINING_SHARE).openGraph,
    ).toMatchObject({ images: [{ url: "/training/opengraph-image" }] });
  });
});
