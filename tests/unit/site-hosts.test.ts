/**
 * Tests for src/lib/site/hosts.ts and src/lib/site/sitemap.ts: which host
 * serves which part, the redirects between them, and the links and sitemap
 * each host gets (brief 29E).
 */
import { describe, expect, it } from "vitest";

import {
  CONTEXT_HEADERS,
  DECODER_ORIGIN,
  TRAINING_ORIGIN,
  contextFromHeaders,
  isTrainingPath,
  routeRequest,
  siteHref,
  trainingCanonical,
} from "@/lib/site/hosts";
import { sitemapOrigin, sitemapUrls } from "@/lib/site/sitemap";

const TRAINING = "transformer-training-explained.vercel.app";
const DECODER = "transformer-decoder-explained.vercel.app";

describe("routeRequest on the training host", () => {
  it("sends / to /training (temporary, keeping the query)", () => {
    expect(routeRequest(TRAINING, "/", "?x=1")).toEqual({
      kind: "redirect",
      status: 307,
      location: "/training?x=1",
    });
  });

  it("serves Part 2 and names the Training site, linking Part 1 on the decoder host", () => {
    for (const path of ["/training", "/training/13-slow-links"]) {
      expect(routeRequest(TRAINING, path)).toEqual({
        kind: "serve",
        context: {
          site: "training",
          decoderOrigin: DECODER_ORIGIN,
          trainingOrigin: "",
        },
      });
    }
  });

  it("serves sign-in, the account page and the API itself", () => {
    for (const path of ["/signin", "/account", "/api/auth/session"])
      expect(routeRequest(TRAINING, path).kind).toBe("serve");
  });

  it("sends Part 1 and the decoder's other pages to the decoder host (308)", () => {
    for (const path of [
      "/learn",
      "/learn/03-attention",
      "/playground",
      "/about",
      "/trainingx",
    ])
      expect(routeRequest(TRAINING, path, "?q=2")).toEqual({
        kind: "redirect",
        status: 308,
        location: `${DECODER_ORIGIN}${path}?q=2`,
      });
  });

  it("ignores case and a trailing dot in the host", () => {
    expect(
      routeRequest("Transformer-Training-Explained.Vercel.App.", "/").kind,
    ).toBe("redirect");
  });

  it("plays the training host locally as training.localhost, with decoder.localhost as its decoder", () => {
    expect(routeRequest("training.localhost:3000", "/learn")).toEqual({
      kind: "redirect",
      status: 308,
      location: "http://decoder.localhost:3000/learn",
    });
    expect(routeRequest("training.localhost", "/training")).toEqual({
      kind: "serve",
      context: {
        site: "training",
        decoderOrigin: "http://decoder.localhost",
        trainingOrigin: "",
      },
    });
  });
});

describe("routeRequest on the decoder host", () => {
  it("moves /training* to the training host permanently, keeping path and query", () => {
    expect(
      routeRequest(DECODER, "/training/11-data-parallelism", "?a=b"),
    ).toEqual({
      kind: "redirect",
      status: 308,
      location: `${TRAINING_ORIGIN}/training/11-data-parallelism?a=b`,
    });
  });

  it("serves everything else as the decoder site, linking Training on the training host", () => {
    expect(routeRequest(DECODER, "/learn/01-overview")).toEqual({
      kind: "serve",
      context: {
        site: "decoder",
        decoderOrigin: "",
        trainingOrigin: TRAINING_ORIGIN,
      },
    });
  });
});

describe("routeRequest on any other host (localhost, CI, previews)", () => {
  it("never redirects; Training pages still name the Training site", () => {
    for (const host of [
      "localhost:3000",
      "127.0.0.1:3000",
      "transformer-explainer-abc123.vercel.app",
      null,
    ]) {
      expect(routeRequest(host, "/training/01-training-loop")).toEqual({
        kind: "serve",
        context: { site: "training", decoderOrigin: "", trainingOrigin: "" },
      });
      expect(routeRequest(host, "/")).toEqual({
        kind: "serve",
        context: { site: "decoder", decoderOrigin: "", trainingOrigin: "" },
      });
    }
  });
});

describe("siteHref", () => {
  const onTraining = {
    site: "training",
    decoderOrigin: DECODER_ORIGIN,
    trainingOrigin: "",
  } as const;
  const onDecoder = {
    site: "decoder",
    decoderOrigin: "",
    trainingOrigin: TRAINING_ORIGIN,
  } as const;

  it("points each part's paths at its own host", () => {
    expect(siteHref("/learn/03-attention#x", onTraining)).toBe(
      `${DECODER_ORIGIN}/learn/03-attention#x`,
    );
    expect(siteHref("/training/02-data-and-tokens", onTraining)).toBe(
      "/training/02-data-and-tokens",
    );
    expect(siteHref("/training?x", onDecoder)).toBe(
      `${TRAINING_ORIGIN}/training?x`,
    );
    expect(siteHref("/about", onDecoder)).toBe("/about");
  });

  it("leaves external links, anchors, protocol-relative and API paths alone", () => {
    for (const href of [
      "https://example.com/",
      "#t-loss",
      "//cdn.example.com/x",
      "/api/health",
    ])
      expect(siteHref(href, onTraining)).toBe(href);
  });
});

describe("helpers", () => {
  it("isTrainingPath matches /training and below only", () => {
    expect(isTrainingPath("/training")).toBe(true);
    expect(isTrainingPath("/training/01-training-loop")).toBe(true);
    expect(isTrainingPath("/trainings")).toBe(false);
    expect(isTrainingPath("/learn")).toBe(false);
  });

  it("canonical Part 2 addresses are on the training host", () => {
    expect(trainingCanonical("/training/13-slow-links")).toBe(
      `${TRAINING_ORIGIN}/training/13-slow-links`,
    );
  });

  it("contextFromHeaders reads the middleware's headers, defaulting to the decoder", () => {
    const h = new Map<string, string>([
      [CONTEXT_HEADERS.site, "training"],
      [CONTEXT_HEADERS.decoderOrigin, DECODER_ORIGIN],
      [CONTEXT_HEADERS.trainingOrigin, ""],
    ]);
    expect(contextFromHeaders((n) => h.get(n) ?? null)).toEqual({
      site: "training",
      decoderOrigin: DECODER_ORIGIN,
      trainingOrigin: "",
    });
    expect(contextFromHeaders(() => null)).toEqual({
      site: "decoder",
      decoderOrigin: "",
      trainingOrigin: "",
    });
  });
});

describe("sitemap", () => {
  it("the training host lists Part 2 at the training host", () => {
    expect(sitemapOrigin(TRAINING)).toBe(TRAINING_ORIGIN);
    expect(
      sitemapUrls(TRAINING, ["01-overview"], ["01-training-loop"]),
    ).toEqual([
      `${TRAINING_ORIGIN}/training`,
      `${TRAINING_ORIGIN}/training/01-training-loop`,
    ]);
    expect(sitemapOrigin("training.localhost:3000")).toBe(TRAINING_ORIGIN);
  });

  it("every other host lists Part 1 at the decoder host, and no Training page", () => {
    expect(sitemapOrigin(DECODER)).toBe(DECODER_ORIGIN);
    const urls = sitemapUrls(
      "localhost:3000",
      ["01-overview"],
      ["01-training-loop"],
    );
    expect(urls).toContain(`${DECODER_ORIGIN}/learn/01-overview`);
    expect(urls).toContain(`${DECODER_ORIGIN}/foundations`);
    expect(urls.some((u) => u.includes("/training"))).toBe(false);
    expect(sitemapOrigin(null)).toBe(DECODER_ORIGIN);
  });
});
