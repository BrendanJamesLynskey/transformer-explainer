/**
 * One deployment, two addresses.
 *
 * Part 1 (Inference, `/learn`) lives at the decoder host; Part 2 (Training,
 * `/training`) has its own address, the training host, served by the same
 * Vercel project and the same build. `src/middleware.ts` calls `routeRequest`
 * on every page request to decide, from the host and the path, whether to
 * serve the page or redirect it to the other host, and which site the header
 * should name.
 *
 * The rules (brief 29E):
 *   - training host `/`            → 307 to `/training` (its home page)
 *   - training host `/training*`   → served; the header names the Training site
 *   - training host `/signin`, `/account` → served (sign-in works on both hosts)
 *   - training host anything else  → 308 to the same path on the decoder host
 *   - decoder host `/training*`    → 308 to the same path on the training host,
 *                                    so links to the old address keep working
 *   - any other host (localhost, CI, preview URLs) → served as it is, so the
 *     tests run without either production address.
 *
 * `training.localhost:<port>` plays the training host locally (browsers send
 * `*.localhost` to the loopback), with `decoder.localhost:<port>` as its
 * decoder host, so the e2e tests can exercise the host split. (Not plain
 * `localhost:<port>`: `next start` turns a middleware redirect to its own
 * address into a relative one, which would send the reader back to the
 * training stand-in.)
 *
 * Pure: no Next.js imports, so it is unit-tested directly.
 */

export const DECODER_HOST = "transformer-decoder-explained.vercel.app";
export const TRAINING_HOST = "transformer-training-explained.vercel.app";
export const DECODER_ORIGIN = `https://${DECODER_HOST}`;
export const TRAINING_ORIGIN = `https://${TRAINING_HOST}`;

/**
 * Production addresses of the decoder site whose `/training*` redirects.
 * (The project's old alias, transformer-explainer-three, already redirects
 * everything to the decoder host: `redirects()` in next.config.mjs.)
 */
const DECODER_PROD_HOSTS = new Set([DECODER_HOST]);

/** Local stand-in for the training host (`training.localhost:<port>`). */
const LOCAL_TRAINING = /^training\.localhost(:\d+)?$/;

/** Which site the header names. */
export type SiteName = "decoder" | "training";

/**
 * What a page needs to know about where it is being served:
 *   - `site`: the site to name in the header (`training` on the training
 *     host and on every `/training*` page, wherever it is served);
 *   - `decoderOrigin` / `trainingOrigin`: the prefix for links to the other
 *     part. Empty means "this host": a relative link is right.
 */
export type SiteContext = {
  site: SiteName;
  decoderOrigin: string;
  trainingOrigin: string;
};

export type RouteDecision =
  | { kind: "serve"; context: SiteContext }
  | { kind: "redirect"; status: 307 | 308; location: string };

/** Is this path one of Part 2's (`/training` or below it)? */
export function isTrainingPath(pathname: string): boolean {
  return pathname === "/training" || pathname.startsWith("/training/");
}

/** Paths the training host serves itself besides `/training*`. */
function servedOnTrainingHost(pathname: string): boolean {
  return (
    isTrainingPath(pathname) ||
    pathname === "/signin" ||
    pathname === "/account" ||
    pathname.startsWith("/api/")
  );
}

/** Lower-cased host header without a trailing dot. */
function normaliseHost(host: string | null | undefined): string {
  return (host ?? "").trim().toLowerCase().replace(/\.$/, "");
}

/**
 * Decide what to do with a request for `pathname` (+ `search`, with its
 * leading `?` or empty) arriving at `host`.
 */
export function routeRequest(
  hostHeader: string | null | undefined,
  pathname: string,
  search = "",
): RouteDecision {
  const host = normaliseHost(hostHeader);
  const local = LOCAL_TRAINING.exec(host);
  const onTrainingHost = host === TRAINING_HOST || local !== null;

  if (onTrainingHost) {
    // the training host's decoder: production, or plain localhost locally
    const decoderOrigin =
      local !== null
        ? `http://decoder.localhost${local[1] ?? ""}`
        : DECODER_ORIGIN;
    if (pathname === "/") {
      return { kind: "redirect", status: 307, location: `/training${search}` };
    }
    if (!servedOnTrainingHost(pathname)) {
      return {
        kind: "redirect",
        status: 308,
        location: `${decoderOrigin}${pathname}${search}`,
      };
    }
    return {
      kind: "serve",
      context: { site: "training", decoderOrigin, trainingOrigin: "" },
    };
  }

  if (DECODER_PROD_HOSTS.has(host)) {
    if (isTrainingPath(pathname)) {
      return {
        kind: "redirect",
        status: 308,
        location: `${TRAINING_ORIGIN}${pathname}${search}`,
      };
    }
    return {
      kind: "serve",
      context: {
        site: "decoder",
        decoderOrigin: "",
        trainingOrigin: TRAINING_ORIGIN,
      },
    };
  }

  // localhost, CI, preview deployments: both parts on one host
  return {
    kind: "serve",
    context: {
      site: isTrainingPath(pathname) ? "training" : "decoder",
      decoderOrigin: "",
      trainingOrigin: "",
    },
  };
}

/**
 * Turn a site-relative link into the right one for the current host: a
 * `/training*` path gets the training origin, any other site path the
 * decoder origin (each empty when it is this host). External links, anchors
 * and API paths are returned unchanged.
 */
export function siteHref(href: string, ctx: SiteContext): string {
  if (!href.startsWith("/") || href.startsWith("//")) return href;
  const path = href.split(/[?#]/, 1)[0]!;
  if (path.startsWith("/api/")) return href;
  return (isTrainingPath(path) ? ctx.trainingOrigin : ctx.decoderOrigin) + href;
}

/** The canonical address of a Part 2 page (always the training host). */
export function trainingCanonical(pathname: string): string {
  return `${TRAINING_ORIGIN}${pathname}`;
}

/** Request headers the middleware uses to hand the context to pages. */
export const CONTEXT_HEADERS = {
  site: "x-te-site",
  decoderOrigin: "x-te-decoder-origin",
  trainingOrigin: "x-te-training-origin",
} as const;

/** Read a context back from request headers (defaults: decoder, relative). */
export function contextFromHeaders(
  get: (name: string) => string | null,
): SiteContext {
  return {
    site: get(CONTEXT_HEADERS.site) === "training" ? "training" : "decoder",
    decoderOrigin: get(CONTEXT_HEADERS.decoderOrigin) ?? "",
    trainingOrigin: get(CONTEXT_HEADERS.trainingOrigin) ?? "",
  };
}
