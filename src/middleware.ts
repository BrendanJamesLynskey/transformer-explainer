/**
 * Host-based routing for the site's two addresses (see `src/lib/site/hosts.ts`).
 *
 * Runs before every page request: redirects between the decoder host and the
 * training host where a page belongs on the other one, and otherwise passes
 * the request on with three headers that tell the pages which site to name
 * and how to link to the other part.
 */
import { NextResponse, type NextRequest } from "next/server";

import { CONTEXT_HEADERS, routeRequest } from "@/lib/site/hosts";

export function middleware(request: NextRequest): NextResponse {
  const { pathname, search } = request.nextUrl;
  const decision = routeRequest(request.headers.get("host"), pathname, search);

  if (decision.kind === "redirect") {
    // a relative location stays on this host (the one the reader asked for)
    const host = request.headers.get("host") ?? request.nextUrl.host;
    const location = decision.location.startsWith("/")
      ? new URL(decision.location, `${request.nextUrl.protocol}//${host}`)
      : decision.location;
    return NextResponse.redirect(location, decision.status);
  }

  const headers = new Headers(request.headers);
  headers.set(CONTEXT_HEADERS.site, decision.context.site);
  headers.set(CONTEXT_HEADERS.decoderOrigin, decision.context.decoderOrigin);
  headers.set(CONTEXT_HEADERS.trainingOrigin, decision.context.trainingOrigin);
  return NextResponse.next({ request: { headers } });
}

export const config = {
  // pages and API routes; not Next's own files or anything with an extension
  // (public files such as /training/corpus.txt are served on both hosts)
  matcher: ["/((?!_next/static|_next/image|.*\\.[a-zA-Z0-9]+$).*)"],
};
