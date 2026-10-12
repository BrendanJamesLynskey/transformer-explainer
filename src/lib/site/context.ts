/**
 * The current request's site context, for Server Components: which site the
 * header names, and the origins for links to the other part. Set by
 * `src/middleware.ts` (see `./hosts.ts`); without it (unit tests, a request
 * the middleware skipped) the page is the decoder site with relative links.
 */
import { headers } from "next/headers";

import { contextFromHeaders, type SiteContext } from "./hosts";

/** Read the site context the middleware put on this request. */
export function getSiteContext(): SiteContext {
  const h = headers();
  return contextFromHeaders((name) => h.get(name));
}
