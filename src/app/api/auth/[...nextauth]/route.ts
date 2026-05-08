/**
 * Auth.js v5 catch-all route. Auth.js owns this URL space — every callback,
 * sign-in, sign-out, and CSRF endpoint lives here.
 *
 * The `handlers` object from `NextAuth(config)` packages both verbs.
 */
import { handlers } from "@/lib/auth";

export const { GET, POST } = handlers;
