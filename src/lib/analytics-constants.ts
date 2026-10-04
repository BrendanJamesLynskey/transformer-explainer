/**
 * Analytics constants shared by server and client code.
 *
 * Kept free of imports so client components can use them without pulling
 * Zod (via `analytics-shared.ts`) into the browser bundle.
 */

/**
 * Header the client sends its analytics session id in, so events that the
 * *server* records (a comment posted, a progress upsert) join the same
 * session as the page view that led to them.
 */
export const SESSION_HEADER = "x-te-session";
