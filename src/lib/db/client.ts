/**
 * Drizzle client singleton.
 *
 * `postgres-js` (the driver) is happy with a single connection in
 * dev/serverless. We attach the instance to `globalThis` to survive Next.js
 * dev-mode hot reload — without that, every reload spawns a fresh pool and
 * Neon eventually rejects new connections.
 */
import { drizzle } from "drizzle-orm/postgres-js";
import postgres from "postgres";

import { env } from "@/lib/env";

import * as schema from "./schema";

declare global {
  // eslint-disable-next-line no-var
  var __pg_pool__: ReturnType<typeof postgres> | undefined;
}

function makePool() {
  if (!env.DATABASE_URL) {
    throw new Error(
      "db: DATABASE_URL is not set. Add it to .env.local or your deployment env.",
    );
  }
  // Neon (and most managed Postgres) require TLS; local containers (CI,
  // `docker run postgres`) typically don't speak TLS at all. Sniff the host
  // to decide so a single client config works everywhere.
  const url = new URL(env.DATABASE_URL);
  const isLocal =
    url.hostname === "localhost" ||
    url.hostname === "127.0.0.1" ||
    url.hostname.endsWith(".internal");
  return postgres(env.DATABASE_URL, {
    ssl: isLocal ? false : "require",
    max: 5,
  });
}

const pool: ReturnType<typeof postgres> = globalThis.__pg_pool__ ?? makePool();
if (env.NODE_ENV !== "production") globalThis.__pg_pool__ = pool;

/**
 * Drizzle DB handle, typed against `schema`. Import this for queries:
 *
 *     import { db } from "@/lib/db/client";
 *     await db.select().from(schema.users);
 */
export const db = drizzle(pool, { schema });

export { schema };
