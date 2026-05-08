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
  // sslmode=require lives in the URL for Neon; we set it explicitly here too
  // so a misconfigured URL still works.
  return postgres(env.DATABASE_URL, { ssl: "require", max: 5 });
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
