/**
 * Load `.env.local` (then `.env`) into `process.env` for tsx + drizzle-kit
 * scripts. Next.js loads these automatically; standalone Node processes don't.
 *
 * Import for side-effects from the top of any script that needs env:
 *
 *   import "./_load-env";   // must be first
 *   import { db } from "@/lib/db/client";
 */
import path from "node:path";

import { config as dotenv } from "dotenv";

const root = path.resolve(__dirname, "..");

// `.env.local` wins over `.env` — same precedence Next.js uses.
// `quiet: true` suppresses dotenv's "tip" promo lines.
dotenv({ path: path.join(root, ".env.local"), override: false, quiet: true });
dotenv({ path: path.join(root, ".env"), override: false, quiet: true });
