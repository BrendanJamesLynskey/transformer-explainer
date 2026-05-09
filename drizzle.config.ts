import path from "node:path";

import { config as dotenv } from "dotenv";
import { defineConfig } from "drizzle-kit";

// drizzle-kit doesn't auto-load .env.local (only Next.js does), so do it here.
// `quiet: true` suppresses dotenv's "tip" promo lines.
dotenv({
  path: path.resolve(__dirname, ".env.local"),
  override: false,
  quiet: true,
});
dotenv({
  path: path.resolve(__dirname, ".env"),
  override: false,
  quiet: true,
});

if (!process.env.DATABASE_URL) {
  throw new Error(
    "DATABASE_URL is required for drizzle-kit. Set it in .env.local — see .env.example.",
  );
}

export default defineConfig({
  schema: "./src/lib/db/schema.ts",
  out: "./drizzle",
  dialect: "postgresql",
  dbCredentials: {
    url: process.env.DATABASE_URL,
  },
  verbose: true,
  strict: true,
});
