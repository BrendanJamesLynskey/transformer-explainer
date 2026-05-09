/**
 * scripts/seed-db.ts
 *
 * Idempotent seed: creates the admin user (if not present), one named demo
 * experiment, and a single section-progress row so /learn lights up. Safe
 * to re-run.
 *
 * Run with `pnpm db:seed`. Reads DATABASE_URL from `.env.local` (or `.env`).
 */
import "./_load-env"; // must come before any module that reads env

import { eq } from "drizzle-orm";

import { db } from "@/lib/db/client";
import {
  experiments,
  progress,
  type NewExperiment,
  type NewProgress,
  type NewUser,
  users,
} from "@/lib/db/schema";

const ADMIN_GITHUB_LOGIN = "BrendanJamesLynskey";
const DEMO_SLUG = "demo-hello-world";

async function ensureAdminUser(): Promise<string> {
  const existing = await db
    .select()
    .from(users)
    .where(eq(users.githubLogin, ADMIN_GITHUB_LOGIN))
    .limit(1);
  if (existing[0]) return existing[0].id;

  const newUser: NewUser = {
    name: "Brendan Lynskey",
    email: "brendanlynskey@googlemail.com",
    githubLogin: ADMIN_GITHUB_LOGIN,
    // Sentinel — replaced by the real GitHub id on first OAuth sign-in.
    githubId: "seed:placeholder",
  };
  const inserted = await db.insert(users).values(newUser).returning();
  return inserted[0]!.id;
}

async function ensureDemoExperiment(ownerId: string): Promise<void> {
  const existing = await db
    .select()
    .from(experiments)
    .where(eq(experiments.slug, DEMO_SLUG))
    .limit(1);
  if (existing[0]) return;

  const e: NewExperiment = {
    slug: DEMO_SLUG,
    ownerId,
    name: "Hello, world",
    visibility: "public",
    inputText: "hello!",
    configJson: {
      seq_len: 8,
      d_model: 16,
      n_heads: 2,
      d_ff: 32,
      n_blocks: 2,
      vocab_size: 64,
      seed: 42,
    },
    layoutJson: { layers: { concept: true, maths: false, code: false } },
  };
  await db.insert(experiments).values(e);
}

async function ensureSeedProgress(userId: string): Promise<void> {
  const existing = await db
    .select()
    .from(progress)
    .where(eq(progress.userId, userId))
    .limit(1);
  if (existing[0]) return;

  const rows: NewProgress[] = [
    { userId, sectionSlug: "01-overview", status: "completed" },
    { userId, sectionSlug: "02-embeddings", status: "in_progress" },
  ];
  await db.insert(progress).values(rows);
}

async function main(): Promise<void> {
  console.log("Seeding database…");
  const adminId = await ensureAdminUser();
  await ensureDemoExperiment(adminId);
  await ensureSeedProgress(adminId);
  console.log("Done.");
  process.exit(0);
}

main().catch((err: unknown) => {
  console.error("Seed failed:", err);
  process.exit(1);
});
