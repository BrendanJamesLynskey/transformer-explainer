/**
 * Experiments data layer.
 *
 * Zod schemas for the API surface + thin Drizzle helpers. The route
 * handlers stay focused on auth/HTTP and delegate to these functions for
 * the actual queries.
 */
import { and, desc, eq, sql } from "drizzle-orm";
import { z } from "zod";

import { db } from "@/lib/db/client";
import {
  type Experiment,
  type NewExperiment,
  experiments,
} from "@/lib/db/schema";
import { experimentSlug } from "@/lib/slug";

export const VISIBILITY = ["public", "unlisted", "private"] as const;
export const visibilitySchema = z.enum(VISIBILITY);

export const configSchema = z.object({
  seq_len: z.number().int().min(1).max(64),
  d_model: z.number().int().min(2).max(256),
  n_heads: z.number().int().min(1).max(8),
  d_ff: z.number().int().min(2).max(256),
  n_blocks: z.number().int().min(1).max(8),
  vocab_size: z.number().int().min(2).max(256),
  seed: z.number().int(),
});

export const layoutSchema = z.record(z.string(), z.unknown());

export const createSchema = z.object({
  name: z.string().min(1).max(80),
  visibility: visibilitySchema.default("private"),
  inputText: z.string().max(200).default(""),
  config: configSchema,
  layout: layoutSchema.default({}),
});

export const updateSchema = z.object({
  name: z.string().min(1).max(80).optional(),
  visibility: visibilitySchema.optional(),
  inputText: z.string().max(200).optional(),
  config: configSchema.optional(),
  layout: layoutSchema.optional(),
});

export type CreateInput = z.infer<typeof createSchema>;
export type UpdateInput = z.infer<typeof updateSchema>;

/** Public-listing query: paginated, newest first. */
export async function listPublic(
  limit = 20,
  offset = 0,
): Promise<Experiment[]> {
  return db
    .select()
    .from(experiments)
    .where(eq(experiments.visibility, "public"))
    .orderBy(desc(experiments.createdAt))
    .limit(limit)
    .offset(offset);
}

/** Owner's experiments, regardless of visibility. */
export async function listByOwner(ownerId: string): Promise<Experiment[]> {
  return db
    .select()
    .from(experiments)
    .where(eq(experiments.ownerId, ownerId))
    .orderBy(desc(experiments.createdAt));
}

export async function getBySlug(slug: string): Promise<Experiment | null> {
  const rows = await db
    .select()
    .from(experiments)
    .where(eq(experiments.slug, slug))
    .limit(1);
  return rows[0] ?? null;
}

export async function create(
  ownerId: string,
  input: CreateInput,
): Promise<Experiment> {
  const slug = experimentSlug(input.name);
  const row: NewExperiment = {
    slug,
    ownerId,
    name: input.name,
    visibility: input.visibility,
    inputText: input.inputText,
    configJson: input.config,
    layoutJson: input.layout,
  };
  const inserted = await db.insert(experiments).values(row).returning();
  return inserted[0]!;
}

export async function update(
  ownerId: string,
  slug: string,
  patch: UpdateInput,
): Promise<Experiment | null> {
  const next: Partial<NewExperiment> & { updatedAt?: Date } = {
    ...(patch.name !== undefined ? { name: patch.name } : {}),
    ...(patch.visibility !== undefined ? { visibility: patch.visibility } : {}),
    ...(patch.inputText !== undefined ? { inputText: patch.inputText } : {}),
    ...(patch.config !== undefined ? { configJson: patch.config } : {}),
    ...(patch.layout !== undefined ? { layoutJson: patch.layout } : {}),
    updatedAt: new Date(),
  };
  const updated = await db
    .update(experiments)
    .set(next)
    .where(and(eq(experiments.slug, slug), eq(experiments.ownerId, ownerId)))
    .returning();
  return updated[0] ?? null;
}

export async function remove(ownerId: string, slug: string): Promise<boolean> {
  const deleted = await db
    .delete(experiments)
    .where(and(eq(experiments.slug, slug), eq(experiments.ownerId, ownerId)))
    .returning({ id: experiments.id });
  return deleted.length > 0;
}

export async function fork(
  ownerId: string,
  source: Experiment,
): Promise<Experiment> {
  const newSlug = experimentSlug(source.name);
  const row: NewExperiment = {
    slug: newSlug,
    ownerId,
    name: `${source.name} (fork)`,
    visibility: "private",
    inputText: source.inputText,
    configJson: source.configJson,
    layoutJson: source.layoutJson,
    forkedFrom: source.id,
  };
  const inserted = await db.insert(experiments).values(row).returning();
  return inserted[0]!;
}

/** Increment view_count atomically. Best-effort — failures are swallowed. */
export async function incrementViews(slug: string): Promise<void> {
  try {
    await db
      .update(experiments)
      .set({ viewCount: sql`${experiments.viewCount} + 1` })
      .where(eq(experiments.slug, slug));
  } catch {
    // Telemetry-style; never block a page render on it.
  }
}
