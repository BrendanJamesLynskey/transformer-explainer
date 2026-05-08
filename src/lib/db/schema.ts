/**
 * Drizzle schema (Postgres / Neon).
 *
 * Mirrors SPEC.md §8. The Auth.js-mandated `users / accounts / sessions /
 * verificationTokens` tables follow the shape that `@auth/drizzle-adapter`
 * expects (see https://authjs.dev/getting-started/adapters/drizzle).
 *
 * Conventions:
 * - Snake_case column names (Postgres-native), camelCase TS field names —
 *   Drizzle handles the mapping via the second arg to `text()` etc.
 * - Every domain table has `created_at` and (where applicable) `updated_at`
 *   as `timestamptz`.
 * - We use `text` rather than `varchar` for all strings; Postgres treats
 *   them identically and TEXT keeps schema diffs cleaner.
 */
import { sql } from "drizzle-orm";
import {
  bigserial,
  boolean,
  integer,
  jsonb,
  pgTable,
  primaryKey,
  text,
  timestamp,
  uniqueIndex,
  uuid,
  index,
} from "drizzle-orm/pg-core";
import type { AdapterAccountType } from "next-auth/adapters";

// ---------------------------------------------------------------------------
// Auth.js core tables (https://authjs.dev/getting-started/adapters/drizzle).
// ---------------------------------------------------------------------------

export const users = pgTable("users", {
  id: uuid("id").defaultRandom().primaryKey(),
  // Auth.js standard fields
  name: text("name"),
  email: text("email").unique(),
  emailVerified: timestamp("email_verified", {
    mode: "date",
    withTimezone: true,
  }),
  image: text("image"),
  // Project-specific extras
  githubId: text("github_id").unique(),
  githubLogin: text("github_login"),
  createdAt: timestamp("created_at", { withTimezone: true })
    .notNull()
    .default(sql`now()`),
});

export const accounts = pgTable(
  "accounts",
  {
    userId: uuid("user_id")
      .notNull()
      .references(() => users.id, { onDelete: "cascade" }),
    type: text("type").$type<AdapterAccountType>().notNull(),
    provider: text("provider").notNull(),
    providerAccountId: text("provider_account_id").notNull(),
    refresh_token: text("refresh_token"),
    access_token: text("access_token"),
    expires_at: integer("expires_at"),
    token_type: text("token_type"),
    scope: text("scope"),
    id_token: text("id_token"),
    session_state: text("session_state"),
  },
  (t) => [primaryKey({ columns: [t.provider, t.providerAccountId] })],
);

export const sessions = pgTable("sessions", {
  sessionToken: text("session_token").primaryKey(),
  userId: uuid("user_id")
    .notNull()
    .references(() => users.id, { onDelete: "cascade" }),
  expires: timestamp("expires", { withTimezone: true, mode: "date" }).notNull(),
});

export const verificationTokens = pgTable(
  "verification_tokens",
  {
    identifier: text("identifier").notNull(),
    token: text("token").notNull(),
    expires: timestamp("expires", {
      withTimezone: true,
      mode: "date",
    }).notNull(),
  },
  (t) => [primaryKey({ columns: [t.identifier, t.token] })],
);

// ---------------------------------------------------------------------------
// Domain tables (SPEC §8).
// ---------------------------------------------------------------------------

export const experiments = pgTable(
  "experiments",
  {
    id: uuid("id").defaultRandom().primaryKey(),
    /** Human-readable slug suffixed with a nanoid for uniqueness. */
    slug: text("slug").notNull(),
    ownerId: uuid("owner_id")
      .notNull()
      .references(() => users.id, { onDelete: "cascade" }),
    name: text("name").notNull(),
    /** Enforced in Zod at the API boundary (SPEC §7) — Postgres lets us
     * widen the values without a migration if the spec evolves. */
    visibility: text("visibility").notNull().default("private"),
    inputText: text("input_text").notNull().default(""),
    configJson: jsonb("config_json").notNull(),
    layoutJson: jsonb("layout_json").notNull(),
    forkedFrom: uuid("forked_from"),
    viewCount: integer("view_count").notNull().default(0),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .default(sql`now()`),
    updatedAt: timestamp("updated_at", { withTimezone: true })
      .notNull()
      .default(sql`now()`),
  },
  (t) => [
    uniqueIndex("experiments_slug_idx").on(t.slug),
    index("experiments_owner_idx").on(t.ownerId),
    index("experiments_public_idx").on(t.visibility, t.createdAt),
  ],
);

export const comments = pgTable(
  "comments",
  {
    id: uuid("id").defaultRandom().primaryKey(),
    sectionSlug: text("section_slug").notNull(),
    userId: uuid("user_id")
      .notNull()
      .references(() => users.id, { onDelete: "cascade" }),
    /** One-level threading; null = top-level. (SPEC §6.4) */
    parentId: uuid("parent_id"),
    bodyMd: text("body_md").notNull(),
    hidden: boolean("hidden").notNull().default(false),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .default(sql`now()`),
  },
  (t) => [index("comments_section_idx").on(t.sectionSlug, t.createdAt)],
);

export const progress = pgTable(
  "progress",
  {
    userId: uuid("user_id")
      .notNull()
      .references(() => users.id, { onDelete: "cascade" }),
    sectionSlug: text("section_slug").notNull(),
    /** 'not_started' | 'in_progress' | 'completed' — validated at API edge. */
    status: text("status").notNull().default("not_started"),
    updatedAt: timestamp("updated_at", { withTimezone: true })
      .notNull()
      .default(sql`now()`),
  },
  (t) => [primaryKey({ columns: [t.userId, t.sectionSlug] })],
);

export const events = pgTable(
  "events",
  {
    id: bigserial("id", { mode: "bigint" }).primaryKey(),
    userId: uuid("user_id"), // null = anonymous; no FK so anon events survive user deletion
    sessionId: text("session_id").notNull(),
    /** 'page_view' | 'widget_interact' | … — free-form, validated at the edge. */
    kind: text("kind").notNull(),
    sectionSlug: text("section_slug"),
    metaJson: jsonb("meta_json"),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .default(sql`now()`),
  },
  (t) => [
    index("events_created_idx").on(t.createdAt),
    index("events_kind_idx").on(t.kind, t.createdAt),
  ],
);

// ---------------------------------------------------------------------------
// Inferred types — re-exported so the rest of the codebase doesn't import
// drizzle-orm's `InferSelectModel` directly.
// ---------------------------------------------------------------------------

export type User = typeof users.$inferSelect;
export type NewUser = typeof users.$inferInsert;
export type Experiment = typeof experiments.$inferSelect;
export type NewExperiment = typeof experiments.$inferInsert;
export type Comment = typeof comments.$inferSelect;
export type NewComment = typeof comments.$inferInsert;
export type Progress = typeof progress.$inferSelect;
export type NewProgress = typeof progress.$inferInsert;
export type Event = typeof events.$inferSelect;
export type NewEvent = typeof events.$inferInsert;
