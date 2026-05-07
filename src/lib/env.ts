/**
 * Centralised, Zod-validated environment access.
 *
 * Per CLAUDE.md §10, this is the *only* place in the codebase that reads
 * `process.env`. Every other file imports the typed `env` object from here.
 * The Zod schema validates at module load time and throws a clear error if
 * anything is missing, so a bad deploy fails fast at boot rather than at the
 * first request that touches a missing variable.
 *
 * The schema mirrors `.env.example`. When you add a new variable:
 *   1. Add it to `.env.example` with a comment.
 *   2. Add it to the Zod schema below.
 *   3. Use it via `import { env } from "@/lib/env";` — never `process.env.X`.
 */
import { z } from "zod";

const isProd = process.env.NODE_ENV === "production";

/**
 * In production, server-only secrets MUST be set. In development and CI we
 * tolerate placeholder values to keep the loop fast — secrets are only
 * exercised by the auth and DB code paths, both of which have their own
 * narrower checks.
 */
const requiredInProd = (label: string) =>
  z
    .string()
    .min(1, `${label} is required in production`)
    .or(z.literal("").transform(() => undefined))
    .optional()
    .superRefine((val, ctx) => {
      if (isProd && !val) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: `${label} is required in production`,
        });
      }
    });

const envSchema = z.object({
  NODE_ENV: z
    .enum(["development", "test", "production"])
    .default("development"),

  // --- Database ----------------------------------------------------------
  DATABASE_URL: z
    .string()
    .url()
    .refine(
      (s) => s.startsWith("postgres://") || s.startsWith("postgresql://"),
      { message: "DATABASE_URL must be a postgres:// or postgresql:// URL" },
    )
    .optional()
    .superRefine((val, ctx) => {
      if (isProd && !val) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: "DATABASE_URL is required in production",
        });
      }
    }),

  // --- Auth.js -----------------------------------------------------------
  AUTH_SECRET: requiredInProd("AUTH_SECRET"),
  AUTH_GITHUB_ID: requiredInProd("AUTH_GITHUB_ID"),
  AUTH_GITHUB_SECRET: requiredInProd("AUTH_GITHUB_SECRET"),

  // --- App ---------------------------------------------------------------
  NEXT_PUBLIC_SITE_URL: z.string().url().default("http://localhost:3000"),

  // Comma-separated list of GitHub logins granted access to /admin.
  // Stored as the raw string here; consumers split on the comma.
  ADMIN_GITHUB_LOGINS: z.string().default(""),

  // --- Compute limits (SPEC §4) -----------------------------------------
  // Coerce because env vars are strings; .pipe() keeps the parsed shape clean.
  MAX_SEQ_LEN: z.coerce.number().int().positive().max(64).default(16),
  MAX_D_MODEL: z.coerce.number().int().positive().max(256).default(64),
  MAX_BLOCKS: z.coerce.number().int().positive().max(8).default(4),

  // --- Optional ----------------------------------------------------------
  NEXT_PUBLIC_DEV_DEMOS_ENABLED: z
    .enum(["true", "false"])
    .default("false")
    .transform((v) => v === "true"),
});

const parsed = envSchema.safeParse(process.env);

if (!parsed.success) {
  // Pretty-print so the failure mode is obvious in dev and in deploy logs.
  console.error("\n❌ Invalid environment variables:");
  for (const issue of parsed.error.issues) {
    console.error(`  - ${issue.path.join(".") || "(root)"}: ${issue.message}`);
  }
  console.error("");
  throw new Error("Invalid environment configuration. See .env.example.");
}

/**
 * Typed environment object. Import this everywhere instead of `process.env`.
 *
 * Note: only variables prefixed with `NEXT_PUBLIC_` are inlined into the
 * client bundle. Anything else is server-only.
 */
export const env = parsed.data;

/** Comma-split admin login allow-list, normalised to lowercase. */
export const adminLogins: ReadonlySet<string> = new Set(
  env.ADMIN_GITHUB_LOGINS.split(",")
    .map((s) => s.trim().toLowerCase())
    .filter(Boolean),
);
