/**
 * Slug helpers.
 *
 * `slugify` lowercases, ASCII-folds where safe, and dasherises whitespace
 * + punctuation. `experimentSlug` appends a 6-char nanoid suffix so two
 * experiments with the same name don't collide.
 */
import { customAlphabet } from "nanoid";

const NANOID_SUFFIX = customAlphabet("abcdefghijklmnopqrstuvwxyz0123456789", 6);

/**
 * Lowercase, replace anything non-alphanumeric with a dash, collapse
 * runs of dashes, trim leading/trailing dashes. Empty / weird inputs
 * fall back to "experiment".
 */
export function slugify(name: string): string {
  const base = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 32);
  return base || "experiment";
}

export function experimentSlug(name: string): string {
  return `${slugify(name)}-${NANOID_SUFFIX()}`;
}
