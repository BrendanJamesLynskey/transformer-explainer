/** The training corpus as the trainer reads it (public/training/corpus.txt). */
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { encodeChars } from "@/lib/training/data";

export const CORPUS_TEXT = readFileSync(
  join(__dirname, "..", "..", "..", "public", "training", "corpus.txt"),
  "utf8",
);
export const CORPUS_IDS = encodeChars(CORPUS_TEXT);
