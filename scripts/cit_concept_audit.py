#!/usr/bin/env python3
"""
cit_concept_audit.py — does Part 2 (Training) explain every training concept
the Compute in Transit results site relies on? (brief 29 §2)

For each term in cit_terms.py: its chapter must be a Training chapter, and,
once that chapter's MDX has shipped (content/training/<slug>.mdx), the MDX
must hold its anchor (`<span id="t-…" />`) at the start of a paragraph that
names the term (one of its aliases) and explains it (at least 150 characters
of prose). Verdicts:

  explained   the anchored paragraph names and explains the term
  MISSING     the chapter has shipped but the anchor or the explanation is not there
  pending     the chapter has not shipped yet (Part 2 ships in stages: 29A
              chapters 1-5, 29B 6-10, 29C 11-16)

    python3 scripts/cit_concept_audit.py             # CI: fails on MISSING
    python3 scripts/cit_concept_audit.py --strict    # also fails on pending (from 29C)
    python3 scripts/cit_concept_audit.py --refresh   # also re-reads the live results
                                                     # site and checks every evidence
                                                     # phrase is still on its page
    python3 scripts/cit_concept_audit.py --out FILE  # write the table as Markdown

CI runs it offline (the term list is committed; --refresh is for people).
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from cit_terms import TERMS  # noqa: E402

CONTENT = ROOT / "content" / "training"
SECTIONS_TS = ROOT / "src" / "lib" / "mdx" / "sections.ts"
SITE = "https://compute-in-transit-results.vercel.app"
PAGES = {"home": "/", "method": "/method", "experiment": "/experiment", "transit": "/transit",
         "optics": "/optics", "energy": "/energy", "notebook": "/notebook"}
MIN_EXPLANATION = 150


def training_slugs() -> list[str]:
    """The Training chapter slugs, read from TRAINING_SECTIONS in sections.ts."""
    src = SECTIONS_TS.read_text()
    block = src[src.index("export const TRAINING_SECTIONS"):]
    block = block[: block.index("] as const;")]
    return re.findall(r'slug: "([^"]+)"', block)


def paragraph_at(mdx: str, anchor: str) -> str | None:
    """The paragraph that starts with <span id="anchor" />, as plain-ish text."""
    m = re.search(r'<span id="' + re.escape(anchor) + r'"\s*/>', mdx)
    if not m:
        return None
    rest = mdx[m.end():]
    para = re.split(r"\n\s*\n", rest, maxsplit=1)[0]
    text = re.sub(r"<[^>]+>", " ", para)          # JSX tags
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links
    return re.sub(r"\s+", " ", text).strip()


def audit() -> list[dict]:
    slugs = training_slugs()
    rows = []
    for key, name, chapter, anchor, aliases, (page, evidence) in TERMS:
        row = {"key": key, "name": name, "chapter": chapter, "anchor": anchor,
               "page": page, "evidence": evidence}
        if chapter not in slugs:
            row.update(verdict="MISSING", detail=f"no Training chapter '{chapter}'")
        elif not (CONTENT / f"{chapter}.mdx").exists():
            row.update(verdict="pending", detail="chapter not shipped yet")
        else:
            para = paragraph_at((CONTENT / f"{chapter}.mdx").read_text(), anchor)
            if para is None:
                row.update(verdict="MISSING", detail=f'no <span id="{anchor}" /> in {chapter}.mdx')
            elif not any(re.search(a, para, re.I) for a in aliases):
                row.update(verdict="MISSING", detail=f"the anchored paragraph does not name it ({aliases[0]})")
            elif len(para) < MIN_EXPLANATION:
                row.update(verdict="MISSING", detail=f"only {len(para)} characters: a mention, not an explanation")
            else:
                hit = next(re.search(a, para, re.I) for a in aliases if re.search(a, para, re.I))
                s = max(0, hit.start() - 40)
                row.update(verdict="explained", detail="…" + para[s: s + 110] + "…")
        rows.append(row)
    return rows


def refresh(rows: list[dict]) -> list[str]:
    """Re-read the live results site; every evidence phrase must still be on its page."""
    texts = {}
    for page, path in PAGES.items():
        req = urllib.request.Request(SITE + path, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=60).read().decode("utf-8")
        raw = re.sub(r"<script.*?</script>|<style.*?</style>", " ", raw, flags=re.S)
        texts[page] = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw)))
    gone = []
    for r in rows:
        if r["evidence"] not in texts[r["page"]]:
            gone.append(f"{r['key']}: '{r['evidence']}' is no longer on {PAGES[r['page']]}")
    return gone


def table(rows: list[dict]) -> str:
    out = ["| Concept (results-site page) | Chapter · anchor | Verdict | Detail |", "|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['name']} ({PAGES[r['page']]}) | `{r['chapter']}#{r['anchor']}` | "
                   f"{r['verdict']} | {r['detail'].replace('|', '/')} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    rows = audit()
    print(table(rows))
    counts = {v: sum(r["verdict"] == v for r in rows) for v in ("explained", "pending", "MISSING")}
    print(f"\n{len(rows)} concepts: {counts['explained']} explained, {counts['pending']} pending, "
          f"{counts['MISSING']} missing")
    if args.out:
        Path(args.out).write_text(table(rows) + "\n")
    bad = counts["MISSING"] > 0 or (args.strict and counts["pending"] > 0)
    if args.refresh:
        gone = refresh(rows)
        print("\n".join(gone) if gone else "refresh: every evidence phrase is still on the live site")
        bad = bad or bool(gone)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
