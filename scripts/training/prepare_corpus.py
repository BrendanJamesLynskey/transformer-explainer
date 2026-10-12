"""
prepare_corpus.py — build the training corpus for Part 2 (Training).

The corpus is the text of Lewis Carroll's "Alice's Adventures in Wonderland"
(1865), Project Gutenberg eBook #11, which is in the public domain. Only the
book's text is kept: the Project Gutenberg header, footer and licence are
removed, so the result carries no Project Gutenberg trademark (see
public/training/CORPUS.md for the licence note and the source's SHA-256).

The text is reduced to the site's 64-character alphabet
(src/lib/transformer/tokenizer.ts), because Part 2 trains Part 1's model
and that model reads characters from that alphabet:

  - lower case;
  - curly quotes to straight quotes, dashes to '-';
  - '_' (Gutenberg's italics marks), '*' (section breaks) and
    "[Illustration]" lines removed;
  - lines inside a paragraph joined with spaces; one '\n' between
    paragraphs; runs of spaces collapsed.

USAGE
    curl -sL https://gutenberg.pglaf.org/1/11/11-0.txt -o pg11-0.txt
    python scripts/training/prepare_corpus.py pg11-0.txt

It refuses a source whose SHA-256 differs from the one it was built from,
so the committed corpus can always be rebuilt byte for byte.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "public" / "training" / "corpus.txt"
SOURCE_SHA256 = "a3a27f8edbf7fcd9b8ba8435494440e24952deaa3e2f2d65192d4cb7ca403754"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import CHAR_TO_ID  # noqa: E402


def normalise(raw: str) -> str:
    start = raw.index("CHAPTER I.\n")
    # the table of contents also lists "CHAPTER I." with spaces after it;
    # the first heading on a line of its own is the start of the text
    end = raw.index("*** END OF THE PROJECT GUTENBERG EBOOK")
    body = raw[start:end].replace("\r\n", "\n")
    for a, b in (("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'"), ("—", "-"), ("–", "-")):
        body = body.replace(a, b)
    body = body.replace("_", "")
    paras = []
    for block in re.split(r"\n\s*\n", body):
        lines = [ln.strip() for ln in block.split("\n")]
        lines = [ln for ln in lines if ln and ln != "[Illustration]" and set(ln) - {"*", " "}]
        if not lines:
            continue
        p = " ".join(lines).lower()
        p = "".join(c for c in p if c in CHAR_TO_ID and c != "\n")
        p = re.sub(r" +", " ", p).strip()
        if p:
            paras.append(p)
    return "\n".join(paras) + "\n"


def main() -> None:
    src = Path(sys.argv[1])
    data = src.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SOURCE_SHA256:
        sys.exit(f"source SHA-256 {digest} != expected {SOURCE_SHA256}")
    text = normalise(data.decode("utf-8"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}: {len(text)} characters, sha256 {hashlib.sha256(text.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
