# Training corpus

`corpus.txt` is the text of **Alice's Adventures in Wonderland** by Lewis
Carroll (1865), from Project Gutenberg eBook #11
(<https://www.gutenberg.org/ebooks/11>, file `11-0.txt`, fetched from the
mirror `https://gutenberg.pglaf.org/1/11/11-0.txt` on 2026-10-12; SHA-256 of
the downloaded file `a3a27f8edbf7fcd9b8ba8435494440e24952deaa3e2f2d65192d4cb7ca403754`).

**Licence.** The book is in the public domain (published 1865; the author
died in 1898). Only the book's text is used: the Project Gutenberg header,
footer and licence were removed, so, as the Project Gutenberg licence
provides, the text is not distributed under the Project Gutenberg trademark
and no Project Gutenberg terms apply to it.

**Changes.** `scripts/training/prepare_corpus.py` builds `corpus.txt` from the
downloaded file: lower case; curly quotes to straight quotes and dashes to
`-`; Gutenberg's italics marks (`_`), section breaks and `[Illustration]`
lines removed; each paragraph on one line. Every character is in the site's
64-character alphabet. The result is 142,041 characters, SHA-256
`8a2b46fd9524cc572aecb9b8293d8143a9ebc3c6e816601d1e11e3f8937d0178`
(also in results.md §1).

`default-run-weights.json` is the model's weights after the default training
run (results.md §4), written by `scripts/training/make_fixtures.py`; MIT, as
the rest of the repository.
