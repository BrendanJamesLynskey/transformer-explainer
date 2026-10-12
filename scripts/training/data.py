"""
data.py — the corpus, the batches and the BPE tokeniser for Part 2, the
reference for src/lib/training/data.ts and bpe.ts.

Batches: each training step draws B windows of S + 1 characters from the
training split at starts floor(u * (N - S - 1)), u from mulberry32(data
seed); the first S characters are the input, the last S the targets
(teacher forcing: every position predicts the character after it).

Validation: the last 10% of the corpus, never trained on, scored on 64
fixed, evenly spaced windows.

BPE (Sennrich et al., arXiv:1508.07909; byte-level in GPT-2): start from
the 64 characters, count adjacent pairs inside words (a word is an optional
leading space and a run of non-space characters, or a lone space or
newline), merge the most frequent pair into a new token (ties: the smallest
pair of ids), and repeat. Encoding applies the merges in the order learnt.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from model import ALPHABET, CHAR_TO_ID, encode_chars

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "public" / "training" / "corpus.txt"
VAL_FRACTION = 0.1
VAL_WINDOWS = 64


def load_corpus() -> str:
    return CORPUS.read_text(encoding="utf-8")


def split(ids: list[int]) -> tuple[list[int], list[int]]:
    """Train on the first 90%, validate on the last 10%."""
    cut = len(ids) - math.floor(len(ids) * VAL_FRACTION)
    return ids[:cut], ids[cut:]


def sample_batch(train: list[int], B: int, S: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """B random windows of S + 1 tokens: inputs [B,S], targets [B,S]."""
    rows = []
    for _ in range(B):
        start = math.floor(rng() * (len(train) - S - 1))
        rows.append(train[start : start + S + 1])
    a = np.array(rows, dtype=np.int64)
    return a[:, :-1], a[:, 1:]


def val_batch(val: list[int], S: int, n: int = VAL_WINDOWS) -> tuple[np.ndarray, np.ndarray]:
    """n evenly spaced windows of the validation split (fixed for a run)."""
    span = len(val) - S - 1
    rows = [val[(k * span) // n : (k * span) // n + S + 1] for k in range(n)]
    a = np.array(rows, dtype=np.int64)
    return a[:, :-1], a[:, 1:]


# ---------------------------------------------------------------------------
# BPE
# ---------------------------------------------------------------------------
WORD_RE = re.compile(r" ?[^ \n]+|\n| ")


def words_of(text: str) -> list[str]:
    return WORD_RE.findall(text)


def word_counts(text: str) -> dict[tuple[int, ...], int]:
    counts: dict[tuple[int, ...], int] = {}
    for w in words_of(text):
        key = tuple(encode_chars(w))
        counts[key] = counts.get(key, 0) + 1
    return counts


def merge_word(word: tuple[int, ...], a: int, b: int, new: int) -> tuple[int, ...]:
    out, i = [], 0
    while i < len(word):
        if i + 1 < len(word) and word[i] == a and word[i + 1] == b:
            out.append(new)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def pair_counts(counts: dict) -> dict[tuple[int, int], int]:
    pairs: dict[tuple[int, int], int] = {}
    for word, n in counts.items():
        for i in range(len(word) - 1):
            k = (word[i], word[i + 1])
            pairs[k] = pairs.get(k, 0) + n
    return pairs


def best_pairs(pairs: dict, k: int) -> list[tuple[tuple[int, int], int]]:
    """The k most frequent pairs; ties broken by the smaller pair of ids."""
    return sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


def bpe_train(text: str, n_merges: int, top: int = 5) -> dict:
    """Learn n_merges merges. Also records, per merge, the top pairs and the token count."""
    counts = word_counts(text)
    vocab = list(ALPHABET)
    merges, history = [], []
    total = sum(len(w) * n for w, n in counts.items())
    for step in range(n_merges):
        pairs = pair_counts(counts)
        if not pairs:
            break
        ranked = best_pairs(pairs, top)
        (a, b), n = ranked[0]
        new = len(vocab)
        vocab.append(vocab[a] + vocab[b])
        merges.append([a, b])
        counts_new: dict = {}
        for w, c in counts.items():
            m = merge_word(w, a, b, new)
            counts_new[m] = counts_new.get(m, 0) + c
        counts = counts_new
        before = total
        total = sum(len(w) * c for w, c in counts.items())
        history.append({"step": step, "pair": [a, b], "count": n, "new": new, "token": vocab[new],
                        "top": [{"pair": list(p), "count": c} for p, c in ranked],
                        "tokensBefore": before, "tokensAfter": total})
    return {"vocab": vocab, "merges": merges, "history": history}


def bpe_encode(text: str, merges: list[list[int]]) -> list[int]:
    out: list[int] = []
    cache: dict[str, tuple[int, ...]] = {}
    for w in words_of(text):
        if w not in cache:
            t = tuple(encode_chars(w))
            for k, (a, b) in enumerate(merges):
                t = merge_word(t, a, b, 64 + k)
            cache[w] = t
        out.extend(cache[w])
    return out
