"""
precision.py — low-precision training, emulated in float64 (chapter 9).

The number formats are vendored from the Numerics site
(github.com/BrendanJamesLynskey/numerics-explained, reference/numerics.py at
commit ca9bbd9, MIT licence, same author): `encode`, `decode` and
`round_to` below are its bit-exact scalar code for FP32, FP16, BF16 and the
two OCP FP8 formats, unchanged apart from dropping the formats and rounding
modes this site does not use. `round_array` is a vectorised NumPy version
of the same round-to-nearest-even, checked against the scalar code on every
FP8 value, the midpoints between them and random values in every format
(checks.py:check_rounding).

The emulation (a `Cast`, consumed by model.forward/backward):
  - each weight matrix is rounded to the compute format before the forward
    pass (the "compute copy"); the master weights stay in float64 (standing
    in for FP32);
  - each activation a matrix product reads or writes is rounded to the
    format (accumulation inside the product is exact, as tensor cores
    accumulate in FP32); softmax, LayerNorm, GELU and the loss run in
    float64 (as autocast runs them in FP32);
  - each activation gradient a product reads or writes, and each weight
    gradient, is rounded to the gradient format;
  - for FP8, only what the products read is FP8 (with per-tensor scaling);
    what they write is BF16, as in the FP8 training recipes.
FP8 uses per-tensor scaling: x is multiplied by s = max_format / amax(x)
before rounding and divided by it after (the "current scaling" of the FP8
recipes; Micikevicius et al. 2022, arXiv:2209.05433), E4M3 forward, E5M2
for gradients. FP16 uses loss scaling (Micikevicius et al. 2018,
arXiv:1710.03740), done by the training loop (experiments.py).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Vendored from numerics-explained/reference/numerics.py (ca9bbd9).
# ---------------------------------------------------------------------------
FORMATS: dict[str, dict[str, Any]] = {
    "fp32": {"id": "fp32", "name": "FP32", "e": 8, "m": 23, "bias": 127, "kind": "ieee"},
    "fp16": {"id": "fp16", "name": "FP16", "e": 5, "m": 10, "bias": 15, "kind": "ieee"},
    "bf16": {"id": "bf16", "name": "BF16", "e": 8, "m": 7, "bias": 127, "kind": "ieee"},
    "e5m2": {"id": "e5m2", "name": "FP8 E5M2", "e": 5, "m": 2, "bias": 15, "kind": "ieee"},
    "e4m3": {"id": "e4m3", "name": "FP8 E4M3", "e": 4, "m": 3, "bias": 7, "kind": "fn"},
}
FORMAT_ORDER = ("fp32", "bf16", "fp16", "e5m2", "e4m3")


def pow2(k: int) -> float:
    """2**k as a double, exactly."""
    return math.ldexp(1.0, k)


def exponent_of(a: float) -> int:
    """floor(log2(a)) for a finite a > 0, exactly."""
    return math.frexp(a)[1] - 1


def info(fid: str) -> dict[str, Any]:
    """A format's derived constants (all exact)."""
    f = FORMATS[fid]
    e, m, bias, kind = f["e"], f["m"], f["bias"], f["kind"]
    top = (1 << e) - 1
    emin = 1 - bias
    if kind == "ieee":
        emax = top - 1 - bias
        max_mant = (1 << m) - 1
        inf_code = top << m
        nan_code = (top << m) | (1 << (m - 1))
    else:  # "fn": no infinities, NaN only at S.1111.111 (OCP E4M3)
        emax = top - bias
        max_mant = (1 << m) - 2
        inf_code = -1
        nan_code = (top << m) | ((1 << m) - 1)
    max_code = ((emax + bias) << m) | max_mant
    return {
        "id": fid, "bits": 1 + e + m, "e": e, "m": m, "bias": bias, "kind": kind,
        "emin": emin, "emax": emax,
        "max": pow2(emax) * (1 + max_mant / (1 << m)),
        "min_normal": pow2(emin), "min_sub": pow2(emin - m), "eps": pow2(-m),
        "max_code": max_code, "inf_code": inf_code, "nan_code": nan_code,
        "sign_bit": 1 << (e + m),
    }


INFO = {fid: info(fid) for fid in FORMAT_ORDER}


def decode(code: int, fid: str) -> float:
    """The value of a code (NaN for NaN codes; signed zeros and infinities)."""
    f = INFO[fid]
    e, m, bias, kind = f["e"], f["m"], f["bias"], f["kind"]
    neg = (code >> (e + m)) & 1
    E = (code >> m) & ((1 << e) - 1)
    M = code & ((1 << m) - 1)
    top = (1 << e) - 1
    if kind == "ieee" and E == top:
        if M == 0:
            return -math.inf if neg else math.inf
        return math.nan
    if kind == "fn" and E == top and M == (1 << m) - 1:
        return math.nan
    if E == 0:
        v = pow2(f["emin"] - m) * M
    else:
        v = pow2(E - bias - m) * ((1 << m) + M)
    return -v if neg else v


def _magnitude_code(v: float, f: dict[str, Any]) -> int:
    m = f["m"]
    if v < f["min_normal"]:
        return int(v / f["min_sub"])
    E = exponent_of(v)
    return ((E + f["bias"]) << m) | (int(v / pow2(E - m)) - (1 << m))


def encode(x: float, fid: str, sat: bool = False) -> int:
    """Round the double x to the format (round to nearest, ties to even) and
    return its code. `sat` saturates on overflow (OCP's SAT mode)."""
    f = INFO[fid]
    s = f["sign_bit"]
    neg = math.copysign(1.0, x) < 0
    sign = s if neg else 0
    if x != x:
        return sign | f["nan_code"]
    a = abs(x)
    if a == math.inf:
        if not sat and f["kind"] == "ieee":
            return sign | f["inf_code"]
        if not sat:
            return sign | f["nan_code"]
        return sign | f["max_code"]
    if a == 0:
        return sign
    ee = max(exponent_of(a), f["emin"])
    q = pow2(ee - f["m"])
    t = a / q
    n = math.floor(t)
    frac = t - n
    if frac > 0.5 or (frac == 0.5 and n % 2 == 1):
        n += 1
    if n == 0:
        return sign
    v = n * q
    if v > f["max"]:
        if sat:
            return sign | f["max_code"]
        return sign | (f["inf_code"] if f["kind"] == "ieee" else f["nan_code"])
    return sign | _magnitude_code(v, f)


def round_to(x: float, fid: str, sat: bool = False) -> float:
    """x rounded to the format, as a double."""
    return decode(encode(x, fid, sat), fid)


def positive_values(fid: str) -> list[float]:
    """Every finite value >= 0, in order (8-bit formats)."""
    f = INFO[fid]
    return [decode(c, fid) for c in range(f["max_code"] + 1)]


# ---------------------------------------------------------------------------
# Vectorised rounding (checked against round_to by checks.py)
# ---------------------------------------------------------------------------
def round_array(x: np.ndarray, fid: str, sat: bool = False) -> np.ndarray:
    """Round every element of x to the format, ties to even, as float64.

    The same steps as `encode`: q = 2^(max(floor(log2 |x|), emin) - m) is
    the spacing of the format at |x|; |x| / q is exact, np.rint rounds it to
    the nearest integer with ties to even; overflow gives inf (IEEE formats),
    NaN (E4M3) or the largest value (sat)."""
    f = INFO[fid]
    x = np.asarray(x, dtype=np.float64)
    a = np.abs(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        _, e2 = np.frexp(a)  # a = mant * 2^e2, mant in [0.5, 1)
        ee = np.maximum(e2 - 1, f["emin"])
        q = np.ldexp(1.0, ee - f["m"])
        v = np.rint(a / q) * q
    big = v > f["max"]
    if sat:
        v = np.where(big, f["max"], v)
    else:
        v = np.where(big, np.inf if f["kind"] == "ieee" else np.nan, v)
    v = np.where(np.isinf(a), (f["max"] if sat else (np.inf if f["kind"] == "ieee" else np.nan)), v)
    v = np.where(np.isnan(a), np.nan, v)
    return np.copysign(v, x)


def scaled_round(x: np.ndarray, fid: str) -> np.ndarray:
    """Per-tensor scaling: round x * s to the format (saturating), divide by s,
    with s = max_format / amax(x) rounded down to a power of two (so the
    scaling itself is exact)."""
    amax = float(np.abs(x).max()) if x.size else 0.0
    if amax == 0.0 or not math.isfinite(amax):
        return round_array(x, fid, sat=True)
    s = pow2(math.floor(math.log2(INFO[fid]["max"] / amax)))
    return round_array(x * s, fid, sat=True) / s


# ---------------------------------------------------------------------------
# Casts for model.forward / model.backward
# ---------------------------------------------------------------------------
class Cast:
    """Round what the matrix products read: weights' compute copies and
    activations to `fwd`, activation gradients to `bwd`; and what they
    write: activations to `out` and gradients (and weight gradients) to
    `out_bwd` (default: the same formats). None leaves float64. `scaled`:
    per-tensor scaling of what the products read (the FP8 recipes; their
    outputs are BF16, unscaled)."""

    def __init__(self, fwd: str | None, bwd: str | None, scaled: bool = False,
                 out: str | None = "same", out_bwd: str | None = "same"):
        self.fwd, self.bwd, self.scaled = fwd, bwd, scaled
        self.out = fwd if out == "same" else out
        self.out_bwd = bwd if out_bwd == "same" else out_bwd
        self.counts = {"g_total": 0, "g_zero": 0, "g_inf": 0}

    def _r(self, x, fid, scaled):
        if fid is None:
            return x
        if scaled:
            return scaled_round(x, fid)
        return round_array(x, fid)

    def _count(self, x, y, fid):
        if fid is not None:
            self.counts["g_total"] += x.size
            self.counts["g_zero"] += int(((y == 0) & (x != 0)).sum())
            self.counts["g_inf"] += int((~np.isfinite(y)).sum())
        return y

    def w(self, W):
        return self._r(W, self.fwd, self.scaled)

    def a(self, x):
        return self._r(x, self.fwd, self.scaled)

    def o(self, x):
        return self._r(x, self.out, False)

    def g(self, x):
        return self._count(x, self._r(x, self.bwd, self.scaled), self.bwd)

    def og(self, x):
        return self._count(x, self._r(x, self.out_bwd, False), self.out_bwd)

    def pg(self, x):
        return self._count(x, self._r(x, self.out_bwd, False), self.out_bwd)


def quantise_int(x: np.ndarray, bits: int) -> np.ndarray:
    """Symmetric absmax integer quantisation of a whole tensor (as the
    Numerics site's int_params(..., "sym") with one scale): q = rne(x / s),
    s = amax / (2^(bits-1) - 1), values in [-(2^(bits-1) - 1), 2^(bits-1) - 1];
    returns q * s. The 'full scale' is amax: nothing clips."""
    amax = float(np.abs(x).max()) if x.size else 0.0
    if amax == 0.0:
        return x.copy()
    levels = (1 << (bits - 1)) - 1
    s = amax / levels
    return np.rint(x / s) * s
