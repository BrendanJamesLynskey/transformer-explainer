"""
anim_c.py — the state lists behind chapters 11-16's animations (brief
29C), the references their TypeScript builders are frame-tested against
(tests/unit/fixtures/training/anim_*.json):

  ring       chapter 11: a ring all-reduce of four workers' real gradients
             (the trained model's final LayerNorm gain, each worker's own
             four windows), chunk by chunk (systems.ring_states)
  pipeline   chapter 12: the GPipe, 1F1B and interleaved schedules for 4
             devices and 8 micro-batches (systems.pipeline)
  demo       chapter 13: two DeMo syncs of four workers from the trained
             model (slowlinks.sync_states)
  isoflop    chapter 14: the measured IsoFLOP points, a budget at a time,
             with each budget's parabola and the frontier through them
  goodput    chapter 15: a week of a job with failures and checkpoints
             (systems.simulate_goodput)
  sft        chapter 16: the trained model's loss on a prompt and response,
             position by position, with the prompt masked out, and the log
             probabilities of two continuations
"""

from __future__ import annotations

import math

import numpy as np

import data as D
import isoflop as I
import model as M
import slowlinks as SL
import systems as SY

PART1 = dict(M.PART1_CONFIG)

# chapter 11 ----------------------------------------------------------------
RING_TENSOR = "ln_final.gamma"
RING_SEED = 11
RING_WORKERS = 4


def worker_gradients(trained: dict, ids, tensor: str, seed: int, workers: int, per: int = 4):
    train, _ = D.split(ids)
    x, y = D.sample_batch(train, workers * per, PART1["seq_len"], M.mulberry32(seed))
    out = []
    for w in range(workers):
        _, g, _ = M.loss_and_grads(trained, PART1, x[w * per:(w + 1) * per], y[w * per:(w + 1) * per])
        out.append(g[tensor].reshape(-1).tolist())
    return out


def ring_fixture(trained: dict, ids) -> dict:
    grads = worker_gradients(trained, ids, RING_TENSOR, RING_SEED, RING_WORKERS)
    states = SY.ring_states(grads)
    total = [sum(g[i] for g in grads) for i in range(len(grads[0]))]
    return {"tensor": RING_TENSOR, "seed": RING_SEED, "grads": grads, "states": states, "sum": total}


# chapter 12 ----------------------------------------------------------------
PIPE = dict(p=4, m=8, v=2)


def pipeline_fixture() -> dict:
    out = {}
    for kind in ("gpipe", "1f1b", "interleaved"):
        r = SY.pipeline(kind, PIPE["p"], PIPE["m"], PIPE["v"])
        out[kind] = r
    return out


# chapter 14 ----------------------------------------------------------------
def isoflop_states(iso: dict) -> list[dict]:
    """One state per measured point (budget by budget, small model first),
    then each budget's fit, then the frontier."""
    states = []
    shown: list[dict] = []
    for bi, row in enumerate(iso["rows"]):
        for p in row["points"]:
            if p["mean"] is None:
                continue
            shown = shown + [{"budget": bi, "params": p["params"], "loss": p["mean"]}]
            states.append({"kind": "point", "budget": bi, "params": p["params"], "loss": p["mean"],
                           "shown": len(shown)})
        near = I.fit_points([p for p in row["points"] if p["mean"] is not None])
        if near is None:
            states.append({"kind": "nofit", "budget": bi, "shown": len(shown)})
            continue
        fit = I.fit_parabola([math.log10(p["params"]) for p in near], [p["mean"] for p in near])
        states.append({"kind": "fit", "budget": bi, "c": fit["c"],
                       "nOpt": 10 ** fit["xMin"] if fit["xMin"] is not None else None,
                       "lossOpt": fit["yMin"], "fitParams": [p["params"] for p in near], "shown": len(shown)})
    states.append({"kind": "frontier", "a": iso["a"], "intercept": iso["frontier"]["intercept"],
                   "shown": len(shown)})
    return states


# chapter 15 ----------------------------------------------------------------
# A week of a job the size of Llama 3 405B's (Grattafiori et al. 2024 §3.3.4:
# 419 unexpected interruptions in 54 days, so a mean time between failures of
# 54 days / 419 = 3.1 hours), checkpointing every hour; the checkpoint and
# restart costs are illustrative (see results.md §15).
GOODPUT = dict(mtbf=54 * 24 * 3600 / 419, delta=16384 * 4e9 / 2e12, restart=600.0,
               horizon=7 * 24 * 3600.0, tau=3600.0, seed=3)


def goodput_fixture() -> dict:
    g = GOODPUT
    sim = SY.simulate_goodput(g["tau"], g["delta"], g["mtbf"], g["restart"], g["horizon"], g["seed"])
    return {"params": g, "sim": sim}


# chapter 16 ----------------------------------------------------------------
SFT_PROMPT = "who ran? "
SFT_RESPONSE = "the rabbit."
PREF_PROMPT = "said the "
PREF_CHOSEN = "queen."
PREF_REJECTED = "qxzjv."


def token_logprobs(params: dict, context: str, continuation: str) -> list[dict]:
    """The model's log probability of each character of `continuation`
    after `context`, from its last 8 characters (Part 1's context)."""
    S = PART1["seq_len"]
    ids = M.encode_chars(context)
    out = []
    for ch in continuation:
        window = ids[-S:]
        logits, _ = M.forward(params, PART1, np.array([window], dtype=np.int64))
        last = logits[0, len(window) - 1]
        m = float(last.max())
        lse = m + math.log(float(np.exp(last - m).sum()))
        t = M.CHAR_TO_ID[ch]
        out.append({"char": ch, "logp": float(last[t]) - lse})
        ids.append(t)
    return out


def sft_fixture(trained: dict) -> dict:
    prompt = token_logprobs(trained, SFT_PROMPT[0], SFT_PROMPT[1:])
    resp = token_logprobs(trained, SFT_PROMPT, SFT_RESPONSE)
    chosen = token_logprobs(trained, PREF_PROMPT, PREF_CHOSEN)
    rejected = token_logprobs(trained, PREF_PROMPT, PREF_REJECTED)
    resp_loss = -sum(r["logp"] for r in resp) / len(resp)
    all_loss = -(sum(r["logp"] for r in prompt) + sum(r["logp"] for r in resp)) / (len(prompt) + len(resp))
    return {"prompt": SFT_PROMPT, "response": SFT_RESPONSE, "promptTokens": prompt, "responseTokens": resp,
            "sftLoss": resp_loss, "unmaskedLoss": all_loss,
            "pref": {"prompt": PREF_PROMPT, "chosen": PREF_CHOSEN, "rejected": PREF_REJECTED,
                     "chosenTokens": chosen, "rejectedTokens": rejected,
                     "chosenLogp": sum(c["logp"] for c in chosen),
                     "rejectedLogp": sum(c["logp"] for c in rejected)}}
