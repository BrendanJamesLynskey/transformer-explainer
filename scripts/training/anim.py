"""
anim.py — the state lists behind Part 2's animations (chapters 1-5), built
here from the reference model so the TypeScript builders in
src/lib/anim/training/*.ts can be checked against them (frame tests).

Each builder mirrors its TypeScript twin field for field:
  hero_states       ch 1  three training steps on a batch of 4 windows
  bpe_states        ch 2  the first merges learnt on the corpus
  objective_states  ch 3  cross-entropy position by position, and p - y
  backprop_states   ch 4  the gradient at each stop on the way back
  optim2d_states    ch 5  five optimisers on a 2-D valley (Nesterov added by brief 44)
"""

from __future__ import annotations

import math

import numpy as np

import data as D
import model as M
import optim as O

PART1 = dict(M.PART1_CONFIG)


def tolist(a):
    return np.asarray(a, dtype=np.float64).tolist()


# ---------------------------------------------------------------------------
# Chapter 1: the training loop
# ---------------------------------------------------------------------------
HERO = dict(batch=4, steps=3, lr=0.01, beta2=0.99, weight_decay=0.01, clip=1.0, data_seed=1)

# Parameter groups in the order the gradient reaches them on the way back.
GROUPS = [
    ("final LayerNorm", ["ln_final.gamma", "ln_final.beta"]),
    ("block 1 · FFN", ["block_1.ln2.gamma", "block_1.ln2.beta", "block_1.ffn.W1", "block_1.ffn.b1",
                       "block_1.ffn.W2", "block_1.ffn.b2"]),
    ("block 1 · attention", ["block_1.ln1.gamma", "block_1.ln1.beta", "block_1.attn.W_q",
                             "block_1.attn.W_k", "block_1.attn.W_v", "block_1.attn.W_o"]),
    ("block 0 · FFN", ["block_0.ln2.gamma", "block_0.ln2.beta", "block_0.ffn.W1", "block_0.ffn.b1",
                       "block_0.ffn.W2", "block_0.ffn.b2"]),
    ("block 0 · attention", ["block_0.ln1.gamma", "block_0.ln1.beta", "block_0.attn.W_q",
                             "block_0.attn.W_k", "block_0.attn.W_v", "block_0.attn.W_o"]),
    ("embedding E", ["tok_emb"]),
]


def group_norm(t: dict, names) -> float:
    s = 0.0
    for n in names:
        s += float((t[n] * t[n]).sum())
    return math.sqrt(s)


def hero_states(ids) -> list[dict]:
    train, _ = D.split(ids)
    S = PART1["seq_len"]
    params = M.init_params(PART1)
    opt = O.Optimizer("adamw", params, beta2=HERO["beta2"], weight_decay=HERO["weight_decay"])
    rng = M.mulberry32(HERO["data_seed"])
    states = []
    G = len(GROUPS)
    for t in range(HERO["steps"]):
        x, y = D.sample_batch(train, HERO["batch"], S, rng)
        logits, _ = M.forward(params, PART1, x)
        loss, probs, per = M.cross_entropy_fwd(logits, y)
        p_t = np.take_along_axis(probs, y[..., None], -1)[..., 0]
        _, grads, _ = M.loss_and_grads(params, PART1, x, y)
        norms = [group_norm(grads, names) for _, names in GROUPS]
        gnorm, scale = O.clip_by_global_norm(grads, HERO["clip"])
        before = {k: v.copy() for k, v in params.items()}
        opt.step(params, grads, HERO["lr"])
        delta = {k: params[k] - before[k] for k in params}
        ratios = [group_norm(delta, names) / group_norm(before, names) for _, names in GROUPS]
        after = M.loss_only(params, PART1, x, y)
        base = dict(step=t, group=-1, inputs=x.tolist(), targets=y.tolist(), lr=HERO["lr"],
                    pTarget=None, perPosition=None, loss=None, gradNorms=[None] * G,
                    gradNorm=None, clipScale=None, updateRatios=None, lossAfter=None)
        states.append(dict(base, phase="batch"))
        fwd = dict(base, pTarget=tolist(p_t))
        states.append(dict(fwd, phase="forward"))
        lossd = dict(fwd, perPosition=tolist(per), loss=float(loss))
        states.append(dict(lossd, phase="loss"))
        for g in range(G):
            states.append(dict(lossd, phase="backward", group=g,
                               gradNorms=[norms[k] if k <= g else None for k in range(G)]))
        states.append(dict(lossd, phase="update", gradNorms=norms, gradNorm=gnorm,
                           clipScale=scale, updateRatios=ratios, lossAfter=after))
    return states


# ---------------------------------------------------------------------------
# Chapter 2: BPE merges
# ---------------------------------------------------------------------------
BPE_ANIM_MERGES = 24
BPE_EXCERPT = 96


def bpe_states(text: str) -> list[dict]:
    model = D.bpe_train(text, BPE_ANIM_MERGES)
    excerpt = text[:BPE_EXCERPT]
    vocab = model["vocab"]

    def tokens(k):  # the excerpt after the first k merges
        return [vocab[i] for i in D.bpe_encode(excerpt, model["merges"][:k])]

    first = D.word_counts(text)
    total = sum(len(w) * n for w, n in first.items())
    states = [dict(step=-1, pair=None, count=None, token=None, top=None,
                   tokensBefore=total, tokensAfter=total, excerpt=tokens(0))]
    for h in model["history"]:
        states.append(dict(step=h["step"], pair=h["pair"], count=h["count"], token=h["token"],
                           top=h["top"], tokensBefore=h["tokensBefore"], tokensAfter=h["tokensAfter"],
                           excerpt=tokens(h["step"] + 1)))
    return states


# ---------------------------------------------------------------------------
# Chapters 3 and 4: one window, at initialisation and trained
# ---------------------------------------------------------------------------
WINDOW_TEXT = "alice was"  # S + 1 = 9 characters: 8 inputs, 8 targets


def window():
    ids = M.encode_chars(WINDOW_TEXT)
    return np.array([ids[:-1]]), np.array([ids[1:]])


def objective_states(params) -> list[dict]:
    x, y = window()
    logits, _ = M.forward(params, PART1, x)
    _, probs, per = M.cross_entropy_fwd(logits, y)
    S = x.shape[1]
    n = S
    states = []
    losses = [None] * S
    for i in range(S):
        z, p, tgt = logits[0, i], probs[0, i], int(y[0, i])
        grad = (p - np.eye(len(p))[tgt]) / n
        base = dict(pos=i, inputs=x[0].tolist(), targets=y[0].tolist(), target=tgt,
                    logits=tolist(z), probs=None, pTarget=None, loss=None, grad=None, mean=None)
        states.append(dict(base, phase="logits", losses=list(losses)))
        states.append(dict(base, phase="softmax", probs=tolist(p), losses=list(losses)))
        states.append(dict(base, phase="target", probs=tolist(p), pTarget=float(p[tgt]),
                           losses=list(losses)))
        losses[i] = float(per[0, i])
        states.append(dict(base, phase="loss", probs=tolist(p), pTarget=float(p[tgt]),
                           loss=float(per[0, i]), losses=list(losses)))
        states.append(dict(base, phase="grad", probs=tolist(p), pTarget=float(p[tgt]),
                           loss=float(per[0, i]), grad=tolist(grad), losses=list(losses)))
    last = states[-1]
    states.append(dict(last, phase="mean", mean=float(np.mean(per))))
    return states


def backprop_record(params) -> dict:
    x, y = window()
    rec = []
    loss, grads, _ = M.loss_and_grads(params, PART1, x, y, rec)
    return {"loss": float(loss),
            "record": [{"name": n, "value": tolist(v)} for n, v in rec],
            "gradNorms": {k: math.sqrt(float((g * g).sum())) for k, g in grads.items()}}


# ---------------------------------------------------------------------------
# Chapter 5: five optimisers on a 2-D valley. Nesterov (brief 44, appended so
# the first four keep their places) runs at momentum's learning rate and
# momentum, so the two differ only in where the gradient is taken.
# ---------------------------------------------------------------------------
VALLEY = dict(a=1.0, b=50.0, x_star=2.0, y_star=1.0, start=[-2.5, -1.5], steps=60)
OPTIM2D = [
    dict(kind="sgd", lr=0.036),
    dict(kind="momentum", lr=0.02, momentum=0.8),
    dict(kind="adam", lr=0.1),
    dict(kind="adamw", lr=0.1, weight_decay=0.3),
    dict(kind="nesterov", lr=0.02, momentum=0.8),
]


def valley_loss(th) -> float:
    v = VALLEY
    return 0.5 * (v["a"] * (th[0] - v["x_star"]) ** 2 + v["b"] * (th[1] - v["y_star"]) ** 2)


def valley_grad(th):
    v = VALLEY
    return np.array([[v["a"] * (th[0] - v["x_star"]), v["b"] * (th[1] - v["y_star"])]])


def optim2d_states() -> list[dict]:
    params = [{"theta": np.array([VALLEY["start"]], dtype=np.float64)} for _ in OPTIM2D]
    opts = [O.Optimizer(s["kind"], params[i], momentum=s.get("momentum", 0.9),
                        weight_decay=s.get("weight_decay", 0.0)) for i, s in enumerate(OPTIM2D)]
    states = []
    for t in range(VALLEY["steps"] + 1):
        pts = [p["theta"][0].tolist() for p in params]
        adam = opts[[s["kind"] for s in OPTIM2D].index("adam")]
        states.append(dict(t=t, points=pts, losses=[valley_loss(p) for p in pts],
                           adamM=adam.m["theta"][0].tolist(), adamV=adam.v["theta"][0].tolist()))
        if t == VALLEY["steps"]:
            break
        for i, s in enumerate(OPTIM2D):
            g = {"theta": valley_grad(params[i]["theta"][0])}
            opts[i].step(params[i], g, s["lr"])
    return states


def optim2d_summary(states: list[dict]) -> dict:
    """What chapter 5's text and self-checks quote about the race, per
    optimiser (in OPTIM2D's order): the final loss; the furthest it goes past
    the minimum along the valley floor (x - x*, 0 if never) and across it
    (|y - y*| after the start, as it swings); the first step whose loss is
    below 1e-3 (None if never)."""
    out = []
    for i, spec in enumerate(OPTIM2D):
        xs = [st["points"][i][0] for st in states]
        ys = [st["points"][i][1] for st in states]
        losses = [st["losses"][i] for st in states]
        below = next((st["t"] for st in states if st["losses"][i] < 1e-3), None)
        out.append({
            "kind": spec["kind"],
            "finalLoss": losses[-1],
            "overshootX": max(0.0, max(x - VALLEY["x_star"] for x in xs)),
            "overshootY": max(0.0, max(y - VALLEY["y_star"] for y in ys[1:])),
            "stepBelow1e3": below,
            "yErrorStep2": ys[2] - VALLEY["y_star"],
        })
    return {"steps": VALLEY["steps"], "optimisers": out,
            "byKind": {o["kind"]: o for o in out}}


def fixtures(ids, text) -> dict:
    """File name -> payload, for make_fixtures.py."""
    return {
        "anim_hero.json": {"config": HERO, "states": hero_states(ids)},
        "anim_bpe.json": {"merges": BPE_ANIM_MERGES, "excerptChars": BPE_EXCERPT,
                          "states": bpe_states(text)},
        "anim_optim2d.json": {"valley": VALLEY, "optimisers": OPTIM2D, "states": optim2d_states()},
    }


def window_fixtures(trained: dict) -> dict:
    """Chapters 3 and 4 need the trained weights too (make_fixtures passes them)."""
    init = M.init_params(PART1)
    return {
        "anim_objective.json": {"text": WINDOW_TEXT,
                                "init": objective_states(init), "trained": objective_states(trained)},
        "anim_backprop.json": {"text": WINDOW_TEXT,
                               "init": backprop_record(init), "trained": backprop_record(trained)},
    }
