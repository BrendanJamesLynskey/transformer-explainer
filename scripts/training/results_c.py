"""
results_c.py — brief 29C's results (chapters 11-16) and the fixtures their
TypeScript ports and animations are checked against. Called by
make_fixtures.py for the groups:

  sys    the training-systems model (systems.py): validation against
         published runs, collectives, ZeRO, pipeline bubbles, reliability,
         stragglers, queueing; the ring, pipeline, goodput and SFT
         animations (deterministic, seconds)
  ch13   training over slow links (slowlinks.py): the four methods, their
         learning rates, three seeds; the DeMo animation (about 20 minutes)
  ch14   the IsoFLOP sweep (isoflop.py; about 30 minutes)
  ch15   the default run as a dashboard sees it and the resume experiment
         (jobs.py; a few minutes)
"""

from __future__ import annotations

import math

import anim_c as AC
import isoflop as ISO
import jobs as J
import model as M
import slowlinks as SL
import systems as SY

GROUPS_C = {
    "sys": ["systems", "pipeline", "sft"],
    "ch13": ["slowLinks"],
    "ch14": ["isoflop"],
    "ch15": ["monitor", "resume"],
}

# chapter 11's worked example: LLaMA 7B's shape (as chapter 10), data
# parallel over 8 GPUs in a server (NVLink) or 64 across servers
LLAMA7B = SY.gpt_arch("LLaMA 7B", L=32, h=4096, a=32, s=4096, V=32000, f=11008, swiglu=True, tied=False)


def systems_results() -> dict:
    val = SY.validation()
    eta = val["eta"]
    h = SY.HARDWARE["h100"]
    N = SY.n_params(LLAMA7B)
    grad_bytes = 2 * N
    coll = []
    for where, p, bw, alpha in (("NVLink, 8 GPUs", 8, h["nvlink"], h["nvlink_alpha"]),
                                ("network, 64 GPUs", 64, h["net"], h["net_alpha"])):
        for kind in ("allreduce", "tree", "reducescatter", "allgather"):
            c = SY.collective(kind, grad_bytes, p, bw, alpha)
            coll.append({"where": where, "p": p, "kind": kind, "time": c["time"], "wire": c["wire"]})
    zero = []
    for d in (8, 64):
        for stage in range(4):
            z = SY.zero_memory(N, d, stage)
            zero.append({"d": d, "stage": stage, **z})
    # one data-parallel step of LLaMA 7B on 64 H100s, 1M tokens per step
    lay = dict(t=1, p=1, d=64, B=256, b=4, recompute="none")
    st = SY.step_time(LLAMA7B, lay, h, eta)
    dp = {"layout": lay, "params": N, "time": st["time"], "compute": st["compute"], "data": st["data"],
          "overlapped": SY.overlapped(st["compute"], st["data"]),
          "mfu": st["mfu"], "mfuOverlapped": st["mfu"] * st["time"] / SY.overlapped(st["compute"], st["data"]),
          "tokens": lay["B"] * LLAMA7B["s"]}
    # reliability, Llama 3's interruption rate
    mtbf = 54 * 24 * 3600 / 419
    delta = 16384 * 4e9 / 2e12
    restart = 600.0
    taus = [round(60 * 2 ** (k / 2)) for k in range(14)]
    rel = SY.goodput_sweep(delta, mtbf, restart, 54 * 24 * 3600, taus, list(range(1, 21)))
    rel["llama"] = {"days": 54, "interruptions": 466, "unexpected": 419, "gpus": 16384,
                    "mtbfHours": mtbf / 3600, "gpuMtbfYears": mtbf * 16384 / (3600 * 24 * 365)}
    # Llama 3 405B's compute budget (3.8e25 FLOPs, §1) at ~400 TFLOP/s on each
    # of 16,384 GPUs (Table 4), in days
    rel["llama"]["computeDays"] = 3.8e25 / (400e12 * 16384) / 86400
    rel["bestGoodput"] = max(r["simulated"] for r in rel["rows"])
    rel["deltaNote"] = "16,384 GPUs x 4 GB at the storage fabric's 2 TB/s sustained (Llama 3 §3.3.1)"
    rel["restartNote"] = "illustrative: 10 minutes to restart and reload"
    strag = [SY.stragglers(8, 2000, H, 1.0, 0.1, 1) for H in (1, 10, 50)]
    queue = [{"load": L, "formula": SY.md1_wait(L, 1.0), "simulated": SY.simulate_queue(L, 1.0, 200000, 20000, 1)}
             for L in (0.3, 0.6, 0.9)]
    return {"hardware": SY.HARDWARE, "validation": val, "collectives": {"bytes": grad_bytes, "params": N,
            "rows": coll}, "zero": {"params": N, "rows": zero}, "dpExample": dp, "reliability": rel,
            "stragglers": strag, "queue": queue, "goodputAnim": AC.GOODPUT}


PIPE_CASES = [("gpipe", 4, 8, 1), ("1f1b", 4, 8, 1), ("interleaved", 4, 8, 2), ("interleaved", 4, 8, 4),
              ("1f1b", 8, 32, 1), ("interleaved", 8, 32, 4), ("gpipe", 16, 32, 1), ("1f1b", 16, 64, 1)]


def pipeline_results() -> dict:
    rows = []
    for kind, p, m, v in PIPE_CASES:
        r = SY.pipeline(kind, p, m, v)
        rows.append({"kind": kind, "p": p, "m": m, "v": r["v"], "makespan": r["makespan"],
                     "bubble": r["bubble"], "closedForm": r["closedForm"]})
    return {"rows": rows, "maxDiff": max(abs(r["bubble"] - r["closedForm"]) for r in rows)}


def results_c(ids, groups, trained) -> dict:
    out: dict = {}
    if "sys" in groups:
        out["systems"] = systems_results()
        out["pipeline"] = pipeline_results()
        out["sft"] = AC.sft_fixture(trained)
    if "ch13" in groups:
        out["slowLinks"] = SL.experiment(ids)
    if "ch14" in groups:
        out["isoflop"] = ISO.sweep(ids)
    if "ch15" in groups:
        out["monitor"] = J.monitor(ids)
        out["resume"] = J.resume(ids)
    return out


# ---------------------------------------------------------------------------
# Fixtures for the TypeScript ports
# ---------------------------------------------------------------------------
def systems_fixture(results_sys: dict) -> dict:
    """Cases for tests/unit/training/systems.test.ts."""
    a100, h100 = SY.HARDWARE["a100"], SY.HARDWARE["h100"]
    coll = []
    for kind in ("allreduce", "tree", "reducescatter", "allgather", "alltoall"):
        for p in (1, 2, 8, 64):
            coll.append({"kind": kind, "size": 1.3e10, "p": p, "bw": 50e9, "alpha": 1e-5,
                         "out": SY.collective(kind, 1.3e10, p, 50e9, 1e-5)})
    zero = [{"params": 6.7e9, "d": d, "stage": s, "out": SY.zero_memory(6.7e9, d, s)}
            for d in (1, 8, 64) for s in range(4)]
    gpt3 = SY.gpt_arch("GPT-3 175B", L=96, h=12288, a=96, s=2048)
    mem = []
    for lay in (dict(t=8, p=8, d=16, sp=True, recompute="selective", v=1),
                dict(t=8, p=8, d=16, sp=False, recompute="none", v=2, zero=1),
                dict(t=4, p=1, d=64, recompute="full", zero=3, b=2),
                dict(t=8, p=16, d=4, cp=2, sp=True, recompute="none", v=4, m=32)):
        mem.append({"arch": gpt3, "lay": lay, "out": SY.device_memory(gpt3, lay)})
    moe = dict(gpt3, name="MoE")
    mem.append({"arch": moe, "lay": dict(t=2, p=4, d=8, ep=8, experts=8, sp=True), "out":
                SY.device_memory(moe, dict(t=2, p=4, d=8, ep=8, experts=8, sp=True))})
    steps = []
    eta = results_sys["validation"]["eta"]
    for row in results_sys["validation"]["rows"]:
        hw = SY.HARDWARE[row["hw"]]
        steps.append({"arch": row["arch"], "lay": row["layout"], "hw": row["hw"], "eta": eta,
                      "flops": row["flops"],
                      "out": SY.step_time(row["arch"], row["layout"], hw, eta, row["flops"])})
    pipes = [SY.pipeline(k, p, m, v) for k, p, m, v in PIPE_CASES[:6]]
    g = AC.GOODPUT
    good = [{"tau": tau, "seed": seed, "out": SY.simulate_goodput(tau, g["delta"], g["mtbf"], g["restart"],
                                                                  g["horizon"], seed)}
            for tau, seed in ((3600.0, 3), (900.0, 1), (7200.0, 2))]
    for x in good:
        x["out"] = {"goodput": x["out"]["goodput"], "failures": x["out"]["failures"],
                    "segments": x["out"]["segments"]}
    return {"hardware": {"a100": a100, "h100": h100}, "collectives": coll, "zero": zero, "memory": mem,
            "stepTime": steps, "pipelines": pipes, "goodputParams": g, "goodput": good,
            "young": [{"delta": d, "mtbf": m, "out": SY.young_interval(d, m)} for d, m in ((30, 1e4), (300, 3e5))],
            "expected": [{"args": [t, 30.0, 1e4, 600.0], "out": SY.expected_goodput(t, 30.0, 1e4, 600.0)}
                         for t in (100.0, 800.0, 5000.0)],
            "md1": [{"load": L, "out": SY.md1_wait(L, 2.0)} for L in (0.1, 0.5, 0.95)]}


SL_FIX_STEPS = 20


def slowlinks_fixture(ids) -> dict:
    """20 steps of each method (DiLoCo with H = 5), validation every 10,
    for tests/unit/training/slowlinks.test.ts."""
    runs = []
    for method, lr, H in (("dense", 0.01, 1), ("topk", 1.0, 1), ("demo", 0.01, 1), ("diloco", 0.03, 5)):
        cfg = SL.base(method=method, lr=lr, H=H, steps=SL_FIX_STEPS)
        r = SL.train_workers(cfg, ids, keep_params=True, eval_every=10)
        runs.append({"cfg": cfg, "evals": r["evals"],
                     "params": {k: v.tolist() for k, v in r["params"].items()}})
    shapes = {k: v.shape for k, v in M.init_params(SL.PART1).items()}
    sizes = [int(math.prod(s)) for s in shapes.values()]
    return {"runs": runs, "sizes": sizes,
            "bytes": {m: SL.bytes_for(m, 10) for m in ("dense", "topk", "demo", "diloco")},
            "dct8": SL.dct_matrix(8).tolist(),
            "chunks": {str(n): {"m": SL.chunk_len(n), "k": SL.keep_count(SL.chunk_len(n))} for n in sizes}}
