/**
 * The training-systems model (systems.ts) against scripts/training/systems.py
 * (tests/unit/fixtures/training/systems.json, written by make_fixtures.py
 * --groups sys): collectives, ZeRO, memory per device, step time on every
 * published configuration of the validation, the pipeline simulation,
 * failures and checkpoints, queueing. Arithmetic in the same order, so
 * 1e-12 (relative above 1); the goodput simulation draws logarithms, 1e-9.
 */
import { describe, expect, it } from "vitest";

import results from "@/data/training/results.json";
import {
  HARDWARE,
  activationsPerLayer,
  collective,
  deviceMemory,
  expectedGoodput,
  gptArch,
  md1Wait,
  modelFlopsPerToken,
  nParams,
  pipeline,
  ringStates,
  simulateGoodput,
  stepTime,
  youngInterval,
  zeroMemory,
  type CollectiveKind,
  type Layout,
  type PipeKind,
  type SysArch,
} from "@/lib/training/systems";

import fx from "../fixtures/training/systems.json";
import rx from "../fixtures/training/anim_ring.json";
import { mismatch } from "../helpers/close";

type PyArch = {
  name: string;
  L: number;
  h: number;
  a: number;
  s: number;
  V: number;
  f: number;
  kv: number;
  swiglu: boolean;
  tied: boolean;
};
const arch = (a: PyArch): SysArch => ({ ...a });
const lay = (l: Record<string, unknown>): Layout & { B: number } =>
  l as unknown as Layout & { B: number };

describe("hardware", () => {
  it("the same numbers as systems.py", () => {
    for (const k of ["a100", "h100"] as const) {
      const py = fx.hardware[k];
      expect(HARDWARE[k].peak).toBe(py.peak);
      expect(HARDWARE[k].nvlink).toBe(py.nvlink);
      expect(HARDWARE[k].net).toBe(py.net);
      expect(HARDWARE[k].nvlinkAlpha).toBe(py.nvlink_alpha);
      expect(HARDWARE[k].netAlpha).toBe(py.net_alpha);
    }
  });
});

describe("collectives", () => {
  it("time and bytes on the wire = Python's", () => {
    for (const c of fx.collectives) {
      const got = collective(
        c.kind as CollectiveKind,
        c.size,
        c.p,
        c.bw,
        c.alpha,
      );
      expect(mismatch(got, c.out, 1e-12)).toBeNull();
    }
  });
  it("a ring all-reduce sends 2(p-1)/p of the vector, whatever p", () => {
    expect(collective("allreduce", 100, 4, 1, 0).wire).toBe(150);
    expect(collective("allreduce", 100, 64, 1, 0).wire).toBeCloseTo(196.875);
  });
});

describe("ring all-reduce, run chunk by chunk (chapter 11's animation)", () => {
  const states = ringStates(rx.grads);
  it("every state = Python's", () => {
    expect(mismatch(states, rx.states, 1e-12)).toBeNull();
  });
  it("ends with every rank holding the sum", () => {
    const last = states.at(-1)!;
    for (const b of last.bufs)
      b.forEach((v, i) => expect(v).toBeCloseTo(rx.sum[i]!, 12));
    expect(states).toHaveLength(1 + 2 * (rx.grads.length - 1));
  });
});

describe("memory", () => {
  it("ZeRO stages = Python's", () => {
    for (const z of fx.zero)
      expect(mismatch(zeroMemory(z.params, z.d, z.stage), z.out)).toBeNull();
  });
  it("memory per device = Python's", () => {
    for (const m of fx.memory)
      expect(
        mismatch(deviceMemory(arch(m.arch), lay(m.lay)), m.out),
      ).toBeNull();
  });
  it("Korthikanti Table 2: selective with SP is 34/t, full is 2", () => {
    const a = gptArch("x", 1, 1024, 16, 2048);
    const sbh = 2048 * 1024;
    expect(activationsPerLayer(a, 1, 8, true, "selective")).toBe(
      (sbh * 34) / 8,
    );
    expect(activationsPerLayer(a, 1, 8, true, "full")).toBe(2 * sbh);
  });
});

describe("step time on the published configurations", () => {
  it("every row = Python's", () => {
    for (const s of fx.stepTime) {
      const got = stepTime(
        arch(s.arch),
        lay(s.lay),
        HARDWARE[s.hw as "a100" | "h100"],
        s.eta,
        s.flops as "palm" | "megatron",
      );
      expect(mismatch(got, s.out, 1e-12)).toBeNull();
    }
  });
  it("the validation in results.json is what the port computes", () => {
    const v = results.systems.validation;
    for (const row of v.rows) {
      const st = stepTime(
        arch(row.arch as PyArch),
        lay(row.layout),
        HARDWARE[row.hw as "a100" | "h100"],
        v.eta,
        row.flops as "palm" | "megatron",
      );
      const pred =
        row.flops === "megatron" ? st.hwFlopsPerGpu : st.modelFlopsPerGpu;
      expect(pred / row.predicted - 1).toBeLessThan(1e-12);
    }
  });
  it("Llama 3 405B has 405.8B parameters by this count", () => {
    const l = results.systems.validation.rows.find(
      (r) => r.source === "llama3",
    )!;
    expect(nParams(arch(l.arch as PyArch)) / 1e9).toBeCloseTo(405.84, 2);
    expect(modelFlopsPerToken(arch(l.arch as PyArch))).toBeGreaterThan(
      6 * 405e9,
    );
  });
});

describe("pipeline schedules (a discrete-event simulation)", () => {
  it("every work item and the bubble = Python's", () => {
    for (const p of fx.pipelines) {
      const got = pipeline(p.kind as PipeKind, p.p, p.m, p.v);
      expect(mismatch(got, p, 1e-12)).toBeNull();
    }
  });
  it("the measured bubble is the closed form: (p-1)/m, (p-1)/(vm)", () => {
    for (const [k, p, m, v] of [
      ["gpipe", 4, 8, 1],
      ["1f1b", 8, 16, 1],
      ["interleaved", 4, 16, 2],
      ["interleaved", 8, 32, 4],
    ] as const) {
      const r = pipeline(k, p, m, v);
      expect(r.bubble).toBeCloseTo(r.closedForm, 12);
    }
  });
  it("interleaving needs m divisible by p", () => {
    expect(() => pipeline("interleaved", 4, 6, 2)).toThrow();
  });
});

describe("reliability and queueing", () => {
  const g = fx.goodputParams;
  it("the goodput simulation = Python's", () => {
    for (const c of fx.goodput) {
      const got = simulateGoodput(
        c.tau,
        g.delta,
        g.mtbf,
        g.restart,
        g.horizon,
        c.seed,
      );
      expect(mismatch(got, c.out, 1e-9)).toBeNull();
    }
  });
  it("Young's interval, the first-order goodput, M/D/1 = Python's", () => {
    for (const y of fx.young)
      expect(youngInterval(y.delta, y.mtbf)).toBeCloseTo(y.out, 9);
    for (const e of fx.expected) {
      const [t, d, m, r] = e.args as [number, number, number, number];
      expect(expectedGoodput(t, d, m, r)).toBeCloseTo(e.out, 12);
    }
    for (const q of fx.md1) expect(md1Wait(q.load, 2)).toBeCloseTo(q.out, 12);
    expect(md1Wait(1, 2)).toBe(Infinity);
  });
  it("the simulated optimum is next to Young's interval", () => {
    const r = results.systems.reliability;
    expect(r.bestTau / r.young).toBeGreaterThan(0.5);
    expect(r.bestTau / r.young).toBeLessThan(2);
  });
});
