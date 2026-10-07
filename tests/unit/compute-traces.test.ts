/**
 * The client-side trace path (brief 27): the animations compute their
 * traces in the browser with `src/lib/compute/traces.ts`. These tests
 * call each `/api/compute/*` route handler and require its JSON `data` to
 * equal the client's trace for the same input, number for number, so
 * stepping an animation shows exactly what the API returns.
 */
import { describe, expect, it } from "vitest";

import { POST as attentionRoute } from "@/app/api/compute/attention/route";
import { POST as blockRoute } from "@/app/api/compute/block/route";
import { POST as embedRoute } from "@/app/api/compute/embed/route";
import { POST as ffnRoute } from "@/app/api/compute/ffn/route";
import { POST as forwardRoute } from "@/app/api/compute/forward/route";
import {
  DEFAULT_PARAMS,
  computeAttention,
  computeBlock,
  computeEmbed,
  computeFfn,
  computeForward,
  configFor,
  tracedForward,
  type ComputeParams,
} from "@/lib/compute/traces";
import { initModelWeights } from "@/lib/transformer/init";
import { forwardTyped } from "@/lib/transformer/model";
import { encode } from "@/lib/transformer/tokenizer";

type Route = (req: Request) => Promise<Response>;

async function call(route: Route, body: unknown): Promise<unknown> {
  const res = await route(
    new Request("http://localhost/api/compute", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
  expect(res.status).toBe(200);
  const json = (await res.json()) as { ok: boolean; data: unknown };
  expect(json.ok).toBe(true);
  return json.data;
}

/** What the client's trace looks like after the same JSON round trip. */
const asJson = (v: unknown): unknown => JSON.parse(JSON.stringify(v));

const INPUTS: Partial<ComputeParams>[] = [
  {},
  { text: "the cat sat", seed: 7 },
  { text: "(a+b)*c", seed: 3, seqLen: 12, dModel: 24, nHeads: 3 },
  { text: "", seed: 0, seqLen: 4, dModel: 8, nHeads: 4, dFf: 16 },
];

const CASES: [string, Route, (p: ComputeParams) => unknown, string[]][] = [
  ["embed", embedRoute, computeEmbed, ["text", "seed", "seqLen", "dModel"]],
  [
    "attention",
    attentionRoute,
    computeAttention,
    ["text", "seed", "seqLen", "dModel", "nHeads"],
  ],
  [
    "ffn",
    ffnRoute,
    computeFfn,
    ["text", "seed", "seqLen", "dModel", "dFf", "nHeads"],
  ],
  [
    "block",
    blockRoute,
    computeBlock,
    ["text", "seed", "seqLen", "dModel", "dFf", "nHeads"],
  ],
  [
    "forward",
    forwardRoute,
    computeForward,
    ["text", "seed", "seqLen", "dModel", "dFf", "nHeads", "nBlocks"],
  ],
];

describe("client trace = API trace", () => {
  for (const [name, route, compute, keys] of CASES) {
    for (const input of INPUTS) {
      it(`${name} ${JSON.stringify(input)}`, async () => {
        const p: ComputeParams = { ...DEFAULT_PARAMS, ...input };
        // send only the fields this route accepts; the rest take defaults
        const body = Object.fromEntries(
          keys.map((k) => [k, p[k as keyof ComputeParams]]),
        );
        const api = await call(route, body);
        expect(api).toEqual(asJson(compute(p)));
      });
    }
  }
});

describe("tracedForward", () => {
  it("records every block and returns the same logits as forwardTyped", () => {
    const config = configFor(DEFAULT_PARAMS);
    const w = initModelWeights(config);
    const ids = encode("hello!", config.seq_len);
    const tr = tracedForward(ids, config, w);
    expect(tr.blocks).toHaveLength(2);
    expect(tr.blocks[0]!.attn.weights).toHaveLength(2);
    expect(tr.logits).toEqual(forwardTyped(ids, config, w).logits);
  });
});
