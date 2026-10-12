/**
 * Chapter 16's hero: what fine-tuning changes about the objective, shown on
 * the trained model. Supervised fine-tuning (SFT) scores only the
 * response: the model reads the prompt, but the prompt's characters are
 * masked out of the loss. Preference optimisation compares two
 * continuations of one prompt by their log probabilities. Every number is
 * the trained model's own (Part 1's forward pass, a window of its last 8
 * characters); scripts/training/anim_c.py: sft_fixture computes the same
 * (tests: anim-frames-c.test.ts).
 */
import { sig, showChar } from "@/lib/anim/format";
import { encodeChars } from "@/lib/training/data";
import { forwardCached } from "@/lib/training/model-grad";
import type { ParamMap } from "@/lib/training/params";
import { PART1_MODEL } from "@/lib/training/trainer";

export type TokenLogp = { char: string; logp: number };

/** log p of each character of `continuation` after `context`. */
export function tokenLogprobs(
  params: ParamMap,
  context: string,
  continuation: string,
): TokenLogp[] {
  const S = PART1_MODEL.seq_len;
  const ids = encodeChars(context);
  const out: TokenLogp[] = [];
  for (const ch of continuation) {
    const win = ids.slice(-S);
    const { logits } = forwardCached(params, PART1_MODEL, win);
    const last = logits[win.length - 1]!;
    let m = -Infinity;
    for (const v of last) m = Math.max(m, v);
    let s = 0;
    for (const v of last) s += Math.exp(v - m);
    const t = encodeChars(ch)[0]!;
    out.push({ char: ch, logp: last[t]! - (m + Math.log(s)) });
    ids.push(t);
  }
  return out;
}

export const SFT = {
  prompt: "who ran? ",
  response: "the rabbit.",
  prefPrompt: "said the ",
  chosen: "queen.",
  rejected: "qxzjv.",
} as const;

export type SftData = {
  promptTokens: TokenLogp[];
  responseTokens: TokenLogp[];
  chosenTokens: TokenLogp[];
  rejectedTokens: TokenLogp[];
};

export function sftData(params: ParamMap): SftData {
  return {
    promptTokens: tokenLogprobs(params, SFT.prompt[0]!, SFT.prompt.slice(1)),
    responseTokens: tokenLogprobs(params, SFT.prompt, SFT.response),
    chosenTokens: tokenLogprobs(params, SFT.prefPrompt, SFT.chosen),
    rejectedTokens: tokenLogprobs(params, SFT.prefPrompt, SFT.rejected),
  };
}

export type SftState =
  | { kind: "prompt"; i: number }
  | { kind: "response"; i: number; sftLoss: number }
  | { kind: "compare"; chosen: number; rejected: number };

export function sftStates(d: SftData): SftState[] {
  const out: SftState[] = [];
  d.promptTokens.forEach((_, i) => out.push({ kind: "prompt", i }));
  let total = 0;
  d.responseTokens.forEach((t, i) => {
    total += -t.logp;
    out.push({ kind: "response", i, sftLoss: total / (i + 1) });
  });
  const sum = (ts: TokenLogp[]) => ts.reduce((a, t) => a + t.logp, 0);
  out.push({
    kind: "compare",
    chosen: sum(d.chosenTokens),
    rejected: sum(d.rejectedTokens),
  });
  return out;
}

export const SFT_HL: Record<SftState["kind"], string> = {
  prompt: "mask2",
  response: "resp",
  compare: "pref",
};

export function sftCaption(s: SftState, d: SftData): string {
  if (s.kind === "prompt") {
    const t = d.promptTokens[s.i]!;
    return `Prompt character ${s.i + 2} "${showChar(t.char)}": the model reads it (−log p would be ${sig(-t.logp)}), but SFT masks it out of the loss. The prompt is given, not learnt.`;
  }
  if (s.kind === "response") {
    const t = d.responseTokens[s.i]!;
    return `Response character ${s.i + 1} of ${d.responseTokens.length} "${showChar(t.char)}": −log p = ${sig(-t.logp)} counts. SFT loss so far ${sig(s.sftLoss)} nats (mean over the response only).`;
  }
  return `Preference: two continuations of "${SFT.prefPrompt.trim()}". log p("${SFT.chosen}") = ${sig(s.chosen)}, log p("${SFT.rejected}") = ${sig(s.rejected)}. DPO raises the first relative to the second (and relative to a frozen copy of the model), with no separate reward model.`;
}
