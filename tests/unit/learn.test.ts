/**
 * Brief 44, the learning layer: every chapter ends with three self-checks
 * and a "What to remember" box, and every numeric answer is the model's or
 * the recorded run's value, printed as the chapter prints its numbers.
 *
 * 1. Structure: each of the 23 chapters has one <CheckYourself> with
 *    exactly three <Check>s (ids unique, numbered 1-3), and one <Remember>
 *    with three to five takeaways and a <Keep>.
 * 2. Numeric answers: each `k` resolves in results.json and each `v` in the
 *    learn values; the values are checked against the fixture or recorded
 *    number they come from (results.json is written by the Python
 *    reference; the fixtures by scripts/reference.py and
 *    scripts/training/make_fixtures.py).
 * 3. Choice answers that rest on a measured number: the right choice is
 *    what the model or the recorded run says.
 * 4. The shared component renders the question and the answer with no
 *    JavaScript (a native <details>), and judges a guess by its tolerance.
 */
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";

import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import optim2dFixture from "./fixtures/training/anim_optim2d.json";
import { SelfCheck, isClose } from "@/components/learn/SelfCheck";
import { resolveAnswer } from "@/components/mdx/Check";
import { formatValue, lookup } from "@/components/training/R";
import results from "@/data/training/results.json";
import trainedHeads from "@/data/decoder/trained-heads.json";
import { LEARN_VALUE_NAMES, learnValue } from "@/lib/learn/values";
import {
  optim2dStates,
  OPTIM2D,
  VALLEY,
} from "@/lib/anim/training/optim2d-steps";

const CONTENT = path.resolve(__dirname, "../../content");
const chapters = ["decoder", "training"].flatMap((part) =>
  readdirSync(path.join(CONTENT, part))
    .filter((f) => f.endsWith(".mdx"))
    .map((f) => ({
      name: `${part}/${f}`,
      source: readFileSync(path.join(CONTENT, part, f), "utf-8"),
    })),
);

type CheckTag = { props: Record<string, string>; body: string };

/** Each <Check …>…</Check> in a chapter: its props (raw) and its body. */
function checks(source: string): CheckTag[] {
  return [
    ...source.matchAll(/<Check\s([\s\S]*?)\n?>\n([\s\S]*?)<\/Check>/g),
  ].map((m) => {
    const props: Record<string, string> = {};
    const raw = m[1] ?? "";
    for (const p of raw.matchAll(
      /(\w+)(?:=("[^"]*"|\{(?:\[[\s\S]*?\]|[^}]*)\}))?/g,
    )) {
      const key = p[1]!;
      const val = p[2];
      props[key] =
        val === undefined
          ? "true"
          : val.startsWith('"')
            ? val.slice(1, -1)
            : val.slice(1, -1);
    }
    return { props, body: m[2] ?? "" };
  });
}

function format(props: Record<string, string>) {
  return {
    d: props.d !== undefined ? Number(props.d) : undefined,
    pct: props.pct === "true",
    sep: props.sep === "true",
    div: props.div !== undefined ? Number(props.div) : undefined,
    sig: props.sig !== undefined ? Number(props.sig) : undefined,
  };
}

const all = chapters.flatMap((c) =>
  checks(c.source).map((t) => ({ chapter: c.name, ...t })),
);

describe("every chapter: three self-checks and a Remember box", () => {
  it("finds all 23 chapters", () => {
    expect(chapters).toHaveLength(23);
  });

  it.each(chapters.map((c) => [c.name, c.source]))("%s", (_name, source) => {
    expect(source.match(/<CheckYourself>/g)).toHaveLength(1);
    const cs = checks(source);
    expect(cs.map((c) => c.props.n)).toEqual(["1", "2", "3"]);
    // the Check blocks sit inside CheckYourself
    const start = source.indexOf("<CheckYourself>");
    const end = source.indexOf("</CheckYourself>");
    expect(source.slice(start, end).match(/<Check\s/g)).toHaveLength(3);
    // Remember: 3-5 takeaways and one Keep, after the self-checks
    const r = source.indexOf("<Remember>");
    expect(r).toBeGreaterThan(end);
    const remember = source.slice(r, source.indexOf("</Remember>"));
    const items = remember.match(/^- /gm) ?? [];
    expect(items.length).toBeGreaterThanOrEqual(3);
    expect(items.length).toBeLessThanOrEqual(5);
    expect(remember.match(/<Keep>/g)).toHaveLength(1);
  });

  it("ids are unique across the site", () => {
    const ids = all.map((c) => c.props.id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(ids).toHaveLength(69);
  });

  it("every self-check has a question, a reason and a link to its section", () => {
    for (const c of all) {
      expect(c.props.q, c.props.id).toBeTruthy();
      expect(c.body.trim().length, c.props.id).toBeGreaterThan(40);
      expect(c.props.href, c.props.id).toMatch(/^(#|\/)/);
      expect(c.props.more, c.props.id).toBeTruthy();
    }
  });

  it("every in-page link points at an anchor that exists in that chapter", () => {
    for (const c of all) {
      const href = c.props.href!;
      if (!href.startsWith("#")) continue;
      const source = chapters.find((ch) => ch.name === c.chapter)!.source;
      expect(source, `${c.props.id} → ${href}`).toContain(
        `id="${href.slice(1)}"`,
      );
    }
  });

  it("every cross-chapter link names a chapter that exists, and its anchor", () => {
    for (const c of all) {
      const href = c.props.href!;
      const m = /^\/(learn|training)\/([\w-]+)(?:#([\w-]+))?$/.exec(href);
      if (!m) continue;
      const dir = m[1] === "learn" ? "decoder" : "training";
      const target = chapters.find((ch) => ch.name === `${dir}/${m[2]}.mdx`);
      expect(target, `${c.props.id} → ${href}`).toBeDefined();
      if (m[3]) expect(target!.source).toContain(`id="${m[3]}"`);
    }
  });
});

describe("numeric answers come from the model or the recorded runs", () => {
  const numeric = all.filter((c) => (c.props.input ?? "choice") !== "choice");

  it("there are numeric self-checks on both sites", () => {
    expect(
      numeric.filter((c) => c.chapter.startsWith("decoder")).length,
    ).toBeGreaterThanOrEqual(7);
    expect(
      numeric.filter((c) => c.chapter.startsWith("training")).length,
    ).toBeGreaterThanOrEqual(16);
  });

  it.each(numeric.map((c) => [c.props.id, c]))(
    "%s resolves to a number and prints as the chapter prints it",
    (_id, c) => {
      const { k, v } = c.props;
      expect(k === undefined ? v : k, "needs k or v").toBeDefined();
      const raw = resolveAnswer(k, v);
      expect(typeof raw).toBe("number");
      if (k !== undefined) expect(raw).toBe(lookup(k));
      const shown = formatValue(raw, format(c.props));
      expect(shown).not.toMatch(/NaN|undefined/);
      // a typed-in answer equal to what is shown counts as right
      const f = format(c.props);
      const typed = Number(shown.replace(/[%,]/g, ""));
      const value = ((f.pct ? (raw as number) * 100 : (raw as number)) /
        (f.div ?? 1)) as number;
      expect(
        isClose(
          typed,
          value,
          c.props.tol !== undefined ? Number(c.props.tol) : 0.1,
          c.props.abs !== undefined ? Number(c.props.abs) : 0,
        ),
      ).toBe(true);
    },
  );

  it("every learn value computes", () => {
    for (const name of LEARN_VALUE_NAMES) {
      expect(learnValue(name), name).toBeDefined();
    }
  });

  it("the Decoder's values equal the recorded model and the head statistics", () => {
    expect(learnValue("decoder.params")).toBe(results.model.params);
    expect(learnValue("decoder.vocab")).toBe(results.model.config.vocab_size);
    expect(learnValue("decoder.seqLen")).toBe(results.model.config.seq_len);
    expect(learnValue("decoder.nBlocks")).toBe(results.model.config.n_blocks);
    const S = results.model.config.seq_len;
    const maskFixture = readFileSync(
      path.join(__dirname, "fixtures/causal_mask.json"),
      "utf-8",
    );
    expect(maskFixture.match(/-Infinity/g)).toHaveLength(
      learnValue("decoder.maskedPerHead") as number,
    );
    expect(learnValue("decoder.maskedPerHead")).toBe((S * (S - 1)) / 2);
    const h = trainedHeads.trained.find((r) => r.block === 0 && r.head === 1)!;
    expect(learnValue("decoder.b1h1CurrentPrev")).toBe(h.self + h.prev);
    expect(learnValue("decoder.headWindows")).toBe(h.windows);
    const dk = results.model.config.d_model / results.model.config.n_heads;
    expect(learnValue("decoder.scoreScale")).toBeCloseTo(1 / Math.sqrt(dk), 15);
    expect(learnValue("decoder.pe0Sum")).toBe(results.model.config.d_model / 2);
    expect(learnValue("decoder.geluMinus1")).toBeCloseTo(-0.158808, 6);
    expect(learnValue("decoder.ln1234Max")).toBeCloseTo(
      1.5 / Math.sqrt(1.25 + 1e-5),
      12,
    );
    expect(learnValue("decoder.lnConstantMaxAbs")).toBe(0);
    expect(learnValue("decoder.top1AtT1")).toBeCloseTo(
      Math.exp(2) / (Math.exp(2) + Math.exp(1) + 1),
      14,
    );
    expect(learnValue("decoder.top1AtT05")).toBeCloseTo(
      Math.exp(4) / (Math.exp(4) + Math.exp(2) + 1),
      14,
    );
  });

  it("the Training values equal the recorded numbers they are built from", () => {
    expect(learnValue("training.valleySgdLimit")).toBe(
      2 / optim2dFixture.valley.b,
    );
    expect(learnValue("training.raceSgdLr")).toBe(
      optim2dFixture.optimisers.find((o) => o.kind === "sgd")!.lr,
    );
    expect(learnValue("training.bytesPerParamMixedAdam")).toBe(16);
    expect(learnValue("training.topkPayloadRatio")).toBe(
      results.slowLinks.rows[1]!.bytes.payload /
        results.slowLinks.rows[0]!.bytes.payload,
    );
    const p = results.systems.collectives.rows[4]!.p;
    expect(learnValue("training.ringWireOverPayload64")).toBeCloseTo(
      (2 * (p - 1)) / p,
      12,
    );
    const { p: pp, m } = results.pipeline.rows[0]!;
    expect(learnValue("training.gpipeIdleOfStep")).toBeCloseTo(
      (pp - 1) / (m + pp - 1),
      12,
    );
  });
});

describe("choice answers that rest on a measured number", () => {
  const answerOf = (id: string) => {
    const c = all.find((x) => x.props.id === id)!;
    const choices = JSON.parse(
      c.props.choices!.replace(/,\s*\]$/, "]"),
    ) as string[];
    return choices[Number(c.props.answer)]!;
  };
  const R = (k: string) => lookup(k) as number;

  it("d06-q1: the trained model's next character after “hello!” is the chosen one", () => {
    expect(learnValue("decoder.helloNext")).toBe("a space");
    expect(answerOf("d06-q1")).toBe("A space");
  });
  it("d04-q1: the FFN and attention matrices are the same size in this toy", () => {
    expect(learnValue("decoder.ffnWeights")).toBe(
      learnValue("decoder.attnWeights"),
    );
    expect(answerOf("d04-q1")).toBe("They are the same size");
  });
  it("t05-q1: Nesterov overshoots less along the floor than heavy-ball, at the same settings", () => {
    const nag = results.optim2d.byKind.nesterov;
    const hb = results.optim2d.byKind.momentum;
    expect(nag.overshootX).toBeLessThan(hb.overshootX);
    expect(answerOf("t05-q1")).toBe("Nesterov momentum");
    // and the browser's own race agrees with the recorded summary
    const states = optim2dStates();
    const i = OPTIM2D.findIndex((o) => o.kind === "nesterov");
    const over = Math.max(...states.map((s) => s.points[i]![0] - VALLEY.xStar));
    expect(over).toBeCloseTo(nag.overshootX, 12);
  });
  it("t05-q3: the four optimisers end within about one standard deviation", () => {
    const best = [2, 4, 7, 10].map((i) => R(`optimizerCompare.${i}.final`));
    const sds = [2, 4, 7, 10].map((i) => R(`optimizerCompare.${i}.sd`));
    expect(Math.max(...best) - Math.min(...best)).toBeLessThan(
      2 * Math.max(...sds),
    );
  });
  it("t06-q1: without warmup the run ends lower", () => {
    expect(R("schedules.rows.0.mean")).toBeLessThan(R("schedules.rows.1.mean"));
  });
  it("t06-q2: 16 times the batch, far fewer than 16 times fewer steps", () => {
    expect(R("batchSize.rows.4.batch") / R("batchSize.rows.0.batch")).toBe(16);
    expect(
      R("batchSize.rows.0.steps") / R("batchSize.rows.4.steps"),
    ).toBeLessThan(8);
  });
  it("t07-q2: post-LN without warmup stays on the plateau", () => {
    expect(R("prePost.rows.2.mean")).toBeGreaterThan(
      R("corpus.unigramEntropy") - 0.1,
    );
    expect(R("prePost.rows.0.mean")).toBeLessThan(
      R("prePost.rows.2.mean") - 0.3,
    );
  });
  it("t07-q3: under SP the logits move further at width 256; under µP they don't", () => {
    expect(R("coordCheck.sp.4.logitsDelta")).toBeGreaterThan(
      2 * R("coordCheck.sp.0.logitsDelta"),
    );
    expect(
      Math.abs(
        R("coordCheck.mup.4.logitsDelta") / R("coordCheck.mup.0.logitsDelta") -
          1,
      ),
    ).toBeLessThan(0.05);
  });
  it("t08-q2: QK-norm keeps the scores small; z-loss does not", () => {
    expect(R("attnGrowth.rows.4.meanStats.-1.maxScore")).toBeLessThan(
      R("attnGrowth.rows.0.meanStats.-1.maxScore"),
    );
    expect(R("attnGrowth.rows.5.meanStats.-1.maxScore")).toBeGreaterThan(
      100 * R("attnGrowth.rows.4.meanStats.-1.maxScore"),
    );
  });
  it("t08-q3: AdamW barely notices the bad batch; momentum blows up", () => {
    expect(
      Math.abs(R("badBatch.rows.3.after.0") - R("badBatch.rows.3.before.0")),
    ).toBeLessThan(0.05);
    expect(R("badBatch.rows.0.after.0")).toBeGreaterThan(
      R("badBatch.rows.0.before.0") + 5,
    );
  });
  it("t09-q1: FP16 flushes more gradients to zero than BF16", () => {
    expect(R("gradRounding.2.zero")).toBeGreaterThan(R("gradRounding.1.zero"));
  });
  it("t11-q1: four workers' averaged gradient trains the same run as one worker", () => {
    expect(
      Math.abs(
        R("slowLinks.rows.0.perLr.1.finals.0") - R("defaultRun.evals.-1.val"),
      ),
    ).toBeLessThan(1e-6);
  });
  it("t13-q2: DeMo's wire bytes exceed its payload", () => {
    expect(R("slowLinks.rows.2.bytes.wire")).toBeGreaterThan(
      2 * R("slowLinks.rows.2.bytes.payload"),
    );
  });
  it("t15-q3: weights-only resume ends within the seed spread", () => {
    expect(
      Math.abs(R("resume.weights.mean") - R("resume.uninterrupted.mean")),
    ).toBeLessThan(R("resume.weights.sd"));
    expect(lookup("resume.bitIdentical")).toBe(true);
  });
});

describe("SelfCheck renders without JavaScript", () => {
  it("puts the question, the choices and the answer in the page, inside <details>", () => {
    const html = renderToStaticMarkup(
      createElement(SelfCheck, {
        id: "x",
        n: 1,
        question: "Which?",
        input: "choice",
        choices: ["A", "B"],
        answer: 1,
        why: "Because.",
        href: "#here",
        hrefLabel: "Here",
      }),
    );
    expect(html).toContain("Which?");
    expect(html).toMatch(
      /<details[^>]*>[\s\S]*<summary[\s\S]*Reveal the answer/,
    );
    expect(html).toMatch(/data-answer="?[^>]*>B</);
    expect(html).toContain("Because.");
    expect(html).toContain('href="#here"');
    expect(html).not.toMatch(/<details[^>]*open/);
  });
  it("prints a numeric answer with its unit and tolerance", () => {
    const html = renderToStaticMarkup(
      createElement(SelfCheck, {
        id: "y",
        n: 2,
        question: "How many?",
        input: "number",
        value: 37.5,
        display: "37.5",
        unit: "%",
        tol: 0.1,
        why: "So.",
      }),
    );
    expect(html).toContain("37.5%");
    expect(html).toContain("within 10%");
  });
  it("judges a guess by its tolerance", () => {
    expect(isClose(9.5, 10)).toBe(true);
    expect(isClose(8.5, 10)).toBe(false);
    expect(isClose(0.004, 0, 0.1, 0.01)).toBe(true);
    expect(isClose(8, 8, 0)).toBe(true);
    expect(isClose(7, 8, 0)).toBe(false);
  });
});
