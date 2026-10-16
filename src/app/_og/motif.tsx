/**
 * This site's favicon glyph (src/app/icon.svg) and its card's graphic.
 * Glyph: the circle of the Foundations track, the letter D, #0891b2.
 */
import results from "@/data/training/results.json";
import { attentionPreview } from "@/lib/attention-preview";

export function Glyph({ size }: { size: number }): JSX.Element {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32">
      <circle cx="16" cy="16" r="15.5" fill="#0891b2" />
      <path
        d="M22.22 15.85Q22.22 19.66 20.07 21.58Q17.91 23.5 14.03 23.5H9.78V8.5H14.49Q16.84 8.5 18.57 9.34Q20.31 10.18 21.26 11.81Q22.22 13.44 22.22 15.85ZM18.92 15.94Q18.92 13.44 17.83 12.27Q16.74 11.11 14.66 11.11H12.95V20.87H14.32Q18.92 20.87 18.92 15.94Z"
        fill="#fff"
      />
    </svg>
  );
}

/**
 * Decoder's card graphic: the landing hero's trained attention weights for "the cat " (block 1,
 * head 1), one cell per (letter, earlier letter), brightness = weight, masked cells outlined.
 * Computed at build time by the same maths library the chapters use.
 */
export function Motif(): JSX.Element {
  const { weights } = attentionPreview({ text: "the cat ", head: 1 });
  const n = weights.length;
  const s = 400 / n;
  return (
    <svg width="440" height="420" viewBox="0 0 440 420">
      {weights.map((row, i) =>
        row.map((w, j) => (
          <rect
            key={`${i}-${j}`}
            x={20 + j * s}
            y={10 + i * s}
            width={s - 4}
            height={s - 4}
            rx={4}
            fill={j <= i ? "#22d3ee" : "#0b1020"}
            fillOpacity={j <= i ? 0.15 + 0.85 * w : 1}
            stroke={j <= i ? "none" : "#334155"}
            strokeWidth={2}
          />
        )),
      )}
    </svg>
  );
}

/** Training's glyph: the Foundations circle, the letter T, #c2410c (src/app/training/icon.svg). */
export function TrainingGlyph({ size }: { size: number }): JSX.Element {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32">
      <circle cx="16" cy="16" r="15.5" fill="#c2410c" />
      <path
        d="M17.59 23.5H14.41V11.15H10.34V8.5H21.66V11.15H17.59Z"
        fill="#fff"
      />
    </svg>
  );
}

/**
 * Training's card graphic: the validation loss of the default run (chapter 1), from the recorded,
 * seeded run in results.json, falling from random guessing.
 */
export function TrainingMotif(): JSX.Element {
  const ev = results.defaultRun.evals;
  const steps = ev[ev.length - 1]!.step;
  const hi = ev[0]!.val + 0.1;
  const lo = Math.min(...ev.map((e) => e.val)) - 0.3;
  const x = (st: number) => 20 + (400 * st) / steps;
  const y = (v: number) => 20 + (380 * (hi - v)) / (hi - lo);
  const pts = ev.map((e) => `${x(e.step)},${y(e.val)}`).join(" ");
  return (
    <svg width="440" height="420" viewBox="0 0 440 420">
      <line
        x1={20}
        y1={404}
        x2={420}
        y2={404}
        stroke="#475569"
        strokeWidth={3}
      />
      <line x1={16} y1={20} x2={16} y2={404} stroke="#475569" strokeWidth={3} />
      <polyline
        points={`20,404 ${pts} 420,404`}
        fill="#c2410c"
        fillOpacity={0.22}
        stroke="none"
      />
      <polyline
        points={pts}
        fill="none"
        stroke="#fb923c"
        strokeWidth={8}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      {ev.map((e) => (
        <circle
          key={e.step}
          cx={x(e.step)}
          cy={y(e.val)}
          r={6}
          fill="#fdba74"
        />
      ))}
    </svg>
  );
}
