"use client";

/**
 * Tiny hook that listens for the `te:preset` CustomEvent broadcast by
 * `<PlaygroundPresets>` and forwards the text/seed onto the widget's
 * setters. Each widget calls `usePreset(setText, setSeed)` once.
 */
import { useEffect } from "react";

type PresetDetail = { text: string; seed: number };

export function usePreset(
  setText: (v: string) => void,
  setSeed: (v: number) => void,
): void {
  useEffect(() => {
    function handler(e: Event) {
      const detail = (e as CustomEvent<PresetDetail>).detail;
      if (!detail) return;
      setText(detail.text);
      setSeed(detail.seed);
    }
    window.addEventListener("te:preset", handler);
    return () => window.removeEventListener("te:preset", handler);
  }, [setText, setSeed]);
}
