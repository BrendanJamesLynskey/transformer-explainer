"use client";

/**
 * The width of an element in CSS pixels, followed with a ResizeObserver
 * (rotation, window resizes). The animations use it to pick a phone
 * layout (a narrower viewBox with fewer elements, visual standard §3)
 * before drawing, so the labels stay at least 11 px (`useSvgFont`).
 */
import { useCallback, useEffect, useRef, useState } from "react";

export function useContainerWidth(): {
  ref: (el: HTMLElement | null) => void;
  width: number;
} {
  const [width, setWidth] = useState(0);
  const observer = useRef<ResizeObserver | null>(null);
  const ref = useCallback((el: HTMLElement | null) => {
    observer.current?.disconnect();
    if (!el) return;
    const measure = () => setWidth(el.getBoundingClientRect().width);
    measure();
    if (typeof ResizeObserver === "undefined") return;
    observer.current = new ResizeObserver(measure);
    observer.current.observe(el);
  }, []);
  useEffect(() => () => observer.current?.disconnect(), []);
  return { ref, width };
}

/** Below this width (CSS px) an animation uses its phone layout. */
export const PHONE_BELOW = 560;
