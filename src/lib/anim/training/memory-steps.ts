/**
 * Chapter 10's hero: the memory in use through one training step, event
 * by event, from the calculator (memory.ts): weights, optimiser states and
 * the FP32 master copy all the time; activations piling up through the
 * forward pass; the logits at the loss; gradients appearing and
 * activations freed through the backward pass (with recomputation, one
 * layer's activations re-created first). scripts/training/memory.py:
 * timeline builds the same list.
 */
import {
  fmtBytes,
  timeline,
  type Arch,
  type MemEvent,
  type Recipe,
  type Recompute,
} from "@/lib/training/memory";

export function memoryStates(
  arch: Arch,
  recipe: Recipe,
  recompute: Recompute,
): MemEvent[] {
  return timeline(arch, recipe, recompute);
}

export function memoryCaption(e: MemEvent, arch: Arch): string {
  const total = fmtBytes(e.total);
  switch (e.event) {
    case "start":
      return `Before the step: weights, optimiser states and master copy, ${total}.`;
    case "forward":
      return `Forward, layer ${e.layer + 1} of ${arch.L}: its activations are saved for the backward pass; ${total} in use.`;
    case "loss":
      return `The loss: the logits (${arch.s} × ${arch.b} positions × ${arch.V.toLocaleString("en-GB")} scores) add ${fmtBytes(e.logits)}; ${total} in use.`;
    case "head":
      return `Backward through the output layer: its gradients appear; ${total} in use.`;
    case "recompute":
      return `Layer ${e.layer + 1}: recompute its activations from the saved input (${fmtBytes(e.transient)}); ${total} in use.`;
    case "backward":
      return `Backward, layer ${e.layer + 1}: its gradients are written and its activations freed; ${total} in use.`;
    case "step":
      return `The optimiser step updates the weights in place: ${total}.`;
  }
}
