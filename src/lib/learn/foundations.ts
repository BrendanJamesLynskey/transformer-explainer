/**
 * The Foundations start page's glossary (/foundations, brief 44): each term,
 * a one-line gloss, and the chapter section that defines it. Paths are this
 * deployment's (`siteHref` sends /training links to the training host);
 * tests/unit/foundations.test.ts checks that every chapter and anchor exists.
 */
export type GlossaryTerm = { term: string; href: string; what: string };

/** Each term, the page that defines it, and the section's anchor. */
export const FOUNDATIONS_GLOSSARY: readonly GlossaryTerm[] = [
  {
    term: "Decoder, logits",
    href: "/learn/01-overview#d-pipeline",
    what: "a stack of blocks that turns tokens into a score (logit) for every possible next token",
  },
  {
    term: "Token, embedding",
    href: "/learn/02-embeddings#d-positions",
    what: "a token's id picks one learned row of the embedding table",
  },
  {
    term: "Positional encoding",
    href: "/learn/02-embeddings#d-pe",
    what: "a vector added to each row that says where the token sits",
  },
  {
    term: "Attention",
    href: "/learn/03-attention",
    what: "each position takes a weighted average of earlier positions' values",
  },
  {
    term: "Causal mask",
    href: "/learn/03-attention#d-mask",
    what: "sets future scores to −∞, so a position never sees what comes next",
  },
  {
    term: "Multi-head attention",
    href: "/learn/03-attention#d-heads",
    what: "several attention heads on slices of the model width, joined",
  },
  {
    term: "Feed-forward network (FFN)",
    href: "/learn/04-ffn#d-ffn",
    what: "a two-layer MLP applied to each position on its own",
  },
  {
    term: "GELU",
    href: "/learn/04-ffn#d-gelu",
    what: "the smooth non-linearity inside the FFN",
  },
  {
    term: "Residual stream",
    href: "/learn/05-layernorm-residuals#d-residual",
    what: "the running vector every sub-layer reads and adds to",
  },
  {
    term: "LayerNorm, pre-norm",
    href: "/learn/05-layernorm-residuals#d-layernorm",
    what: "normalise each position's vector before each sub-layer",
  },
  {
    term: "Logit lens",
    href: "/learn/06-stacking#d-lens",
    what: "reading the stream early with the final LayerNorm and head",
  },
  {
    term: "Temperature, top-k, top-p",
    href: "/learn/07-sampling#d-rules",
    what: "the rules that turn logits into a sampled next token",
  },
  {
    term: "Training step",
    href: "/training/01-training-loop#t-training-step",
    what: "one batch, one forward and backward pass, one update",
  },
  {
    term: "Validation loss",
    href: "/training/01-training-loop#t-validation-loss",
    what: "the loss on held-out text the model never trains on",
  },
  {
    term: "Cross-entropy, perplexity",
    href: "/training/03-objective#t-cross-entropy",
    what: "minus the log of the right token's probability; e to the loss",
  },
  {
    term: "Gradient, backpropagation",
    href: "/training/04-backprop#t-backprop",
    what: "the chain rule, one operation at a time, from the loss back",
  },
  {
    term: "SGD, momentum",
    href: "/training/05-optimisers#t-momentum",
    what: "step downhill; keep a running sum of past gradients",
  },
  {
    term: "Nesterov momentum",
    href: "/training/05-optimisers#t-nesterov",
    what: "momentum with the gradient taken at the look-ahead point",
  },
  {
    term: "Adam, AdamW",
    href: "/training/05-optimisers#t-adam",
    what: "per-weight step sizes; decoupled weight decay",
  },
  {
    term: "Learning-rate sweep, warmup",
    href: "/training/06-learning-rate-and-schedules#t-lr-sweep",
    what: "how the most important number is found and started",
  },
  {
    term: "Critical batch size",
    href: "/training/06-learning-rate-and-schedules#t-critical-batch",
    what: "beyond it, a bigger batch barely saves steps",
  },
  {
    term: "Pre-LN, post-LN, µP",
    href: "/training/07-init-norm-mup#t-pre-post",
    what: "where the norm sits; one learning rate for every width",
  },
  {
    term: "QK-norm, z-loss",
    href: "/training/08-stability#t-logit-growth",
    what: "cures for growing attention scores and drifting logits",
  },
  {
    term: "BF16, mixed precision",
    href: "/training/09-mixed-precision#t-bf16",
    what: "16-bit compute with an FP32 master copy of the weights",
  },
  {
    term: "16 bytes per parameter",
    href: "/training/10-memory#t-sixteen-bytes",
    what: "weights, gradients, Adam's two moments and the master copy",
  },
  {
    term: "Activation checkpointing",
    href: "/training/10-memory#t-checkpointing",
    what: "recompute activations in the backward pass to save memory",
  },
  {
    term: "Data parallelism, all-reduce",
    href: "/training/11-data-parallelism#t-all-reduce",
    what: "every worker a full copy; the gradients summed by a ring",
  },
  {
    term: "ZeRO, FSDP",
    href: "/training/11-data-parallelism#t-zero",
    what: "shard the optimiser states, gradients and weights",
  },
  {
    term: "MFU",
    href: "/training/11-data-parallelism#t-mfu",
    what: "the model's useful FLOPs as a share of the hardware's peak",
  },
  {
    term: "Pipeline bubble",
    href: "/training/12-model-parallelism#t-pipeline",
    what: "time a pipeline stage waits, (p − 1)/m of the work",
  },
  {
    term: "Tensor parallelism",
    href: "/training/12-model-parallelism#t-tensor-parallel",
    what: "each layer's matrices split across the GPUs of a server",
  },
  {
    term: "Top-k, DeMo, DiLoCo",
    href: "/training/13-slow-links#t-top-k",
    what: "sending less, or less often, over slow links",
  },
  {
    term: "Scaling law, compute 6ND",
    href: "/training/14-scaling-laws#t-compute",
    what: "the best model size for a compute budget",
  },
  {
    term: "Checkpoint, goodput",
    href: "/training/15-running-a-real-job#t-checkpoint",
    what: "saving a run to resume it; the share of time spent training",
  },
  {
    term: "SFT, DPO",
    href: "/training/16-after-pretraining#t-sft",
    what: "fine-tuning on responses; optimising preferences directly",
  },
];
