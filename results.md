# Training results (the Training site)

Every number on the `/training` pages comes from this file's source,
`src/data/training/results.json`, written by
`scripts/training/make_fixtures.py` (NumPy float64 reference, seeded).
Regenerate with `.venv-fixtures/bin/python scripts/training/make_fixtures.py`;
CI runs it with `--check` and fails on any difference.

Recorded with Python 3.12.3, NumPy 2.4.4, PyTorch 2.11.0+cpu (CPU), x86_64.

## 1. Corpus

| | |
|---|---|
| Text | Alice's Adventures in Wonderland (Lewis Carroll, 1865) |
| Source | Project Gutenberg eBook #11 (public domain); see `public/training/CORPUS.md` |
| Characters | 142,041 (train 127,837, validation 14,204) |
| SHA-256 of `public/training/corpus.txt` | `8a2b46fd9524cc572aecb9b8293d8143a9ebc3c6e816601d1e11e3f8937d0178` |
| Unigram entropy of the training characters | 3.0000 nats |
| BPE tokens after 64 merges | 85,614 (1.659 characters per token) |

## 2. Model

The Decoder site's model: seq_len 8, d_model 16, 2 heads, d_ff 32, 2 blocks, vocabulary 64, seed 42; 5,376 parameters. A uniform guess scores ln 64 = 4.1589.

## 3. Correctness checks

| Check | Measured | Tolerance |
|---|---|---|
| Initialiser = `scripts/reference.py` port | max abs err 0.0 | exact |
| Backward = PyTorch autograd, at initialisation | max err / (1 + max abs grad) 4.4e-17; loss diff 0.0e+00 | 1e-12 |
| Backward = PyTorch autograd, perturbed weights | max err / (1 + max abs grad) 9.4e-17; loss diff 8.9e-16 | 1e-12 |
| Backward = central differences (h = 1e-6), at initialisation, 552 entries | max abs err 7.2e-10 | 1e-8 + 1e-6 abs grad |
| Backward = central differences (h = 1e-6), perturbed weights, 552 entries | max abs err 7.6e-10 | 1e-8 + 1e-6 abs grad |
| `sgd` = torch.optim over 25 steps | max abs err 0.0e+00 | 1e-12 |
| `momentum` = torch.optim over 25 steps | max abs err 0.0e+00 | 1e-12 |
| `nesterov` = torch.optim over 25 steps | max abs err 0.0e+00 | 1e-12 |
| `adam` = torch.optim over 25 steps | max abs err 2.2e-16 | 1e-12 |
| `adam_l2` = torch.optim over 25 steps | max abs err 2.2e-16 | 1e-12 |
| `adamw` = torch.optim over 25 steps | max abs err 1.1e-16 | 1e-12 |
| Nesterov on chapter 5's valley (60 steps, lr 0.02, momentum 0.8) = `torch.optim.SGD(nesterov=True)` on autograd's gradient; = the gradient-at-the-look-ahead form | max abs err 0.0e+00; 4.4e-16 | 1e-12 |
| Cosine schedule = CosineAnnealingLR | max abs err 3.5e-18 | 1e-15 |
| Vectorised rounding (`precision.round_array`) = the Numerics site's scalar code, bf16, e4m3, e5m2, fp16, fp32, 202,090 values | mismatches 0 | exact |

## 4. The default run (chapter 1)

2,000 steps of 16 windows of 8 characters; adamw (beta1 0.9, beta2 0.99, weight decay 0.01); learning rate 0.01, 50 warmup steps, cosine to 0.1 of it; clip at global norm 1.0; data seed 1. 8.3 s on the machine above (NumPy).

| Step | Validation loss | Training loss (this step's batch) |
|---:|---:|---:|
| 0 | 4.1774 | 4.1733 |
| 100 | 2.9702 | 2.8921 |
| 200 | 2.9446 | 2.9532 |
| 300 | 2.9567 | 2.8815 |
| 400 | 2.9591 | 2.9979 |
| 500 | 2.9604 | 3.1384 |
| 600 | 2.9514 | 3.0708 |
| 700 | 2.9493 | 3.1938 |
| 800 | 2.9548 | 2.9304 |
| 900 | 2.8593 | 2.8402 |
| 1,000 | 2.7777 | 2.8336 |
| 1,100 | 2.6193 | 2.5887 |
| 1,200 | 2.4653 | 2.5619 |
| 1,300 | 2.4394 | 2.6064 |
| 1,400 | 2.3981 | 2.5985 |
| 1,500 | 2.3685 | 2.4889 |
| 1,600 | 2.3502 | 2.5190 |
| 1,700 | 2.3241 | 2.5052 |
| 1,800 | 2.3204 | 2.3280 |
| 1,900 | 2.3140 | 2.4152 |
| 2,000 | 2.2993 | – |

Samples ('alice ' + 120 characters, temperature 0.8, seed 7):

- at initialisation: `ae$(814q -rk+8nyt9t$q[mjkt +f{ ?"$?}=clx2bg+y25@("wxjj{vm;wc"} =:}i8:]#<!$$ /a7v) 
$ =2;- uqab@=;b?!)%s%:*?=j%7.k0o/>-:%`
- after 2,000 steps: `ad, onle the pang hul cag soe she."
"ainnd the wos id welund wit war to srur ang poy the shecavt as twe, thok" pand the;`

## 5. Optimisers on the real model (chapter 5)

Each optimiser at three constant learning rates, 1,500 steps of 16 windows, clip 1.0,
beta2 0.99 for Adam and AdamW, AdamW weight decay 0.01; three data seeds (1, 2, 3) per
setting. Mean ± standard deviation of the final validation loss over the seeds.
Single runs at the higher learning rates are chaotic: recomputed on another CPU, a run's
final loss moved by about 0.1 (last-bit differences in the matrix products, amplified),
so only differences well beyond the spread mean anything.

| Optimiser | Learning rate | Final validation loss (mean ± sd) | Seeds 1, 2, 3 | Best for this optimiser |
|---|---:|---:|---|:---:|
| sgd | 0.1 | 2.897 ± 0.005 | 2.899, 2.891, 2.901 |  |
| sgd | 0.3 | 2.374 ± 0.014 | 2.370, 2.362, 2.390 |  |
| sgd | 1.0 | 2.241 ± 0.034 | 2.251, 2.202, 2.268 | yes |
| momentum | 0.03 | 2.342 ± 0.021 | 2.360, 2.319, 2.347 |  |
| momentum | 0.1 | 2.213 ± 0.008 | 2.205, 2.220, 2.215 | yes |
| momentum | 0.3 | 2.253 ± 0.055 | 2.316, 2.229, 2.215 |  |
| adam | 0.003 | 2.310 ± 0.031 | 2.327, 2.329, 2.274 |  |
| adam | 0.01 | 2.185 ± 0.041 | 2.168, 2.231, 2.155 | yes |
| adam | 0.03 | 2.301 ± 0.026 | 2.271, 2.317, 2.316 |  |
| adamw | 0.003 | 2.311 ± 0.038 | 2.334, 2.331, 2.267 |  |
| adamw | 0.01 | 2.230 ± 0.041 | 2.192, 2.225, 2.273 | yes |
| adamw | 0.03 | 2.357 ± 0.029 | 2.344, 2.337, 2.390 |  |

### Chapter 5's valley race (60 steps)

f(x, y) = ½((x − 2)² + 50 (y − 1)²) from (−2.5, −1.5); Nesterov at momentum's learning rate
and momentum. Overshoot = the furthest past the minimum, along the floor (x) and across (y).

| Optimiser | Final loss | Overshoot along x | Overshoot across y | First step with loss < 1e-3 |
|---|---:|---:|---:|---:|
| sgd | 1.24e-01 | 0.000 | 2.000 | never |
| momentum | 1.17e-04 | 0.133 | 2.000 | 49 |
| adam | 1.63e-01 | 0.000 | 0.222 | never |
| adamw | 1.40e+00 | 0.000 | 0.000 | never |
| nesterov | 5.24e-07 | 0.060 | 2.000 | 26 |

## Chapters 6-10: how to read these numbers

Every setting below ran with three data seeds; the pages quote the mean and the standard
deviation. Most of these runs are chaotic: whether and when a run leaves the plateau of
chapter 1 depends on the last bits of its arithmetic, so recomputing them on another CPU
moved single seeds' final losses by up to about 0.35, step counts by up to a factor of 2,
and made one run at learning rate 1.0 diverge. CI recomputes every group, sanity-checks
the means, and re-checks each claim the chapters make (`CLAIMS` in `make_fixtures.py`);
the deterministic parts (scale through depth, the coordinate check, the noise scale,
memory, rounding) are checked to 1e-7 or better.

## 6. Learning rate, schedules and batch size (chapter 6)

### 6a. Learning-rate sweep

AdamW (beta2 0.99, weight decay 0.01), warmup 50 steps then cosine to 0.1 of the peak,
2,000 steps of 16 windows, clip 1.0, three data seeds per peak learning rate.
A run counts as diverged if its loss stops being finite or exceeds 12 nats.

| Peak learning rate | Final validation loss (mean ± sd) | Seeds 1, 2, 3 |
|---:|---:|---|
| 0.001 | 2.948 ± 0.001 | 2.947, 2.948, 2.949 |
| 0.003 | 2.577 ± 0.009 | 2.579, 2.585, 2.567 |
| 0.01 | 2.277 ± 0.090 | 2.299, 2.354, 2.178 |
| 0.03 | 2.280 ± 0.107 | 2.218, 2.404, 2.218 |
| 0.1 | 2.571 ± 0.102 | 2.541, 2.684, 2.487 |
| 0.3 | 2.737 ± 0.084 | 2.643, 2.766, 2.802 |
| 1.0 | 2.906 ± 0.027 | 2.906, 2.879, 2.933 |

Best peak learning rate (lowest mean): 0.01.

### 6b. Schedules

AdamW at peak learning rate 0.01, 2,000 steps, three seeds; validation every 50 steps.

| Schedule | Final validation loss (mean ± sd) | Seeds 1, 2, 3 |
|---|---:|---|
| constant | 2.159 ± 0.045 | 2.111, 2.165, 2.201 |
| warmup + constant | 2.246 ± 0.025 | 2.275, 2.238, 2.227 |
| warmup + cosine | 2.277 ± 0.090 | 2.299, 2.354, 2.178 |
| WSD | 2.224 ± 0.038 | 2.264, 2.220, 2.189 |

### 6c. Batch size: steps to a target loss

Steps until the validation loss first reaches 2.6 (checked every 25 steps), at each
batch size (windows of 8 characters), AdamW at a constant learning rate after a 50-step warmup;
the better of two learning rates (lower mean steps), three seeds.

| Batch (windows) | Learning rate | Steps (mean ± sd) | Seeds | Windows processed |
|---:|---:|---:|---|---:|
| 4 | 0.01 | 1950 ± 459 | 2275, 1425, 2150 | 7,800 |
| 8 | 0.03 | 1483 ± 95 | 1550, 1375, 1525 | 11,867 |
| 16 | 0.03 | 1000 ± 90 | 1075, 1025, 900 | 16,000 |
| 32 | 0.03 | 842 ± 263 | 1100, 850, 575 | 26,933 |
| 64 | 0.01 | 625 ± 25 | 600, 650, 625 | 40,000 |

Fit of S = S_min (1 + B_crit / B) (McCandlish et al. 2018): B_crit = 10.4 windows, S_min = 592 steps, E_min = 6133 windows (RMS error 0.077 in log steps).

### 6d. Gradient noise scale along the default run

B_simple = tr(Sigma) / |G|^2 from 256 single-window gradients (McCandlish et al. 2018, eq. 2.9).

| Step | tr(Sigma) | abs(G)^2 | B_simple (windows) |
|---:|---:|---:|---:|
| 0 | 1.6e+00 | 5.8e-01 | 2.8 |
| 250 | 1.9e+00 | 3.1e-02 | 62.8 |
| 500 | 1.3e+00 | 1.0e-02 | 126.4 |
| 1,000 | 2.3e+00 | 2.8e-02 | 79.5 |
| 1,500 | 9.0e+00 | 5.6e-02 | 161.4 |
| 2,000 | 1.2e+01 | 2.6e-02 | 455.7 |

## 7. Initialisation, normalisation and muP (chapter 7)

### 7a. Scale through depth at initialisation

An 8-block version of the model on "alice was": RMS of the residual stream entering each
block (and leaving the last), and the norm of each block's weight-matrix gradients.

- std 0.02, pre-LN: residual RMS 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70; gradient norms 0.0269, 0.0372, 0.0305, 0.0316, 0.0309, 0.032, 0.03, 0.0352
- std 0.02, post-LN: residual RMS 0.70, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00; gradient norms 0.0192, 0.0192, 0.0155, 0.0162, 0.0157, 0.0163, 0.0152, 0.018
- std 0.25, pre-LN: residual RMS 0.72, 1.27, 1.86, 2.56, 2.91, 3.06, 3.36, 3.81, 4.30; gradient norms 1.14, 1.01, 0.947, 0.671, 0.778, 1.02, 1.04, 0.777
- std 0.25, post-LN: residual RMS 0.72, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00; gradient norms 1.29, 1.64, 1.63, 1.98, 1.99, 2.03, 2.73, 2.29

### 7b. Pre-LN vs post-LN

6 blocks, AdamW at peak learning rate 0.01, cosine, 1,500 steps, three seeds.

| Norm placement | Warmup steps | Final validation loss (mean ± sd) | Seeds 1, 2, 3 |
|---|---:|---:|---|
| pre-LN | 0 | 2.378 ± 0.128 | 2.302, 2.306, 2.526 |
| pre-LN | 200 | 2.310 ± 0.074 | 2.272, 2.395, 2.263 |
| post-LN | 0 | 2.949 ± 0.002 | 2.950, 2.950, 2.947 |
| post-LN | 200 | 2.877 ± 0.124 | 2.735, 2.950, 2.947 |

### 7c. LayerNorm vs RMSNorm

The default run (2 blocks) with each, three seeds.

| Norm | Parameters | Final validation loss (mean ± sd) | Seeds 1, 2, 3 |
|---|---:|---:|---|
| LayerNorm | 5,376 | 2.277 ± 0.090 | 2.299, 2.354, 2.178 |
| RMSNorm | 5,296 | 2.140 ± 0.008 | 2.148, 2.132, 2.141 |

### 7d. muP coordinate check

RMS change of the logits and of the last residual stream after 10 AdamW steps at
learning rate 0.01 (constant, no clipping), on 32 fixed windows.

| Width | SP: logits | SP: residual | muP: logits | muP: residual |
|---:|---:|---:|---:|---:|
| 16 | 1.4e+00 | 3.6e+00 | 1.4e+00 | 3.6e+00 |
| 32 | 2.2e+00 | 4.5e+00 | 1.4e+00 | 3.1e+00 |
| 64 | 2.5e+00 | 7.4e+00 | 1.4e+00 | 2.7e+00 |
| 128 | 3.2e+00 | 4.1e+01 | 1.4e+00 | 2.3e+00 |
| 256 | 4.5e+00 | 1.2e+02 | 1.4e+00 | 2.4e+00 |

### 7e. Learning-rate transfer across width

Final validation loss after 1,500 steps (warmup 50 + cosine), three seeds;
muP's base width is 16, so width 16 is the same model either way.

| Parameterisation | Width | Parameters | Peak learning rate | Final validation loss (mean ± sd) | Seeds 1, 2, 3 |
|---|---:|---:|---:|---:|---|
| standard | 16 | 5,376 | 0.01 | 2.420 ± 0.101 | 2.504, 2.448, 2.308 |
| standard | 16 | 5,376 | 0.03 | 2.457 ± 0.213 | 2.390, 2.696, 2.286 |
| standard | 16 | 5,376 | 0.1 | 2.656 ± 0.025 | 2.655, 2.632, 2.681 |
| standard | 64 | 70,656 | 0.01 | 2.440 ± 0.193 | 2.346, 2.662, 2.312 |
| standard | 64 | 70,656 | 0.03 | 2.531 ± 0.205 | 2.676, 2.622, 2.296 |
| standard | 64 | 70,656 | 0.1 | 2.811 ± 0.083 | 2.877, 2.718, 2.840 |
| muP | 16 | 5,376 | 0.01 | 2.420 ± 0.101 | 2.504, 2.448, 2.308 |
| muP | 16 | 5,376 | 0.03 | 2.457 ± 0.213 | 2.390, 2.696, 2.286 |
| muP | 16 | 5,376 | 0.1 | 2.656 ± 0.025 | 2.655, 2.632, 2.681 |
| muP | 64 | 70,656 | 0.01 | 2.314 ± 0.073 | 2.398, 2.275, 2.269 |
| muP | 64 | 70,656 | 0.03 | 2.310 ± 0.048 | 2.269, 2.297, 2.363 |
| muP | 64 | 70,656 | 0.1 | 2.682 ± 0.127 | 2.607, 2.829, 2.611 |

## 8. Stability (chapter 8)

### 8a. One bad batch

At step 1,000 every window of the batch is the character '@' repeated. 1,400 steps, constant learning rate, three seeds.

| Run | Learning rate | Validation loss before | Just after | At the end |
|---|---:|---|---|---|
| momentum, no clipping | 0.5 | 2.334, 2.424, 2.296 | 17.138, 8.175, 14.494 | diverged, diverged, diverged |
| momentum, clip at 1.0 | 0.5 | 2.424, 2.454, 2.359 | 2.477, 2.598, 2.448 | 2.325, 2.351, 2.318 |
| momentum, skip the batch | 0.5 | 2.334, 2.424, 2.296 | 2.334, 2.424, 2.296 | 2.322, 2.276, 2.306 |
| AdamW (lr 0.01), no clipping | 0.01 | 2.367, 2.329, 2.435 | 2.377, 2.342, 2.441 | 2.229, 2.250, 2.282 |

### 8b. Attention-logit growth at a high learning rate

AdamW, warmup 50 + cosine, 2,000 steps, clip 1.0, three seeds. Largest attention score and
mean attention entropy on the validation windows at the end (mean over seeds).

| Peak learning rate | Run | Final validation loss (mean ± sd) | Largest score | Entropy (nats) |
|---:|---|---:|---:|---:|
| 0.01 | baseline | 2.277 ± 0.090 | 42.2 | 0.62 |
| 0.01 | QK-norm | 2.272 ± 0.033 | 11.0 | 0.60 |
| 0.01 | z-loss 1e-3 | 2.310 ± 0.113 | 67.5 | 0.38 |
| 0.1 | baseline | 2.571 ± 0.102 | 16,104.7 | 0.05 |
| 0.1 | QK-norm | 2.102 ± 0.039 | 24.4 | 0.34 |
| 0.1 | z-loss 1e-3 | 2.589 ± 0.102 | 19,648.6 | 0.05 |

## 9. Mixed precision and quantised gradients (chapter 9)

### 9a. Emulated precision

The default run with weights' compute copies, activations at every matrix product and
gradients rounded to the formats (`scripts/training/precision.py`; float64 stands in for FP32
in the elementwise operations and the master weights). Three seeds.

| Recipe | Final validation loss (mean ± sd) | Seeds 1, 2, 3 | Gradient values flushed to zero | Steps skipped |
|---|---:|---|---:|---:|
| float64 (reference) | 2.277 ± 0.090 | 2.299, 2.354, 2.178 | 0.00% | 0.0 |
| FP32 | 2.316 ± 0.085 | 2.298, 2.408, 2.242 | 0.00% | 0.0 |
| BF16, FP32 master weights | 2.329 ± 0.059 | 2.269, 2.334, 2.386 | 0.00% | 0.0 |
| BF16, no master copy | 2.296 ± 0.027 | 2.325, 2.290, 2.272 | 0.00% | 0.0 |
| FP16, no loss scaling | 2.331 ± 0.027 | 2.340, 2.352, 2.301 | 4.56% | 0.0 |
| FP16, dynamic loss scaling | 2.288 ± 0.131 | 2.266, 2.429, 2.169 | 0.13% | 0.0 |
| FP8 (E4M3 / E5M2), no scaling | 2.957 ± 0.011 | 2.945, 2.959, 2.967 | 15.82% | 0.0 |
| FP8 (E4M3 / E5M2), per-tensor scaling | 2.574 ± 0.338 | 2.383, 2.965, 2.374 | 1.96% | 0.0 |

### 9b. Quantised gradients

Every weight gradient quantised to k-bit integers (one scale per tensor, full scale = its
largest magnitude) before clipping and the optimiser step; three seeds.

| Bits | Levels | Final validation loss (mean ± sd) | Seeds 1, 2, 3 |
|---:|---:|---:|---|
| 8 | 255 | 2.286 ± 0.069 | 2.321, 2.331, 2.207 |
| 4 | 15 | 2.413 ± 0.209 | 2.653, 2.269, 2.315 |
| 2 | 3 | 2.860 ± 0.182 | 2.650, 2.966, 2.964 |

## 10. Memory (chapter 10)

LLaMA 7B (Touvron et al. 2023, Table 2): 6,738,415,616 parameters by the
calculator; sequence 4,096, micro-batch 1. Bytes:

| Recipe | Weights | Gradients | Optimiser | Master | Activations | Logits | Peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed precision Adam, no recomputation | 13.5 GB | 13.5 GB | 53.9 GB | 27.0 GB | 104.2 GB | 0.8 GB | 199.5 GB |
| mixed precision Adam, selective recomputation | 13.5 GB | 13.5 GB | 53.9 GB | 27.0 GB | 18.3 GB | 0.8 GB | 116.1 GB |
| mixed precision Adam, full recomputation | 13.5 GB | 13.5 GB | 53.9 GB | 27.0 GB | 1.1 GB | 0.8 GB | 110.7 GB |
| FP32 Adam, no recomputation | 27.0 GB | 27.0 GB | 53.9 GB | 0.0 GB | 104.2 GB | 0.8 GB | 186.3 GB |

The Decoder site's model keeps 2320, 2320 values per sequence per block
for its backward pass (counted in the NumPy forward pass); Korthikanti et al.'s formula for the same shape
gives 2496.

## 11. The training-systems model (chapters 11, 12 and 15)

`scripts/training/systems.py` (ported to `src/lib/training/systems.ts`, matched to its fixtures):
collectives on an alpha-beta link model, ZeRO and per-device memory, a discrete-event simulation
of pipeline schedules, a step-time model, and failures and checkpoints. Nothing here trains; it
is arithmetic and seeded simulation.

### 11a. Validation against published runs

One constant is fitted: eta = 0.473, the fraction of peak the matrix products reach,
on Megatron-LM's 1.7B run (no tensor or pipeline parallelism). Every other row is a prediction.
Published numbers: Narayanan et al. 2021 (arXiv:2104.04473) Table 1 (A100, 312 TFLOP/s peak,
hardware FLOP/s with full recomputation by their eq. 3) and Grattafiori et al. 2024 (Llama 3,
arXiv:2407.21783) Table 4 (H100, BF16 MFU). Assumptions where the papers are silent:

- bytes: BF16 (2 bytes) for activations, parameters and gradients on the wire
- flops: Megatron rows: equation (3) of Narayanan et al. (with full recomputation); Llama rows: the PaLM MFU count 6N + 12 L H Q T per token (no recomputation)
- interleave: v = layers per pipeline rank (one layer per model chunk), the most interleaving the layer count allows; Llama 3 counts 128 layer slots (126 layers + the embedding and output stages, its §3.3.2)
- latency: 5 us per NVLink hop, 10 us per network hop (Llama 3 §3.3.3: 'latency up to tens of microseconds' across a large cluster)
- microBatch: 1 sequence per micro-batch (neither paper gives it for these runs)
- overlap: nothing overlaps: every collective, send and the gradient sync is exposed

| Run | GPUs | Layout (t, p, d, cp, v) | Micro-batches | Published TFLOP/s per GPU | Model | Error |
|---|---:|---|---:|---:|---:|---:|
| Megatron 1.7B (fitted) | 32 | 1, 1, 32, 1, 1 | 16 | 137 (44%) | 137.0 (43.9%) | +0.0% |
| Megatron 3.6B | 64 | 2, 1, 32, 1, 1 | 16 | 138 (44%) | 131.5 (42.1%) | -4.7% |
| Megatron 7.5B | 128 | 4, 1, 32, 1, 1 | 16 | 142 (46%) | 123.9 (39.7%) | -12.8% |
| Megatron 18.4B | 256 | 8, 1, 32, 1, 1 | 32 | 135 (43%) | 119.6 (38.3%) | -11.4% |
| Megatron 39.1B | 512 | 8, 2, 32, 1, 24 | 48 | 138 (44%) | 123.6 (39.6%) | -10.5% |
| Megatron 76.1B | 1,024 | 8, 4, 32, 1, 15 | 56 | 140 (45%) | 127.7 (40.9%) | -8.8% |
| Megatron 145.6B | 1,536 | 8, 8, 24, 1, 10 | 96 | 148 (47%) | 131.1 (42.0%) | -11.4% |
| Megatron 310.1B | 1,920 | 8, 16, 15, 1, 6 | 144 | 155 (50%) | 133.9 (42.9%) | -13.6% |
| Megatron 529.6B | 2,520 | 8, 35, 9, 1, 3 | 280 | 163 (52%) | 133.5 (42.8%) | -18.1% |
| Megatron 1008.0B | 3,072 | 8, 64, 6, 1, 2 | 512 | 163 (52%) | 132.7 (42.5%) | -18.6% |
| Llama 3 405B 8,192 GPUs, seq 8,192 | 8,192 | 8, 16, 64, 1, 8 | 32 | 430 (43%) | 385.8 (39.0%) | -10.3% |
| Llama 3 405B 16,384 GPUs, seq 8,192 | 16,384 | 8, 16, 128, 1, 8 | 16 | 400 (41%) | 360.1 (36.4%) | -10.0% |
| Llama 3 405B 16,384 GPUs, seq 131,072 | 16,384 | 8, 16, 8, 16, 8 | 16 | 380 (38%) | 379.4 (38.4%) | -0.2% |

**Validation band:** every held-out row within 18.6% of the published
throughput (Megatron rows 18.6%, Llama 3 rows 10.3%; mean absolute
error 10.8%). Every prediction is at or below the published number: the model
overlaps nothing, and one eta cannot follow the matrix products getting more efficient as the
models widen (Megatron's per-GPU throughput rises from 137 to 163 TFLOP/s; the model's stays near 130).

### 11b. Collectives, ZeRO and data parallelism

LLaMA 7B's gradient (6,738,149,376 parameters, 13.48 GB in BF16) on H100s:

| Where | Collective | Time | Bytes each GPU sends |
|---|---|---:|---:|
| NVLink, 8 GPUs | ring all-reduce | 52.5 ms | 23.58 GB |
| NVLink, 8 GPUs | binary-tree all-reduce (not pipelined) | 179.7 ms | 26.95 GB |
| NVLink, 8 GPUs | reduce-scatter | 26.2 ms | 11.79 GB |
| NVLink, 8 GPUs | all-gather | 26.2 ms | 11.79 GB |
| network, 64 GPUs | ring all-reduce | 531.9 ms | 26.53 GB |
| network, 64 GPUs | binary-tree all-reduce (not pipelined) | 3.23 s | 26.95 GB |
| network, 64 GPUs | reduce-scatter | 265.9 ms | 13.27 GB |
| network, 64 GPUs | all-gather | 265.9 ms | 13.27 GB |

ZeRO (Rajbhandari et al. 2020, arXiv:1910.02054): bytes per GPU for the same model's states
(2 + 2 + 12 bytes per parameter), and what each GPU sends per step:

| Data-parallel GPUs | Stage | Weights | Gradients | Optimiser | Total | Sent per step |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0 | 13.48 GB | 13.48 GB | 80.86 GB | 107.81 GB | 23.58 GB |
| 8 | 1 | 13.48 GB | 13.48 GB | 10.11 GB | 37.06 GB | 23.58 GB |
| 8 | 2 | 13.48 GB | 1.68 GB | 10.11 GB | 25.27 GB | 23.58 GB |
| 8 | 3 | 1.68 GB | 1.68 GB | 10.11 GB | 13.48 GB | 35.38 GB |
| 64 | 0 | 13.48 GB | 13.48 GB | 80.86 GB | 107.81 GB | 26.53 GB |
| 64 | 1 | 13.48 GB | 13.48 GB | 1.26 GB | 28.22 GB | 26.53 GB |
| 64 | 2 | 13.48 GB | 0.21 GB | 1.26 GB | 14.95 GB | 26.53 GB |
| 64 | 3 | 0.21 GB | 0.21 GB | 1.26 GB | 1.68 GB | 39.80 GB |

One data-parallel step of LLaMA 7B on 64 H100s, 1,048,576 tokens, eta as fitted:
compute 1.64 s, gradient all-reduce 531.9 ms; step 2.17 s with
nothing overlapped (MFU 35.7%), 1.64 s with the all-reduce hidden behind
the backward pass (MFU 47.3%).

### 11c. Failures and checkpoints (chapter 15)

Llama 3 405B: 466 interruptions in 54 days, 419 unexpected
(Grattafiori et al. 2024 §3.3.4), on up to 16,384 GPUs: a failure every 3.09 hours
for the job, about one per GPU every 5.8 years. Checkpoint time delta = 32.8 s
(16,384 GPUs x 4 GB at the storage fabric's 2 TB/s sustained (Llama 3 §3.3.1)); restart R = 600 s (illustrative: 10 minutes to restart and reload). Goodput over 54
simulated days, mean of 20 seeds, against the first-order formula:

| Checkpoint every | Simulated goodput | First-order formula |
|---:|---:|---:|
| 60 s | 61.11% | 61.02% |
| 85 s | 68.12% | 68.01% |
| 120 s | 74.02% | 73.89% |
| 170 s | 78.83% | 78.68% |
| 240 s | 82.47% | 82.30% |
| 339 s | 85.08% | 84.88% |
| 480 s | 86.77% | 86.55% |
| 679 s | 87.65% | 87.35% |
| 960 s | 87.73% | 87.32% |
| 1,358 s | 86.96% | 86.43% |
| 1,920 s | 85.33% | 84.55% |
| 2,715 s | 82.75% | 81.44% |
| 3,840 s | 78.78% | 76.71% |
| 5,431 s | 73.25% | 69.80% |

Best simulated interval 960 s; Young's tau* = sqrt(2 delta M) = 854 s
(the grid steps by a factor of 1.41).

### 11d. Stragglers and queueing (chapter 13)

Eight workers, each step 1 + 0.1 x Exp(1) time units (seeded); the mean time per step when
everyone waits for the slowest, syncing every step or every H steps:

| Sync every | Time per step | Mean worker's step |
|---:|---:|---:|
| 1 | 1.270 | 1.099 |
| 10 | 1.148 | 1.099 |
| 50 | 1.121 | 1.099 |

A sync message's mean wait on a link carrying background transfers (each 1 time unit, Poisson
arrivals): the M/D/1 formula W = load / (2 (1 - load)) against a seeded simulation:

| Load | Formula | Simulated |
|---:|---:|---:|
| 0.3 | 0.214 | 0.214 |
| 0.6 | 0.750 | 0.765 |
| 0.9 | 4.500 | 4.518 |

## 12. Pipeline schedules (chapter 12)

A discrete-event simulation of every forward and backward on every device (backward = 2 x forward),
against the closed forms of Narayanan et al. (2021) §2.2: bubble = (p - 1) / m, interleaved (p - 1) / (v m).

| Schedule | Devices p | Micro-batches m | Chunks v | Simulated bubble | Closed form |
|---|---:|---:|---:|---:|---:|
| gpipe | 4 | 8 | 1 | 0.3750 | 0.3750 |
| 1f1b | 4 | 8 | 1 | 0.3750 | 0.3750 |
| interleaved | 4 | 8 | 2 | 0.1875 | 0.1875 |
| interleaved | 4 | 8 | 4 | 0.0938 | 0.0938 |
| 1f1b | 8 | 32 | 1 | 0.2188 | 0.2188 |
| interleaved | 8 | 32 | 4 | 0.0547 | 0.0547 |
| gpipe | 16 | 32 | 1 | 0.4688 | 0.4688 |
| 1f1b | 16 | 64 | 1 | 0.2344 | 0.2344 |

Largest difference: 0.0e+00.

## 13. Training over slow links (chapter 13)

The Decoder site's model, 4 simulated workers with 4 windows each per step (the default
run's 16), 2,000 steps, warmup 50 + cosine to 0.1 of the peak, each worker's gradient clipped
to 1.0, weight decay 0.01 on matrices; three seeds. Compressors keep 1 in 16: top-k per tensor
(momentum 0.9), DeMo per DCT chunk of up to 64 (beta 0.999, sign update);
DiLoCo's outer step is Nesterov momentum (lr 0.7, momentum 0.9). Each method's
learning rate is the best mean of three tried. Bytes per worker per step, BF16 values; ring
collectives (all-reduce for dense and DiLoCo, all-gather for the compressors).

| Method | Best lr | Final validation loss (mean ± sd) | Steps to 2.6 (seeds) | Payload | Wire | Wire vs dense |
|---|---:|---:|---|---:|---:|---:|
| dense | 0.01 | 2.277 ± 0.090 | 1150, 1300, 1000 | 10,752 B | 16,128 B | 1.0x less |
| topk | 1.0 | 2.225 ± 0.039 | 550, 550, 500 | 1,344 B | 4,032 B | 4.0x less |
| demo | 0.03 | 2.073 ± 0.047 | 350, 300, 300 | 1,008 B | 3,024 B | 5.3x less |
| diloco H=10 | 0.03 | 2.578 ± 0.078 | not reached, 1700, 1950 | 1,075 B | 1,613 B | 10.0x less |
| diloco H=50 | 0.1 | 2.706 ± 0.094 | not reached, not reached, not reached | 215 B | 323 B | 50.0x less |

Every learning rate tried (mean ± sd over seeds):

| Method | lr | Final validation loss |
|---|---:|---:|
| dense | 0.003 | 2.577 ± 0.009 |
| dense | 0.01 | 2.277 ± 0.090 |
| dense | 0.03 | 2.298 ± 0.168 |
| topk | 0.3 | 2.264 ± 0.017 |
| topk | 1.0 | 2.225 ± 0.039 |
| topk | 3.0 | diverged |
| demo | 0.003 | 2.531 ± 0.329 |
| demo | 0.01 | 2.128 ± 0.016 |
| demo | 0.03 | 2.073 ± 0.047 |
| diloco H=10 | 0.01 | 2.703 ± 0.085 |
| diloco H=10 | 0.03 | 2.578 ± 0.078 |
| diloco H=10 | 0.1 | 2.784 ± 0.031 |
| diloco H=50 | 0.01 | 2.807 ± 0.016 |
| diloco H=50 | 0.03 | 2.863 ± 0.075 |
| diloco H=50 | 0.1 | 2.706 ± 0.094 |

## 14. A toy IsoFLOP sweep (chapter 14)

The Decoder site's model at widths 6, 8, 12, 16, 24, 32, 48, 64 (2 heads, FFN twice the width, 2 blocks),
each trained for exactly the tokens a budget C = 6 N D allows (16 windows of 8 per step,
AdamW peak lr 0.01, warmup then cosine to 10% over the run); three seeds. N counts every parameter.
A parabola in log10 N through each budget's lowest mean and its two neighbours gives N_opt;
a line through log N_opt against log C gives the exponent a.

| Budget C (FLOPs) | Width | N | Steps | Tokens | Final validation loss (mean ± sd) |
|---:|---:|---:|---:|---:|---:|
| 3e+09 | 6 | 1,056 | 3,699 | 473,472 | 2.349 ± 0.008 |
| 3e+09 | 8 | 1,664 | 2,348 | 300,544 | 2.373 ± 0.058 |
| 3e+09 | 12 | 3,264 | 1,197 | 153,216 | 2.610 ± 0.203 |
| 3e+09 | 16 | 5,376 | 727 | 93,056 | 2.948 ± 0.004 |
| 3e+09 | 24 | 11,136 | 351 | 44,928 | 2.951 ± 0.004 |
| 3e+09 | 32 | 18,944 | 206 | 26,368 | 2.949 ± 0.002 |
| 1e+10 | 6 | 1,056 | 12,330 | 1,578,240 | 2.208 ± 0.025 |
| 1e+10 | 8 | 1,664 | 7,825 | 1,001,600 | 2.104 ± 0.035 |
| 1e+10 | 12 | 3,264 | 3,989 | 510,592 | 2.080 ± 0.058 |
| 1e+10 | 16 | 5,376 | 2,422 | 310,016 | 2.220 ± 0.120 |
| 1e+10 | 24 | 11,136 | 1,169 | 149,632 | 2.768 ± 0.186 |
| 1e+10 | 32 | 18,944 | 687 | 87,936 | 2.952 ± 0.004 |
| 1e+10 | 48 | 40,704 | 320 | 40,960 | 2.953 ± 0.005 |
| 1e+10 | 64 | 70,656 | 184 | 23,552 | 2.950 ± 0.003 |
| 3e+10 | 6 | 1,056 | 36,991 | 4,734,848 | 2.152 ± 0.020 |
| 3e+10 | 8 | 1,664 | 23,475 | 3,004,800 | 1.999 ± 0.021 |
| 3e+10 | 12 | 3,264 | 11,968 | 1,531,904 | 1.891 ± 0.018 |
| 3e+10 | 16 | 5,376 | 7,266 | 930,048 | 1.922 ± 0.050 |
| 3e+10 | 24 | 11,136 | 3,508 | 449,024 | 2.144 ± 0.131 |
| 3e+10 | 32 | 18,944 | 2,062 | 263,936 | 2.181 ± 0.107 |
| 3e+10 | 48 | 40,704 | 960 | 122,880 | 2.846 ± 0.100 |
| 3e+10 | 64 | 70,656 | 553 | 70,784 | 2.924 ± 0.048 |
| 1e+11 | 12 | 3,264 | 39,892 | 5,106,176 | 1.825 ± 0.020 |
| 1e+11 | 16 | 5,376 | 24,220 | 3,100,160 | 1.753 ± 0.018 |
| 1e+11 | 24 | 11,136 | 11,693 | 1,496,704 | 1.790 ± 0.053 |
| 1e+11 | 32 | 18,944 | 6,873 | 879,744 | 1.863 ± 0.063 |
| 1e+11 | 48 | 40,704 | 3,199 | 409,472 | 2.094 ± 0.034 |
| 1e+11 | 64 | 70,656 | 1,843 | 235,904 | 2.398 ± 0.147 |

| Budget | N_opt | D_opt (tokens) | Minimum inside the sizes tried |
|---:|---:|---:|---|
| 3e+09 | – | – | no |
| 1e+10 | 2,493 | 668,646 | yes |
| 3e+10 | 3,550 | 1,408,285 | yes |
| 1e+11 | 6,590 | 2,529,229 | yes |

Fitted exponent a = 0.42 (N_opt ~ C^a; b = 1 - a = 0.58), from the budgets with a minimum
inside the sizes tried. Published: Chinchilla 0.5 / 0.49 / 0.46
(Hoffmann et al. 2022, Table 2, approaches 1-3), Kaplan et al. 2020 0.73; Llama 3 fitted
alpha = 0.53 for the optimal number of tokens.

## 15. Running the job (chapter 15)

The default run, step for step, as a dashboard sees it (final validation loss 2.2993, the
default run's). Every 10th step:

| Step | Training loss | Gradient norm (before clipping) | Learning rate | Update / weight (all) | Update / weight (block 1 FFN W1) |
|---:|---:|---:|---:|---:|---:|
| 0 | 4.173 | 0.854 | 0.00020 | 1.58e-03 | 1.03e-02 |
| 200 | 2.953 | 0.340 | 0.00987 | 4.44e-03 | 3.18e-03 |
| 400 | 2.998 | 0.383 | 0.00930 | 5.16e-03 | 8.13e-03 |
| 600 | 3.071 | 0.287 | 0.00835 | 8.32e-03 | 1.03e-02 |
| 800 | 2.930 | 0.299 | 0.00710 | 5.15e-03 | 8.75e-03 |
| 1000 | 2.834 | 0.397 | 0.00568 | 6.08e-03 | 7.07e-03 |
| 1200 | 2.562 | 0.535 | 0.00425 | 2.82e-03 | 3.96e-03 |
| 1400 | 2.599 | 0.677 | 0.00294 | 1.86e-03 | 2.59e-03 |
| 1600 | 2.519 | 0.614 | 0.00190 | 1.10e-03 | 1.97e-03 |
| 1800 | 2.328 | 0.764 | 0.00123 | 7.73e-04 | 1.43e-03 |
| 1999 | 2.366 | 0.808 | 0.00100 | 6.43e-04 | 1.20e-03 |

### Stopping at step 1,000 and resuming (three seeds)

Resuming with the weights, AdamW's state and the data position continues bit for bit: **yes** (every seed).

| Resumed with | Validation loss 50 steps after resuming | Final validation loss (mean ± sd) |
|---|---:|---:|
| never stopped | 2.643 | 2.277 ± 0.090 |
| weights + optimiser + data position | 2.643 | 2.277 ± 0.090 |
| weights only (fresh AdamW) | 2.652 | 2.327 ± 0.165 |
| weights + optimiser, data restarted | 2.636 | 2.286 ± 0.102 |

## 16. After pretraining (chapter 16)

The trained model on a prompt and a response: `who ran? ` + `the rabbit.`. Mean loss on the
response's characters (the SFT loss, prompt masked): 2.670 nats; on every character
after the first: 2.721. Log probability of two continuations of `said the `:
`queen.` -18.19, `qxzjv.` -42.65.
