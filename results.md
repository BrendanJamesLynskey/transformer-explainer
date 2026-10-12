# Training results (Part 2)

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

Part 1's model: seq_len 8, d_model 16, 2 heads, d_ff 32, 2 blocks, vocabulary 64, seed 42; 5,376 parameters. A uniform guess scores ln 64 = 4.1589.

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
| `adam` = torch.optim over 25 steps | max abs err 2.2e-16 | 1e-12 |
| `adam_l2` = torch.optim over 25 steps | max abs err 2.2e-16 | 1e-12 |
| `adamw` = torch.optim over 25 steps | max abs err 1.1e-16 | 1e-12 |
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

Part 1's model keeps 2320, 2320 values per sequence per block
for its backward pass (counted in the NumPy forward pass); Korthikanti et al.'s formula for the same shape
gives 2496.
