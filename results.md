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
| Backward = PyTorch autograd, perturbed weights | max err / (1 + max abs grad) 1.1e-16; loss diff 0.0e+00 | 1e-12 |
| Backward = central differences (h = 1e-6), at initialisation, 552 entries | max abs err 7.2e-10 | 1e-8 + 1e-6 abs grad |
| Backward = central differences (h = 1e-6), perturbed weights, 552 entries | max abs err 7.6e-10 | 1e-8 + 1e-6 abs grad |
| `sgd` = torch.optim over 25 steps | max abs err 0.0e+00 | 1e-12 |
| `momentum` = torch.optim over 25 steps | max abs err 0.0e+00 | 1e-12 |
| `adam` = torch.optim over 25 steps | max abs err 2.2e-16 | 1e-12 |
| `adam_l2` = torch.optim over 25 steps | max abs err 2.2e-16 | 1e-12 |
| `adamw` = torch.optim over 25 steps | max abs err 1.1e-16 | 1e-12 |
| Cosine schedule = CosineAnnealingLR | max abs err 3.5e-18 | 1e-15 |

## 4. The default run (chapter 1)

2,000 steps of 16 windows of 8 characters; adamw (beta1 0.9, beta2 0.99, weight decay 0.01); learning rate 0.01, 50 warmup steps, cosine to 0.1 of it; clip at global norm 1.0; data seed 1. 8.2 s on the machine above (NumPy).

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

Each optimiser at three constant learning rates, 1,500 steps, the same batches
(data seed 1, 16 windows), clip 1.0, beta2 0.99 for Adam and AdamW, AdamW weight decay 0.01.

| Optimiser | Learning rate | Validation loss at 1,500 | Best for this optimiser |
|---|---:|---:|:---:|
| sgd | 0.1 | 2.8987 |  |
| sgd | 0.3 | 2.3699 |  |
| sgd | 1.0 | 2.2512 | yes |
| momentum | 0.03 | 2.3601 |  |
| momentum | 0.1 | 2.2047 | yes |
| momentum | 0.3 | 2.3157 |  |
| adam | 0.003 | 2.3272 |  |
| adam | 0.01 | 2.1681 | yes |
| adam | 0.03 | 2.2707 |  |
| adamw | 0.003 | 2.3343 |  |
| adamw | 0.01 | 2.1923 | yes |
| adamw | 0.03 | 2.3443 |  |
