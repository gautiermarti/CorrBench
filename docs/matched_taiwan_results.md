# Audited categorical Taiwan comparison — 2026-10-05

Compared with DiffeoCFM, CorrFlow has lower structural and portfolio distances under this fixed protocol. DiffeoCFM has slightly higher structural coverage; both remain readily distinguishable from real data. This is not universal superiority.

| Method | Structural SWD ↓ | Precision | Recall | Portfolio SWD ↓ | C2ST AUC (.5 ideal) |
|---|---:|---:|---:|---:|---:|
| CorrFlow | 0.4709 ± 0.0615 | 0.5239 | 0.3204 | 0.6243 ± 0.0936 | 0.9991 |
| DiffeoCFM | 0.7512 ± 0.0188 | 0.3039 | 0.3549 | 0.8620 ± 0.0263 | 0.9968 |
| Transformed Gaussian | 0.8709 ± 0.0058 | 0.3011 | 0.5043 | 0.8107 ± 0.0133 | 0.9955 |
| Training replay | 0.8770 ± 0.0083 | 0.4499 | 0.5896 | 1.1066 ± 0.0092 | 0.9554 |
| cCorrGAN reconstruction | 4.0390 ± 0.1537 | 0.0000 | 0.4007 | 818.7222 ± 824.7008 | 1.0000 |

All evaluated matrices pass validity checks after required image-model repair; this is not a raw-generator validity claim. The audit verifies frozen source/data/output hashes, all completed budgets, identical case indices and labels, replay membership, validation-minimum selection and reproduction of all 15 native validation scores (maximum absolute difference 0).

CorrFlow selected epochs: 1, 3, 18, 6, 4. DiffeoCFM: 170, 190, 140, 200, 130. Final-weight structural SWD is .8363 ± .0679 for CorrFlow and .7980 ± .0221 for DiffeoCFM. The final-weight ordering therefore differs from the predeclared validation-selected comparison; preserve both.

DiffeoCFM improves structural distance over its fitted Gaussian prior, but its portfolio distance worsens (.8620 versus .8107) and its recall falls (.3549 versus .5043). Prior-versus-endpoint reporting matters.

Ordered GFT distance favors DiffeoCFM (.1082 versus .3348), while permutation-orbit GFT distance favors CorrFlow (.0767 versus .0856). The former retains Ward coordinate order; the latter compares independently relabelled matrices. Neither metric alone certifies useful generation.

The nearest-training diagnostic searches a sampled training reference, so its approximate exact-match fraction is not full-corpus membership. The separate audit confirms every replay matrix is an exact training draw. No privacy claim follows from neural outputs having no exact match there.

Limits: previously inspected Taiwan periods; categorical conditions only; unequal model-specific training budgets and source laws; one paired generation seed per neural training seed. Seed SD is not uncertainty over independent market episodes. Concurrent timings do not establish speed rankings. The complete five-seed cCorrGAN reconstruction is included. Completing its declared budget does not establish faithful original training or recover the original published fidelity.
