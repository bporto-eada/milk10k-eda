# Split verification

Seed 42, created 2026-10-05 by `python milestone1.py splits`.

## Sizes

| split | lesions | images | share_pct |
|---|---|---|---|
| train | 3667 | 7334 | 69.98 |
| val | 786 | 1572 | 15.0 |
| test | 787 | 1574 | 15.02 |

Lesion overlap between splits: {'train-val': 0, 'train-test': 0, 'val-test': 0}  
Lesions without exactly 2 images in one split: 0

## 11-class proportions (% of lesions)

| dx | train | val | test | all |
|---|---|---|---|---|
| AKIEC | 5.81 | 5.73 | 5.72 | 5.78 |
| BCC | 48.13 | 48.09 | 48.16 | 48.13 |
| BEN_OTH | 0.85 | 0.76 | 0.89 | 0.84 |
| BKL | 10.36 | 10.31 | 10.55 | 10.38 |
| DF | 1.01 | 1.02 | 0.89 | 0.99 |
| INF | 0.95 | 1.02 | 0.89 | 0.95 |
| MAL_OTH | 0.14 | 0.38 | 0.13 | 0.17 |
| MEL | 8.56 | 8.65 | 8.64 | 8.59 |
| NV | 14.21 | 14.38 | 14.23 | 14.24 |
| SCCKA | 9.05 | 8.91 | 9.02 | 9.03 |
| VASC | 0.93 | 0.76 | 0.89 | 0.9 |

Largest deviation from the overall share: 0.21 pp

## diagnosis_1 proportions (% of lesions)

| diagnosis_1 | train | val | test | all |
|---|---|---|---|---|
| Benign | 28.31 | 28.24 | 28.34 | 28.3 |
| Indeterminate | 2.37 | 2.29 | 2.29 | 2.35 |
| Malignant | 69.32 | 69.47 | 69.38 | 69.35 |

Largest deviation from the overall share: 0.11 pp
