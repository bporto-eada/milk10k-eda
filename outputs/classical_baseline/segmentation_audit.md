# Lesion-mask audit

Manual verdicts on 120 images (20 lesions per class, both views):

| diagnosis_1 | view | good | partial | failed | fallback ellipse |
|---|---|---|---|---|---|
| Benign | clinical | 10 | 1 | 9 | 0 |
| Benign | dermoscopic | 15 | 2 | 2 | 1 |
| Indeterminate | clinical | 1 | 3 | 15 | 1 |
| Indeterminate | dermoscopic | 2 | 5 | 11 | 2 |
| Malignant | clinical | 4 | 1 | 15 | 0 |
| Malignant | dermoscopic | 6 | 5 | 7 | 2 |

Automatic flag (mask on more than 5% of the image border, or fallback ellipse) against the manual verdicts (bad = failed or fallback): agreement 0.867, sensitivity 0.938, specificity 0.782.

Flagged share of all 10,480 images: 57.6%. By class and view (%):

| diagnosis_1 | clinical | dermoscopic |
|---|---|---|
| Benign | 52.5 | 24.0 |
| Indeterminate | 86.2 | 61.0 |
| Malignant | 75.0 | 54.8 |

By 11-class label (%):

| dx | flagged_pct |
|---|---|
| VASC | 16.0 |
| NV | 21.6 |
| MEL | 24.4 |
| MAL_OTH | 33.3 |
| DF | 47.1 |
| INF | 53.0 |
| BEN_OTH | 55.7 |
| BKL | 59.4 |
| BCC | 68.8 |
| AKIEC | 76.1 |
| SCCKA | 78.3 |
