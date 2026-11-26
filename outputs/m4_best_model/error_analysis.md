# Error analysis of the final model (written by m4_best_model/error_analysis.py)

787 test lesions, 158 misclassified (20.1%).

## Confusion matrix (rows: truth)

| true | pred Benign | pred Indeterminate | pred Malignant |
|---|---|---|---|
| Benign | 155 | 6 | 62 |
| Indeterminate | 1 | 6 | 11 |
| Malignant | 50 | 28 | 468 |

## Per class

| class | test lesions | precision | recall | F1 |
|---|---|---|---|---|
| Benign | 223 | 0.752 | 0.695 | 0.723 |
| Indeterminate | 18 | 0.150 | 0.333 | 0.207 |
| Malignant | 546 | 0.865 | 0.857 | 0.861 |

## Mistakes by 11-class diagnosis

| dx | lesions | wrong | error_rate | called Benign | called Indeterminate | called Malignant |
|---|---|---|---|---|---|---|
| BKL | 83 | 44 | 0.530 | 39 | 4 | 40 |
| MEL | 68 | 33 | 0.485 | 25 | 8 | 35 |
| BCC | 379 | 29 | 0.077 | 19 | 10 | 350 |
| AKIEC | 45 | 19 | 0.422 | 5 | 9 | 31 |
| NV | 112 | 12 | 0.107 | 100 | 0 | 12 |
| SCCKA | 71 | 9 | 0.127 | 2 | 7 | 62 |
| INF | 7 | 5 | 0.714 | 2 | 2 | 3 |
| BEN_OTH | 7 | 3 | 0.429 | 4 | 0 | 3 |
| VASC | 7 | 3 | 0.429 | 4 | 0 | 3 |
| DF | 7 | 1 | 0.143 | 6 | 0 | 1 |
| MAL_OTH | 1 | 0 | 0.000 | 0 | 0 | 1 |

## The 10 images of figures/misclassified_gallery.png

| isic_id | dx | true | pred | confidence | MONET_pigmented |
|---|---|---|---|---|---|
| ISIC_5078639 | BCC | Malignant | Benign | 0.989 | 0.247 |
| ISIC_5019119 | MEL | Malignant | Benign | 0.967 | 0.685 |
| ISIC_2867998 | MEL | Malignant | Benign | 0.958 | 0.775 |
| ISIC_1201025 | MEL | Malignant | Benign | 0.953 | 0.901 |
| ISIC_1909053 | MEL | Malignant | Benign | 0.952 | 0.658 |
| ISIC_8260850 | BKL | Benign | Malignant | 0.998 | 0.068 |
| ISIC_7020196 | BKL | Benign | Malignant | 0.994 | 0.202 |
| ISIC_2110451 | BKL | Benign | Malignant | 0.992 | 0.143 |
| ISIC_5962854 | VASC | Benign | Malignant | 0.991 | 0.243 |
| ISIC_1435977 | BKL | Benign | Malignant | 0.986 | 0.117 |

Mean confidence: 0.85 on correct answers, 0.72 on mistakes. Mistakes made with a confidence of 0.95 or more: 18.

## Error rate by pigmentation, inside each true class

| true | pigmentation | lesions | wrong | error_rate |
|---|---|---|---|---|
| Benign | less pigmented half | 70 | 39 | 0.557 |
| Benign | more pigmented half | 153 | 29 | 0.190 |
| Malignant | less pigmented half | 316 | 27 | 0.085 |
| Malignant | more pigmented half | 230 | 51 | 0.222 |

## Error rate by Otsu mask, inside each true class

| true | otsu mask | lesions | wrong | error_rate |
|---|---|---|---|---|
| Benign | mask failed | 50 | 35 | 0.700 |
| Benign | mask plausible | 173 | 33 | 0.191 |
| Malignant | mask failed | 303 | 26 | 0.086 |
| Malignant | mask plausible | 243 | 52 | 0.214 |
