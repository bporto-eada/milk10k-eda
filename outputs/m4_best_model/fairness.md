# Fairness check of the final model (written by m4_best_model/fairness.py)

787 test lesions. Skin tone: MILK10k's skin_tone_class, 0 = very dark to 5 = very light. macro-F1 averages the classes that occur in a group (as truth or as prediction), so groups with and without Indeterminate lesions are not comparable on it; the recall of Malignant and of Benign is. 95% interval: 2,000 bootstrap resamples of the group's malignant (or benign) lesions.

| attribute | group | lesions | Benign / Indeterminate / Malignant | macro-F1 | Malignant found | recall Malignant | recall Malignant 95% low | recall Malignant 95% high | Benign found | recall Benign | recall Benign 95% low | recall Benign 95% high | Indeterminate found | called Indeterminate | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| skin_tone_class | 0 | 1 | 1 / 0 / 0 | 0.000 | 0 of 0 |  |  |  | 0 of 1 | 0.000 | 0.000 | 0.000 | 0 of 0 | 1 | fewer than 50 lesions; a class has no lesion here: macro-F1 not comparable |
| skin_tone_class | 1 | 9 | 2 / 1 / 6 | 0.556 | 2 of 6 | 0.333 | 0.000 | 0.667 | 2 of 2 | 1.000 | 1.000 | 1.000 | 1 of 1 | 3 | fewer than 50 lesions |
| skin_tone_class | 2 | 72 | 37 / 0 / 35 | 0.439 | 20 of 35 | 0.571 | 0.400 | 0.743 | 27 of 37 | 0.730 | 0.595 | 0.865 | 0 of 0 | 2 | a class has no lesion here: macro-F1 not comparable |
| skin_tone_class | 3 | 466 | 111 / 14 / 341 | 0.589 | 299 of 341 | 0.877 | 0.842 | 0.909 | 66 of 111 | 0.595 | 0.495 | 0.685 | 5 of 14 | 27 |  |
| skin_tone_class | 4 | 170 | 50 / 3 / 117 | 0.575 | 104 of 117 | 0.889 | 0.829 | 0.940 | 41 of 50 | 0.820 | 0.720 | 0.920 | 0 of 3 | 6 |  |
| skin_tone_class | 5 | 69 | 22 / 0 / 47 | 0.596 | 43 of 47 | 0.915 | 0.830 | 0.979 | 19 of 22 | 0.864 | 0.727 | 1.000 | 0 of 0 | 1 | a class has no lesion here: macro-F1 not comparable |
| tone group | darker (0 to 2) | 82 | 40 / 1 / 41 | 0.526 | 22 of 41 | 0.537 | 0.390 | 0.683 | 29 of 40 | 0.725 | 0.575 | 0.850 | 1 of 1 | 6 |  |
| tone group | lighter (3 to 5) | 705 | 183 / 17 / 505 | 0.602 | 446 of 505 | 0.883 | 0.855 | 0.911 | 126 of 183 | 0.689 | 0.623 | 0.754 | 5 of 17 | 34 |  |
| sex | female | 269 | 99 / 5 / 165 | 0.538 | 140 of 165 | 0.848 | 0.788 | 0.897 | 75 of 99 | 0.758 | 0.667 | 0.838 | 0 of 5 | 6 |  |
| sex | male | 518 | 124 / 13 / 381 | 0.602 | 328 of 381 | 0.861 | 0.824 | 0.895 | 80 of 124 | 0.645 | 0.565 | 0.726 | 6 of 13 | 34 |  |

## Malignant lesions by diagnosis and skin-tone group

| tone group | dx | lesions | found | share of the group's malignant lesions | recall |
|---|---|---|---|---|---|
| darker (0 to 2) | AKIEC | 1 | 1 | 0.024 | 1.000 |
| darker (0 to 2) | BCC | 18 | 12 | 0.439 | 0.667 |
| darker (0 to 2) | MAL_OTH | 1 | 1 | 0.024 | 1.000 |
| darker (0 to 2) | MEL | 20 | 8 | 0.488 | 0.400 |
| darker (0 to 2) | SCCKA | 1 | 0 | 0.024 | 0.000 |
| lighter (3 to 5) | AKIEC | 26 | 19 | 0.051 | 0.731 |
| lighter (3 to 5) | BCC | 361 | 338 | 0.715 | 0.936 |
| lighter (3 to 5) | MEL | 48 | 27 | 0.095 | 0.562 |
| lighter (3 to 5) | SCCKA | 70 | 62 | 0.139 | 0.886 |
