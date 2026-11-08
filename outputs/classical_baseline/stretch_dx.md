# Stretch target: 11-class dx

Same feature set and model family as the chosen 3-class baseline (both C+T+S+meta + LogReg), hyper-parameters re-tuned by CV on train, refitted on train+val, scored once on test.

Test: accuracy 0.511, balanced accuracy 0.420, macro-F1 0.329, macro ROC-AUC 0.830. Majority class (BCC): accuracy 0.482, balanced accuracy 0.091.

| class | test_lesions | recall | precision | note |
|---|---|---|---|---|
| AKIEC | 45 | 0.444 | 0.204 |  |
| BCC | 379 | 0.515 | 0.855 |  |
| BEN_OTH | 7 | 0.000 | 0.000 | fewer than 10 test lesions: not estimable |
| BKL | 83 | 0.217 | 0.281 |  |
| DF | 7 | 0.571 | 0.121 | fewer than 10 test lesions: not estimable |
| INF | 7 | 0.286 | 0.080 | fewer than 10 test lesions: not estimable |
| MAL_OTH | 1 | 0.000 | 0.000 | fewer than 10 test lesions: not estimable |
| MEL | 68 | 0.544 | 0.500 |  |
| NV | 112 | 0.652 | 0.745 |  |
| SCCKA | 71 | 0.676 | 0.425 |  |
| VASC | 7 | 0.714 | 0.263 | fewer than 10 test lesions: not estimable |
