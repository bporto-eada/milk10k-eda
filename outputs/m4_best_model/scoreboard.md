# Scoreboard, Milestones 2 to 4 (written by m4_best_model/compare_models.py)

Macro-F1 per lesion on the same 786 validation and 787 test lesions, target diagnosis_1 (3 classes). M2 and M3 rows as reported in Session 7; the M4 row is new. Sorted by validation, which is what every choice was made on. Each test score was computed once. 95% CI: bootstrap of the test lesions.

| milestone | model | input | trainable weights | val macro-F1 | test macro-F1 | test 95% CI | final |
|---|---|---|---|---|---|---|---|
| M3 | ResNet-18 fine-tuned: whole network | derm image, 224x224 | 11,178,051 | 0.682 | 0.597 | 0.549 to 0.645 | best -> final |
| M3 | ResNet-18 fine-tuned: layer4 | derm image, 224x224 | 8,395,267 | 0.681 | 0.634 | 0.584 to 0.682 |  |
| M3 | ResNet-18 fine-tuned: layer3 + layer4 | derm image, 224x224 | 10,494,979 | 0.676 | 0.627 | 0.570 to 0.683 |  |
| M2 | GradBoost (best Session 5 macro-F1) | both images + metadata, 269 features |  | 0.667 | 0.599 | 0.544 to 0.653 |  |
| M2 | Random forest | derm image, 131 features |  | 0.621 | 0.574 | 0.504 to 0.643 |  |
| M3 | MLP on handcrafted features | derm image, 131 features | 8,643 | 0.611 | 0.561 | 0.514 to 0.608 |  |
| M4 | ViT-B/16 frozen + linear head | derm image, 224x224 | 2,307 | 0.574 | 0.592 | 0.556 to 0.629 |  |
| M2 | Logistic regression (Session 5 choice) | both images + metadata, 269 features |  | 0.572 | 0.532 | 0.499 to 0.562 |  |
| M3 | ResNet-18 frozen + linear head | derm image, 224x224 | 1,539 | 0.567 | 0.537 | 0.507 to 0.568 |  |
| M3 | CNN from scratch | derm image, 224x224 | 306,147 | 0.531 | 0.488 | 0.457 to 0.524 |  |
| M3 | MLP on raw pixels | derm image, 32x32 pixels (3,072) | 196,867 | 0.521 | 0.502 | 0.454 to 0.560 |  |
| ref | always Malignant | none | 0 | 0.273 | 0.273 | 0.273 to 0.273 |  |
