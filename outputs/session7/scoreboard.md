# Final scoreboard, Sessions 5 to 7 (written by session7_cnns.ipynb, section 12.2)

Macro-F1 per lesion on the 786 validation and 787 test lesions of the Milestone 1 split. Every choice was made on validation; test was scored once. 95% CI: stratified bootstrap of the test lesions, 2,000 resamples.

| rank | session | model | input | trainable weights | val macro-F1 | test macro-F1 | test 95% CI |
|---|---|---|---|---|---|---|---|
| 1 | 7 | ResNet-18 fine-tuned: whole network | derm image, 224x224 | 11,178,051 | 0.682 | 0.597 | 0.549 to 0.645 |
| 2 | 7 | ResNet-18 fine-tuned: layer4 | derm image, 224x224 | 8,395,267 | 0.681 | 0.634 | 0.584 to 0.682 |
| 3 | 7 | ResNet-18 fine-tuned: layer3 + layer4 | derm image, 224x224 | 10,494,979 | 0.676 | 0.627 | 0.570 to 0.683 |
| 4 | 5 | GradBoost (best Session 5 macro-F1) | both images + metadata, 269 features |  | 0.667 | 0.599 | 0.544 to 0.653 |
| 5 | 5 | Random forest | derm image, 131 features |  | 0.621 | 0.574 | 0.504 to 0.643 |
| 6 | 6 | MLP on handcrafted features | derm image, 131 features | 8,643 | 0.611 | 0.561 | 0.514 to 0.608 |
| 7 | 5 | Logistic regression (Session 5 choice) | both images + metadata, 269 features |  | 0.572 | 0.532 | 0.499 to 0.562 |
| 8 | 7 | ResNet-18 frozen + linear head | derm image, 224x224 | 1,539 | 0.567 | 0.537 | 0.507 to 0.568 |
| 9 | 7 | CNN from scratch | derm image, 224x224 | 306,147 | 0.531 | 0.488 | 0.457 to 0.524 |
| 10 | 6 | MLP on raw pixels | derm image, 32x32 pixels (3,072) | 196,867 | 0.521 | 0.502 | 0.454 to 0.560 |
| 11 | ref | always Malignant | none | 0 | 0.273 | 0.273 | 0.273 to 0.273 |
