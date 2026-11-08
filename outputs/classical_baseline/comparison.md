# Feature sets x models, selection on VAL

Hyper-parameters by 5-fold CV on the train lesions (balanced accuracy); each tuned model scored once on the val lesions.

| feature_set | model | n_features | CV bal.acc | VAL bal.acc | VAL macro-F1 | VAL macro-AUC | rec Ben | rec Ind | rec Mal |
|---|---|---|---|---|---|---|---|---|---|
| derm C+T+S | LogReg | 131 | 0.628 | 0.702 | 0.532 | 0.820 | 0.653 | 0.778 | 0.676 |
| derm C+T+S | kNN | 131 | 0.500 | 0.575 | 0.624 | 0.732 | 0.495 | 0.333 | 0.896 |
| derm C+T+S | SVM-RBF | 131 | 0.631 | 0.671 | 0.558 | 0.817 | 0.653 | 0.611 | 0.749 |
| derm C+T+S | RandomForest | 131 | 0.551 | 0.589 | 0.621 | 0.830 | 0.559 | 0.333 | 0.875 |
| derm C+T+S | GradBoost | 131 | 0.591 | 0.643 | 0.586 | 0.816 | 0.640 | 0.500 | 0.789 |
| clinical C+T+S | LogReg | 131 | 0.585 | 0.622 | 0.502 | 0.776 | 0.644 | 0.556 | 0.665 |
| clinical C+T+S | kNN | 131 | 0.448 | 0.461 | 0.483 | 0.715 | 0.383 | 0.111 | 0.888 |
| clinical C+T+S | SVM-RBF | 131 | 0.562 | 0.558 | 0.491 | 0.762 | 0.662 | 0.333 | 0.679 |
| clinical C+T+S | RandomForest | 131 | 0.546 | 0.549 | 0.557 | 0.823 | 0.653 | 0.167 | 0.828 |
| clinical C+T+S | GradBoost | 131 | 0.561 | 0.607 | 0.555 | 0.799 | 0.667 | 0.389 | 0.766 |
| both colour | LogReg | 144 | 0.607 | 0.619 | 0.483 | 0.776 | 0.622 | 0.611 | 0.625 |
| both colour | kNN | 144 | 0.481 | 0.445 | 0.448 | 0.716 | 0.450 | 0.000 | 0.885 |
| both colour | SVM-RBF | 144 | 0.582 | 0.569 | 0.506 | 0.768 | 0.671 | 0.333 | 0.703 |
| both colour | RandomForest | 144 | 0.545 | 0.520 | 0.541 | 0.823 | 0.581 | 0.111 | 0.868 |
| both colour | GradBoost | 144 | 0.591 | 0.650 | 0.607 | 0.818 | 0.653 | 0.500 | 0.797 |
| both texture | LogReg | 88 | 0.631 | 0.737 | 0.541 | 0.834 | 0.644 | 0.889 | 0.678 |
| both texture | kNN | 88 | 0.457 | 0.518 | 0.569 | 0.738 | 0.374 | 0.278 | 0.903 |
| both texture | SVM-RBF | 88 | 0.592 | 0.700 | 0.538 | 0.803 | 0.649 | 0.778 | 0.672 |
| both texture | RandomForest | 88 | 0.548 | 0.629 | 0.649 | 0.826 | 0.626 | 0.444 | 0.817 |
| both texture | GradBoost | 88 | 0.576 | 0.620 | 0.576 | 0.818 | 0.671 | 0.444 | 0.745 |
| both shape | LogReg | 30 | 0.571 | 0.616 | 0.458 | 0.730 | 0.622 | 0.667 | 0.559 |
| both shape | kNN | 30 | 0.452 | 0.435 | 0.439 | 0.743 | 0.410 | 0.000 | 0.896 |
| both shape | SVM-RBF | 30 | 0.587 | 0.677 | 0.527 | 0.785 | 0.608 | 0.722 | 0.700 |
| both shape | RandomForest | 30 | 0.501 | 0.548 | 0.572 | 0.787 | 0.523 | 0.278 | 0.842 |
| both shape | GradBoost | 30 | 0.564 | 0.597 | 0.531 | 0.772 | 0.604 | 0.444 | 0.742 |
| metadata only | LogReg | 7 | 0.594 | 0.655 | 0.417 | 0.762 | 0.541 | 0.944 | 0.480 |
| metadata only | kNN | 7 | 0.432 | 0.428 | 0.433 | 0.748 | 0.347 | 0.000 | 0.936 |
| metadata only | SVM-RBF | 7 | 0.593 | 0.657 | 0.438 | 0.751 | 0.419 | 0.944 | 0.606 |
| metadata only | RandomForest | 7 | 0.571 | 0.642 | 0.458 | 0.764 | 0.455 | 0.833 | 0.639 |
| metadata only | GradBoost | 7 | 0.580 | 0.683 | 0.471 | 0.769 | 0.446 | 0.944 | 0.658 |
| both C+T+S | LogReg | 262 | 0.623 | 0.679 | 0.543 | 0.815 | 0.653 | 0.667 | 0.716 |
| both C+T+S | kNN | 262 | 0.494 | 0.533 | 0.579 | 0.765 | 0.428 | 0.278 | 0.894 |
| both C+T+S | SVM-RBF | 262 | 0.601 | 0.569 | 0.570 | 0.818 | 0.671 | 0.222 | 0.815 |
| both C+T+S | RandomForest | 262 | 0.548 | 0.611 | 0.649 | 0.845 | 0.626 | 0.333 | 0.874 |
| both C+T+S | GradBoost | 262 | 0.579 | 0.626 | 0.630 | 0.839 | 0.662 | 0.389 | 0.828 |
| both C+T+S+meta | LogReg | 269 | 0.655 | 0.743 | 0.572 | 0.834 | 0.685 | 0.833 | 0.712 |
| both C+T+S+meta | kNN | 269 | 0.514 | 0.575 | 0.626 | 0.783 | 0.482 | 0.333 | 0.910 |
| both C+T+S+meta | SVM-RBF | 269 | 0.611 | 0.649 | 0.576 | 0.824 | 0.689 | 0.500 | 0.756 |
| both C+T+S+meta | RandomForest | 269 | 0.557 | 0.617 | 0.654 | 0.850 | 0.644 | 0.333 | 0.874 |
| both C+T+S+meta | GradBoost | 269 | 0.614 | 0.737 | 0.667 | 0.872 | 0.685 | 0.722 | 0.804 |
| none | majority (always Malignant) | 0 |  | 0.333 | 0.273 | 0.500 | 0.000 | 0.000 | 1.000 |
| none | stratified random | 0 |  | 0.328 | 0.328 | 0.499 | 0.297 | 0.000 | 0.687 |
