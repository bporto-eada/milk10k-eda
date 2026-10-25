# Test results of the chosen classical baseline

Configuration: **both C+T+S+meta + LogReg** {'C': 0.01}, chosen on VAL, refitted on train+val (4,453 lesions) and scored once on the 787 test lesions. 95% intervals: stratified bootstrap, 2,000 resamples.

| metric | test | 95% CI |
|---|---|---|
| accuracy | 0.698 |  |
| balanced_accuracy | 0.721 | 0.648 to 0.782 |
| macro_f1 | 0.551 | 0.519 to 0.581 |
| macro_auc | 0.828 | 0.783 to 0.863 |
| recall_Benign | 0.686 |  |
| precision_Benign | 0.657 |  |
| recall_Indeterminate | 0.778 |  |
| precision_Indeterminate | 0.115 |  |
| recall_Malignant | 0.700 |  |
| precision_Malignant | 0.884 |  |
| sensitivity_malignant | 0.700 | 0.659 to 0.736 |
| specificity_malignant | 0.793 | 0.739 to 0.838 |

## Confusion matrix (lesions)

| index | pred Benign | pred Indeterminate | pred Malignant |
|---|---|---|---|
| true Benign | 153 | 24 | 46 |
| true Indeterminate | 0 | 14 | 4 |
| true Malignant | 80 | 84 | 382 |

## Against trivial baselines (same test lesions)

| model | accuracy | balanced_accuracy | macro_f1 | macro_auc | recall_Benign | recall_Indeterminate | recall_Malignant |
|---|---|---|---|---|---|---|---|
| majority (always Malignant) | 0.694 | 0.333 | 0.273 | 0.500 | 0.000 | 0.000 | 1.000 |
| stratified random | 0.558 | 0.343 | 0.344 | 0.505 | 0.291 | 0.056 | 0.683 |
| LogReg on both C+T+S+meta | 0.698 | 0.721 | 0.551 | 0.828 | 0.686 | 0.778 | 0.700 |

## Triage operating point

Malignant threshold for 90% sensitivity, chosen on out-of-fold predictions of train+val: on test, sensitivity 0.883, specificity 0.560; 43.0% of benign and 55.6% of indeterminate lesions are flagged.


## Feature-group importance (model fitted on train, groups shuffled on val)

| group | n_features | drop_mean | drop_std |
|---|---|---|---|
| derm texture | 44 | 0.168 | 0.028 |
| metadata | 7 | 0.124 | 0.034 |
| derm colour | 72 | 0.082 | 0.029 |
| clinical texture | 44 | 0.070 | 0.021 |
| derm shape | 15 | 0.037 | 0.028 |
| clinical colour | 72 | 0.036 | 0.024 |
| clinical shape | 15 | 0.006 | 0.015 |

## Subgroups: lesions with an 'altered' image

| subgroup | lesions | balanced_accuracy | accuracy | recall_Benign | recall_Indeterminate | recall_Malignant |
|---|---|---|---|---|---|---|
| instrument only | 738 | 0.677 | 0.705 | 0.676 | 0.636 | 0.717 |
| has an altered image | 49 | 0.720 | 0.592 | 0.812 | 1.000 | 0.346 |
