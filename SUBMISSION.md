# Submissions

**Name:** Bruno Curi
**Repository:** https://github.com/bporto-eada/milk10k-eda

## Final project (Moodle hand-in, Friday 27 November)

The three items of the hand-in:

1. **GitHub repository:** https://github.com/bporto-eada/milk10k-eda, how to run everything in [README.md](https://github.com/bporto-eada/milk10k-eda/blob/main/README.md)
2. **Final report:** [report.pdf](https://github.com/bporto-eada/milk10k-eda/blob/main/report.pdf) (source [reports/final_report.md](https://github.com/bporto-eada/milk10k-eda/blob/main/reports/final_report.md))
3. **Leaderboard proof:** [m5_final/submission.csv](https://github.com/bporto-eada/milk10k-eda/blob/main/m5_final/submission.csv), screenshot of the official result [m5_final/leaderboard.png](https://github.com/bporto-eada/milk10k-eda/blob/main/m5_final/leaderboard.png), approach name **EADA-CV26-BC**

Model card: [model_card.md](https://github.com/bporto-eada/milk10k-eda/blob/main/model_card.md).

The pipeline, one folder per milestone:

* Milestone 1, data pipeline: [m1_data/](https://github.com/bporto-eada/milk10k-eda/tree/main/m1_data), entry point [milestone1.py](https://github.com/bporto-eada/milk10k-eda/blob/main/m1_data/milestone1.py), report [milestone1_report.md](https://github.com/bporto-eada/milk10k-eda/blob/main/reports/milestone1_report.md)
* Milestone 2, classical baseline: [m2_baseline/](https://github.com/bporto-eada/milk10k-eda/tree/main/m2_baseline), entry point [classical_baseline.py](https://github.com/bporto-eada/milk10k-eda/blob/main/m2_baseline/classical_baseline.py), report [classical_baseline_report.md](https://github.com/bporto-eada/milk10k-eda/blob/main/reports/classical_baseline_report.md)
* Milestone 3, deep learning: [m3_deep/](https://github.com/bporto-eada/milk10k-eda/tree/main/m3_deep), notebook [session7_cnns.ipynb](https://github.com/bporto-eada/milk10k-eda/blob/main/m3_deep/session7_cnns.ipynb), report [session7_homework_report.md](https://github.com/bporto-eada/milk10k-eda/blob/main/reports/session7_homework_report.md)
* Milestone 4, best model and analysis: [compare_models.py](https://github.com/bporto-eada/milk10k-eda/blob/main/m4_best_model/compare_models.py) (ViT-B/16 against ResNet-18, scoreboard, best model), [error_analysis.py](https://github.com/bporto-eada/milk10k-eda/blob/main/m4_best_model/error_analysis.py), [fairness.py](https://github.com/bporto-eada/milk10k-eda/blob/main/m4_best_model/fairness.py), [class_weights_check.py](https://github.com/bporto-eada/milk10k-eda/blob/main/m4_best_model/class_weights_check.py), weights [model_m4.pt](https://github.com/bporto-eada/milk10k-eda/blob/main/m4_best_model/model_m4.pt)
    * Scoreboard M2 to M4: [scoreboard.md](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/m4_best_model/scoreboard.md) · best model: [best_model.json](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/m4_best_model/best_model.json)
    * Error analysis: [error_analysis.md](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/m4_best_model/error_analysis.md) · fairness: [fairness.md](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/m4_best_model/fairness.md) · figures: [outputs/m4_best_model/figures](https://github.com/bporto-eada/milk10k-eda/tree/main/outputs/m4_best_model/figures)
    * 3-class predictions on the test split: [predictions_test.csv](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/m4_best_model/predictions_test.csv)
* Milestone 5, final pipeline: [m5_final/predict.py](https://github.com/bporto-eada/milk10k-eda/blob/main/m5_final/predict.py), [m5_final/train_11class.py](https://github.com/bporto-eada/milk10k-eda/blob/main/m5_final/train_11class.py), [m5_final/make_submission.py](https://github.com/bporto-eada/milk10k-eda/blob/main/m5_final/make_submission.py), weights [model_11class.pt](https://github.com/bporto-eada/milk10k-eda/blob/main/m5_final/model_11class.pt)
    * 11-class results: [scores_11class.json](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/m5_final/scores_11class.json), [per_class_11class.csv](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/m5_final/per_class_11class.csv)
* Code shared by both: [shared/deep.py](https://github.com/bporto-eada/milk10k-eda/blob/main/shared/deep.py) (the validation transform, the training loop, the metric)
* Tests: [tests/test_deep.py](https://github.com/bporto-eada/milk10k-eda/blob/main/tests/test_deep.py)

## Earlier hand-ins

On 26 November the files of Milestones 1 to 3 moved from the repo root into `m1_data/`, `m2_baseline/` and `m3_deep/`, the structure the project brief suggests. The links below point to their new places; links in earlier PDFs that point to the repo root no longer open.

## Session 7: CNNs and the final scoreboard (homework before Session 8)

* Report with the scoreboard: [reports/session7_homework_report.md](https://github.com/bporto-eada/milk10k-eda/blob/main/reports/session7_homework_report.md), [PDF](https://github.com/bporto-eada/milk10k-eda/blob/main/reports/session7_homework_report.pdf)
* Notebook, run on the real data with all outputs (class lab, mini challenge, homework in section 12): [session7_cnns.ipynb](https://github.com/bporto-eada/milk10k-eda/blob/main/m3_deep/session7_cnns.ipynb), [PDF](https://github.com/bporto-eada/milk10k-eda/blob/main/m3_deep/session7_cnns.pdf)
* Final scoreboard, Sessions 5 to 7, validation and test macro-F1: [outputs/session7/scoreboard.md](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/session7/scoreboard.md), [scoreboard.csv](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/session7/scoreboard.csv)
* Fine-tuning strategies 1 to 3 and the mini challenge: [outputs/session7/finetuning_strategies.csv](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/session7/finetuning_strategies.csv)
* Every model's predictions per lesion: [predictions_val.csv](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/session7/predictions_val.csv), [predictions_test.csv](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/session7/predictions_test.csv)
* Figures: [outputs/session7/figures](https://github.com/bporto-eada/milk10k-eda/tree/main/outputs/session7/figures)
* Code the notebook reuses: split and class weights [outputs/splits](https://github.com/bporto-eada/milk10k-eda/tree/main/outputs/splits), [shared/datasets.py](https://github.com/bporto-eada/milk10k-eda/blob/main/shared/datasets.py), [shared/labels.py](https://github.com/bporto-eada/milk10k-eda/blob/main/shared/labels.py); Session 5 models [shared/classical.py](https://github.com/bporto-eada/milk10k-eda/blob/main/shared/classical.py) with their tuned settings in [comparison.csv](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/classical_baseline/comparison.csv)

## Classical ML baseline

* Report: [reports/classical_baseline_report.md](https://github.com/bporto-eada/milk10k-eda/blob/main/reports/classical_baseline_report.md), [PDF](https://github.com/bporto-eada/milk10k-eda/blob/main/reports/classical_baseline_report.pdf)
* Entry point, steps features, audit, compare, final, stretch: [classical_baseline.py](https://github.com/bporto-eada/milk10k-eda/blob/main/m2_baseline/classical_baseline.py)
* Feature extraction (lesion mask, colour, texture, shape, metadata): [shared/features.py](https://github.com/bporto-eada/milk10k-eda/blob/main/shared/features.py)
* Feature sets, models, metrics, bootstrap, importance: [shared/classical.py](https://github.com/bporto-eada/milk10k-eda/blob/main/shared/classical.py)
* Settings (paths, CV folds, bootstrap size): [shared/config.py](https://github.com/bporto-eada/milk10k-eda/blob/main/shared/config.py)
* Colab notebook for the features step: [classical_baseline_colab.ipynb](https://github.com/bporto-eada/milk10k-eda/blob/main/m2_baseline/classical_baseline_colab.ipynb)
* Tests: [tests/test_features.py](https://github.com/bporto-eada/milk10k-eda/blob/main/tests/test_features.py)
* Results: [comparison.md](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/classical_baseline/comparison.md), [best_config.json](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/classical_baseline/best_config.json), [test_results.md](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/classical_baseline/test_results.md), [stretch_dx.md](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/classical_baseline/stretch_dx.md), [segmentation_audit.md](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/classical_baseline/segmentation_audit.md), [log.txt](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/classical_baseline/log.txt)
* Features of all images: [image_features.csv.gz](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/classical_baseline/image_features.csv.gz)
* Figures: [outputs/classical_baseline/figures](https://github.com/bporto-eada/milk10k-eda/tree/main/outputs/classical_baseline/figures)

# Homework Sessions 1-3 and Milestone 1

## Part A

* Notebook: [homework_part_a.ipynb](https://github.com/bporto-eada/milk10k-eda/blob/main/m1_data/homework_part_a.ipynb)
* PDF: [homework_part_a.pdf](https://github.com/bporto-eada/milk10k-eda/blob/main/m1_data/homework_part_a.pdf)

## Part B: Milestone 1

* README (B0): [README.md](https://github.com/bporto-eada/milk10k-eda/blob/main/README.md)
* Requirements: [requirements.txt](https://github.com/bporto-eada/milk10k-eda/blob/main/requirements.txt)
* Report (B9): [reports/milestone1_report.md](https://github.com/bporto-eada/milk10k-eda/blob/main/reports/milestone1_report.md), [PDF](https://github.com/bporto-eada/milk10k-eda/blob/main/reports/milestone1_report.pdf)

Source code

* Pipeline entry point, steps B1 to B8: [milestone1.py](https://github.com/bporto-eada/milk10k-eda/blob/main/m1_data/milestone1.py)
* Configuration (paths, seed, sizes): [shared/config.py](https://github.com/bporto-eada/milk10k-eda/blob/main/shared/config.py)
* Data loading and lesion table: [shared/data.py](https://github.com/bporto-eada/milk10k-eda/blob/main/shared/data.py), [shared/paths.py](https://github.com/bporto-eada/milk10k-eda/blob/main/shared/paths.py)
* Quality and integrity checks (B1, B4): [shared/checks.py](https://github.com/bporto-eada/milk10k-eda/blob/main/shared/checks.py)
* Label map and class weights (B3, B8): [shared/labels.py](https://github.com/bporto-eada/milk10k-eda/blob/main/shared/labels.py)
* Splitting (B5): [shared/splits.py](https://github.com/bporto-eada/milk10k-eda/blob/main/shared/splits.py)
* Transforms (B6, B7): [shared/transforms.py](https://github.com/bporto-eada/milk10k-eda/blob/main/shared/transforms.py)
* Dataset and DataLoaders (B8): [shared/datasets.py](https://github.com/bporto-eada/milk10k-eda/blob/main/shared/datasets.py)
* Visualiser (Session 2, reused in B2 and B8): [shared/viewer.py](https://github.com/bporto-eada/milk10k-eda/blob/main/shared/viewer.py), [shared/figstyle.py](https://github.com/bporto-eada/milk10k-eda/blob/main/shared/figstyle.py)
* Tests: [tests/test_splits.py](https://github.com/bporto-eada/milk10k-eda/blob/main/tests/test_splits.py), [tests/test_transforms.py](https://github.com/bporto-eada/milk10k-eda/blob/main/tests/test_transforms.py), [tests/test_datasets.py](https://github.com/bporto-eada/milk10k-eda/blob/main/tests/test_datasets.py)

Split files (B5)

* [outputs/splits/train.csv](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/splits/train.csv)
* [outputs/splits/val.csv](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/splits/val.csv)
* [outputs/splits/test.csv](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/splits/test.csv)
* Verification: [outputs/splits/split_report.md](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/splits/split_report.md), [outputs/splits/split_info.json](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/splits/split_info.json)

Label map, normalization statistics, class weights

* [outputs/label_map.json](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/label_map.json)
* [outputs/normalization.json](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/normalization.json)
* [outputs/class_weights.json](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/class_weights.json)

Reports and tables

* Integrity (B1): [outputs/image_size_summary.csv](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/image_size_summary.csv), [outputs/image_sizes.csv](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/image_sizes.csv)
* Session 2 findings re-checked (B2): [outputs/session2_findings.md](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/session2_findings.md)
* Class mapping (B2): [outputs/dx_to_diagnosis_1.csv](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/dx_to_diagnosis_1.csv)
* Data-quality report (B4): [outputs/data_quality_report.md](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/data_quality_report.md)
* Augmentation table (B7): [outputs/augmentation_table.md](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/augmentation_table.md)
* Lesion table (A1.3): [outputs/lesions.csv](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/lesions.csv)
* Pipeline log: [outputs/milestone1_log.txt](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/milestone1_log.txt)

Figures

* Class distribution, 3 classes: [b2_classes_diagnosis_1.png](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/figures/b2_classes_diagnosis_1.png)
* Class distribution, 11 classes (log scale): [b2_classes_dx_log.png](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/figures/b2_classes_dx_log.png)
* Mapping 11 classes to diagnosis_1: [b2_dx_to_diagnosis_1.png](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/figures/b2_dx_to_diagnosis_1.png)
* Gallery, one example per class: [b2_gallery_by_class.png](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/figures/b2_gallery_by_class.png)
* Augmentations (B7): [b7_augmentations.png](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/figures/b7_augmentations.png)
* Train batch after transforms (B8): [b8_train_batch.png](https://github.com/bporto-eada/milk10k-eda/blob/main/outputs/figures/b8_train_batch.png)
