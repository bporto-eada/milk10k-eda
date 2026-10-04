# Submission: Homework Sessions 1-3 and Milestone 1

**Name:** Bruno Curi
**Repository:** https://github.com/bporto-eada/milk10k-eda

## Part A

* Notebook: [homework_part_a.ipynb](https://github.com/bporto-eada/milk10k-eda/blob/main/homework_part_a.ipynb)
* PDF: [homework_part_a.pdf](https://github.com/bporto-eada/milk10k-eda/blob/main/homework_part_a.pdf)

## Part B: Milestone 1

* README (B0): [README.md](https://github.com/bporto-eada/milk10k-eda/blob/main/README.md)
* Requirements: [requirements.txt](https://github.com/bporto-eada/milk10k-eda/blob/main/requirements.txt)
* Report (B9): [reports/milestone1_report.md](https://github.com/bporto-eada/milk10k-eda/blob/main/reports/milestone1_report.md), [PDF](https://github.com/bporto-eada/milk10k-eda/blob/main/reports/milestone1_report.pdf)

Source code

* Pipeline entry point, steps B1 to B8: [milestone1.py](https://github.com/bporto-eada/milk10k-eda/blob/main/milestone1.py)
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
