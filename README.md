# milk10k-eda

Course work for Computer Vision and Speech Recognition (EADA). The project builds towards a model that predicts the
diagnosis of a skin lesion from its dermoscopic and clinical close-up images.

## 1. What the project is

**Task.** Predict `diagnosis_1` (Benign / Indeterminate / Malignant) for each lesion from its two images, a
dermoscopic image and a clinical close-up. Stretch goal: the 11-class label `dx` (AKIEC, BCC, BEN_OTH, BKL, DF, INF,
MAL_OTH, MEL, NV, SCCKA, VASC) from `training_gt.csv`.

**Dataset.** MILK10k, published by the MILK study team on the ISIC Archive (DOI
[10.34970/648456](https://doi.org/10.34970/648456), https://api.isic-archive.com/doi/milk10k/). 10,480 JPEG images
(600x450) of 5,240 lesions, exactly one dermoscopic and one clinical close-up per lesion, with age, sex, body site and
the diagnosis hierarchy in `metadata.csv`. License: CC BY-NC 4.0 (non-commercial use, attribution "MILK study team").

**Goal of Milestone 1.** A leak-free, reproducible data pipeline: integrity checks on the full image set, a label
strategy, lesion-level train / val / test splits, train-only normalization statistics, augmentation, class-imbalance
handling and PyTorch DataLoaders, all evolved from the Session 2 code.

## 2. Setup and run

Python 3.12 (3.10 or newer works).

```
python -m venv venv
source venv/bin/activate            (Windows: venv\Scripts\activate)
pip install -r requirements.txt
```

**Data.** Download `milk10k.zip` from the ISIC link above and unzip it so that these exist:

```
milk10k/metadata.csv
milk10k/images/ISIC_xxxxxxx.jpg      (10,480 files)
milk10k/supplements/training_gt.csv
```

`milk10k/` at the repo root is the default. To keep the data elsewhere, set the environment variable `MILK10K_DIR`
(e.g. `export MILK10K_DIR=/data/milk10k`, Windows: `set MILK10K_DIR=D:\data\milk10k`). The location is read in one
place only, `shared/config.py`. If only the images live elsewhere (as in the class notebook), set
`MILK10K_IMAGES_DIR` to the images folder instead.

**Reproduce everything** (from the repo root):

```
python milestone1.py                       # Part B, steps B1 to B8, about 3 minutes; writes outputs/
python milestone1.py splits                # only one step (integrity, eda, labels, quality, splits,
                                           #   preprocess, augment, loaders)
python -m pytest tests                     # 21 tests on synthetic data, no dataset needed
jupyter nbconvert --to notebook --execute --inplace homework_part_a.ipynb    # re-run Part A
jupyter nbconvert --to html homework_part_a.ipynb                           # then print to PDF in a browser
python session1.py                         # session 1 EDA
python session2/run_session2.py            # session 2 homework
```

`preprocess`, `augment` and `loaders` read the files written by the earlier steps (image sizes, splits,
normalization), so run the steps in the order above the first time.

## 3. Repository structure

```
README.md                      this file
SUBMISSION.md                  name, repo URL and a direct link to every deliverable
requirements.txt               Python packages for the whole repo
.gitignore                     keeps the dataset, caches and virtual environments out of git
homework_part_a.ipynb          Part A: one section per exercise (A1.1 to A3.6), code, outputs and answers
homework_part_a.pdf            Part A exported to PDF
milestone1.py                  Part B entry point: runs steps B1 to B8 and writes outputs/
shared/                        importable package with all reusable code
  config.py                    the only place for paths, seed, label columns, split sizes, image size, loader settings
  paths.py                     finds the image file of an isic_id
  data.py                      loads metadata.csv + training_gt.csv, builds the lesion table, fails on missing images
  checks.py                    label-consistency, integrity (Image.verify) and association checks
  labels.py                    label map (label strategy) and class weights for the loss and the sampler
  splits.py                    split_lesions (lesion-level StratifiedGroupKFold) and split verification
  transforms.py                train_transform, eval_transform, normalization statistics, augmentation table
  datasets.py                  ImageDataset, LesionDataset, aggregate_predictions, make_loaders
  viewer.py                    Session 2 visualiser: image grid, class-balance chart, batch diagnostics
  figstyle.py                  Session 2 colours and figure saving
tests/
  test_splits.py               split properties: no overlap, sizes within 1 pp, reproducible, images follow lesions
  test_transforms.py           eval_transform deterministic, output shape and dtype, augmentation is random
  test_datasets.py             LesionDataset and ImageDataset batches, missing-file errors, aggregate_predictions
outputs/                       generated files only (written by milestone1.py and the notebook)
  splits/train.csv             train images (one row per image, lesions never split)
  splits/val.csv               validation images
  splits/test.csv              test images
  splits/split_report.md       sizes, overlap checks and class proportions per split
  splits/split_info.json       seed, creation date and method of the split
  label_map.json               class name -> integer label, for diagnosis_1 and dx
  class_weights.json           class weights from the train split (loss weights and sampler)
  normalization.json           per-channel mean and std of the train images
  image_size_summary.csv       B1: unreadable files and min / median / max width, height, file size
  image_sizes.csv              B1: size, file size and verify result of every image
  lesions.csv                  one row per lesion (A1.3)
  dx_to_diagnosis_1.csv        lesions per 11-class label and diagnosis_1
  data_quality_report.md       B4: missing values, consistency checks, shortcut check, excluded columns
  session2_findings.md         B2: Session 2 findings re-checked on the full dataset
  augmentation_table.md        B7: each augmentation, its parameters and medical justification
  milestone1_log.txt           console output of the pipeline run
  figures/                     class distributions, class gallery, augmentations, train batch
reports/
  milestone1_report.md         Milestone 1 report (2 pages)
  milestone1_report.pdf        the same report as PDF
session1.py                    session 1 EDA: samples per class, age and sex per class
session1_exercises.py          session 1 practice questions
figures/                       session 1 figures
session2/                      session 2 homework: analyses and runner, figures/ and results/ as submitted
milk10k/                       the dataset (not committed)
```

**Why this layout.** Everything that more than one script needs lives in one importable package, `shared/`: the
notebook, `milestone1.py`, the tests and the session scripts all import from it, so no function is copy-pasted, and
`shared/config.py` is the single place where paths, the seed and sizes are set. Exploration and written answers stay
in the notebook; the reproducible pipeline is a script with named steps, so each step can be re-run alone. Generated
files only ever go to `outputs/`, never next to source code, so it is always clear what is code and what is a
result. The tests use small synthetic data, so they run anywhere without the dataset. Session 1 and 2 files stay as
they were submitted; their reusable parts (paths, loading, visualiser, figure style) were moved into `shared/`.

## 4. Data handling rules

* Not committed: the dataset (`milk10k/`, images and CSVs; it is in `.gitignore`). Anyone can download it from ISIC.
* Committed: code, the notebook and its PDF, and the small generated files in `outputs/` (split CSVs, JSON files,
  reports, figures), so results can be checked without re-running.
* Generated files go to `outputs/` (Session 2 keeps its own `session2/figures` and `session2/results`).
* Seed: **42** (`shared/config.py`), used for the split, sampling and the DataLoaders.
* Splits created on **2026-10-05** with seed 42 (recorded in `outputs/splits/split_info.json`).

## 5. Key decisions and results so far

* **Label strategy.** Primary target `diagnosis_1` with 3 classes; Indeterminate is kept as its own class (123
  lesions, all actinic keratoses). For the 11-class stretch goal all classes are kept and the rare ones are handled
  with class weights. Saved in `outputs/label_map.json`.
* **Split design.** Lesion-level split with StratifiedGroupKFold (groups = lesion_id), stratified on the 11-class label
  refined by diagnosis_1: 70 / 15 / 15 = 3,667 / 786 / 787 lesions. No lesion in two splits, every lesion's two
  images in the same split, class shares within 0.21 pp of the overall shares.
* **Preprocessing.** All images are 600x450, so the input is 224x224 (Resize the shorter side to 224, then
  CenterCrop). Normalization from the train images only: mean (0.686, 0.528, 0.478), std (0.124, 0.132, 0.152).
  Augmentation: random resized crop, flips, quarter turns, brightness and contrast 0.1, no hue jitter.
* **Imbalance.** Train ratio 29.2 : 1 (Malignant vs Indeterminate lesions). Class-weighted loss (weights in
  `outputs/class_weights.json`) and a WeightedRandomSampler are both implemented; one of them is used per run.
* **Views.** Each image is a training sample with its lesion's label; at evaluation the two predictions are averaged
  per lesion, and metrics are computed per lesion.

Details and justification: [reports/milestone1_report.md](reports/milestone1_report.md).
