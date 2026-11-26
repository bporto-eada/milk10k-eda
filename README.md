# MILK10k: skin-lesion classification

Course project for Computer Vision and Speech Recognition (EADA), Bruno Curi. It goes from the MILK10k images to a
model that says whether a skin lesion is benign, malignant or indeterminate, evaluated honestly, and to an official
result on the ISIC Challenge leaderboard.

| Hand-in | Where |
|---|---|
| Final report (7 sections) | [report.pdf](report.pdf), source [reports/final_report.md](reports/final_report.md) |
| Model card | [model_card.md](model_card.md) |
| Leaderboard file | [m5_final/submission.csv](m5_final/submission.csv), approach name **EADA-CV26-BC** |
| Official result | [m5_final/leaderboard.png](m5_final/leaderboard.png) (pending upload) |
| `predict()` for a new image | [m5_final/predict.py](m5_final/predict.py) |
| Every file of every milestone | [SUBMISSION.md](SUBMISSION.md) |

**Result.** The final model is an ImageNet ResNet-18 with the whole network fine-tuned, chosen on validation:
macro-F1 **0.597 on the test lesions** (95% CI 0.549 to 0.645, 3 classes). The same network with 11 outputs scores
0.420 on validation and 0.432 on our own test split under the leaderboard's rule; official score pending upload.

## 1. The project, milestone by milestone

**Task.** Predict `diagnosis_1` (Benign / Indeterminate / Malignant) for each lesion. The whole analysis (scoreboard,
best model, errors, fairness) is on these 3 classes. The leaderboard scores the 11 classes of `training_gt.csv`
(AKIEC, BCC, BEN_OTH, BKL, DF, INF, MAL_OTH, MEL, NV, SCCKA, VASC), so the final model is retrained with 11 outputs
for it (Milestone 5). The classical models use both images and the metadata; the networks read the dermoscopic image
only.

**Dataset.** MILK10k, published by the MILK study team on the ISIC Archive (DOI
[10.34970/648456](https://doi.org/10.34970/648456), https://api.isic-archive.com/doi/milk10k/). 10,480 JPEG images
(600x450) of 5,240 lesions, exactly one dermoscopic and one clinical close-up per lesion, with age, sex, body site and
the diagnosis hierarchy in `metadata.csv`. License: CC BY-NC 4.0 (non-commercial use, attribution "MILK study team").

| Milestone | Folder | What it does | Entry point | Written up in |
|---|---|---|---|---|
| M1 Data pipeline, Sessions 1 to 3 | [m1_data/](m1_data) | integrity checks, EDA, label strategy, lesion-level split, normalisation, augmentation, class weights, DataLoaders | `milestone1.py` | [milestone1_report.md](reports/milestone1_report.md) |
| M2 Classical baseline, Sessions 4 and 5 | [m2_baseline/](m2_baseline) | 131 hand-crafted colour, texture and shape features; 8 feature sets x 5 models, chosen on validation | `classical_baseline.py` | [classical_baseline_report.md](reports/classical_baseline_report.md) |
| M3 Deep learning, Sessions 6 and 7 | [m3_deep/](m3_deep) | MLP, CNN from scratch, ResNet-18 frozen and fine-tuned, scoreboard of Sessions 5 to 7 | `session7_cnns.ipynb` | [session7_homework_report.md](reports/session7_homework_report.md) |
| M4 Best model and analysis, Sessions 8 and 9 | [m4_best_model/](m4_best_model) | ViT-B/16 against ResNet-18, best model chosen on validation, test once, error analysis, fairness | `compare_models.py`, then `error_analysis.py`, `fairness.py`, `class_weights_check.py` | [report.pdf](report.pdf), sections 3 to 5 |
| M5 Final pipeline and leaderboard, Session 10 | [m5_final/](m5_final) | the 11-class model, `predict()`, `submission.csv` | `train_11class.py`, `make_submission.py`, `predict.py` | [report.pdf](report.pdf), section 6; [model_card.md](model_card.md) |

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

**Run everything, in order** (from the repo root; times on a 2-core CPU without a GPU):

```
python m1_data/milestone1.py                   # M1, 3 min: checks, EDA, split, normalisation, class weights
python m2_baseline/classical_baseline.py       # M2, about 50 min: features, mask audit, model comparison, test once
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 m3_deep/session7_cnns.ipynb
                                               # M3, about 70 min: MLP, CNN, ResNet-18, scoreboard of Sessions 5 to 7
python m4_best_model/compare_models.py         # M4, 1 h: best CNN trained again, ViT against ResNet, test once
python m4_best_model/error_analysis.py         #     seconds: confusion matrix, misclassified images
python m4_best_model/fairness.py               #     seconds: scores per skin tone and per sex
python m4_best_model/class_weights_check.py    #     1 min: the same head with and without class weights, 5 seeds
python m5_final/train_11class.py               # M5, 30 min: the 11-class model, scored the leaderboard's way
python m5_final/make_submission.py             #     seconds: m5_final/submission.csv for the 479 blind lesions
python reports/build_pdf.py reports/final_report.md report.pdf    # the report (needs Chrome or Chromium)
python -m pytest tests                         # 35 tests on synthetic data, no dataset needed
```

Each script finds the repo root by itself, so every command runs from the repo root. Results go to `outputs/` only
(locations in `shared/config.py`), and each script reads only what the scripts above it wrote. The first lines of
each script list its steps.

**One step or one milestone alone.** `python m1_data/milestone1.py splits` runs one step (integrity, eda, labels,
quality, splits, preprocess, augment, loaders). `python m2_baseline/classical_baseline.py compare final stretch` needs
only the committed `image_features.csv.gz` (about 20 min); only its `features` step needs the images, and without a
local copy of the dataset `m2_baseline/classical_baseline_colab.ipynb` runs that step on Google Colab. The M3 notebook
also runs on Colab
(`colab.research.google.com/github/bporto-eada/milk10k-eda/blob/main/m3_deep/session7_cnns.ipynb`, Runtime: GPU):
its first cell clones the repo and downloads MILK10k by itself. The images are decoded once into `milk10k/cache/`
(not committed); the ImageNet weights come from torchvision.

**Session exercises** (part of Milestone 1, kept as submitted):

```
python m1_data/session1/session1.py            # session 1 EDA: samples per class, age and sex per class
python m1_data/session2/run_session2.py        # session 2 homework
jupyter nbconvert --to notebook --execute --inplace m1_data/homework_part_a.ipynb    # homework Sessions 1-3, Part A
```

**Predict a new image** (needs only the repo and `pip install -r requirements.txt`, no dataset):

```
python m5_final/predict.py path/to/dermoscopic_image.jpg              # 11 diagnoses, the leaderboard model
python m5_final/predict.py path/to/dermoscopic_image.jpg --model m4   # Benign / Indeterminate / Malignant
```

`predict()` loads the saved model, puts it in evaluation mode and applies exactly the validation transform
(`shared.deep.val_tf`: shorter side to 224, central 224x224 square, ImageNet mean and std). From Python:
`from m5_final.predict import predict; predict("image.jpg")` returns `{class: probability}`.

**The leaderboard's test images.** `make_submission.py` needs the two test files of
[challenge.isic-archive.com/data](https://challenge.isic-archive.com/data) (MILK10k section) in `milk10k_test/` at
the repo root, or in the folder named by `MILK10K_TEST_DIR`: `MILK10k_Test_Metadata.csv` and the unzipped
`MILK10k_Test_Input/`. Those 958 images have no labels and are only ever predicted.

## 3. Repository structure

One folder per milestone, as in the structure the project brief suggests; the code they share is in `shared/`.

```
README.md                      this file: what the project is, how to run it, where everything is
SUBMISSION.md                  every hand-in, milestone by milestone, with a direct link to each file
requirements.txt               Python packages for the whole repo
.gitignore                     keeps the dataset, caches and virtual environments out of git
report.pdf                     the final report (source: reports/final_report.md)
model_card.md                  what the model is for, its data, its numbers, its limits
m1_data/                       Milestone 1, data pipeline (Sessions 1 to 3)
  milestone1.py                steps B1 to B8: integrity, eda, labels, quality, splits, preprocess, augment, loaders
  homework_part_a.ipynb        homework Sessions 1-3, Part A: one section per exercise, code, outputs and answers
  homework_part_a.pdf          the same notebook as PDF
  session1/                    session 1: session1.py (EDA), session1_exercises.py, figures/
  session2/                    session 2 homework: analyses and runner, figures/ and results/ as submitted
m2_baseline/                   Milestone 2, classical baseline (Sessions 4 and 5)
  classical_baseline.py        steps features, audit, compare, final, stretch
  classical_baseline_colab.ipynb   runs the features step on Google Colab (downloads MILK10k from ISIC)
m3_deep/                       Milestone 3, deep learning (Sessions 6 and 7)
  session7_cnns.ipynb          MLP, CNN from scratch, ResNet-18 frozen and fine-tuned, final scoreboard (run)
  session7_cnns.pdf            the run notebook as PDF
m4_best_model/                 Milestone 4, best model and analysis (Sessions 8 and 9)
  compare_models.py            the best CNN trained again and saved, ViT-B/16 against ResNet-18, scoreboard, best model
  error_analysis.py            confusion matrix, mistakes by diagnosis, misclassified images, what they share
  fairness.py                  scores per skin tone and per sex, and why darker skin scores lower
  class_weights_check.py       the same linear head trained with and without class weights, and with five seeds
  model_m4.pt                  the best 3-class model, chosen on validation (weights and class names)
m5_final/                      Milestone 5, final pipeline and leaderboard (Session 10)
  predict.py                   predict(image_path): class probabilities for one new image; also a command
  train_11class.py             the best architecture retrained with 11 outputs, scored the leaderboard's way
  make_submission.py           the 479 blind lesions -> submission.csv, with the format checks
  model_11class.pt             the 11-class model behind the leaderboard file
  submission.csv               the file uploaded to the ISIC Challenge
  submission_lesion_id.csv     the same values with the id column named lesion_id (see section 5)
  leaderboard.png              screenshot of the official result
shared/                        importable package with all reusable code
  config.py                    the only place for paths, seed, label columns, split sizes, image size, loader settings
  deep.py                      M4/M5: the functions of the Session 7 notebook in one place: val_tf (the inference
                               transform), image cache, prep, augment, predict_proba, fit, macro_f1, test_interval
  features.py                  hand-crafted features: lesion mask, colour, texture, shape (ABCD), metadata
  classical.py                 lesion feature table, feature sets, model grids, metrics, bootstrap, importance
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
  test_features.py             lesion mask, feature values, lesion table, feature sets, metrics and bootstrap
  test_deep.py                 val_tf deterministic, same numbers in training and prediction, flips only, fit, predict()
outputs/                       generated files only, written by the scripts and notebooks above
  classical_baseline/          everything m2_baseline/classical_baseline.py writes:
    image_features.csv.gz      131 features of each of the 10,480 images
    segmentation_audit.*       manual mask verdicts on 120 images and the failure estimate for all images
    comparison.csv / .md       8 feature sets x 5 models: CV and validation scores, chosen hyper-parameters
    best_config.json           the configuration chosen on validation
    test_results.json / .md    test metrics with 95% intervals, confusion matrix, baselines, triage point
    stretch_dx.md              the 11-class run
    figures/                   masks, comparison heatmap, confusion matrix, ROC, feature-group importance
    log.txt                    console output of every run
  session7/                    everything m3_deep/session7_cnns.ipynb writes:
    scoreboard.csv / .md       final scoreboard, Sessions 5 to 7: validation and test macro-F1, test 95% CI
    scoreboard_val.csv         the in-class validation scoreboard (sections 10 and 11)
    finetuning_strategies.csv  strategies 1 to 3 and the mini challenge: trainable weights, s/epoch, val macro-F1
    predictions_val.csv        every model's prediction for each validation lesion
    predictions_test.csv       every model's prediction for each test lesion
    figures/                   scoreboard, validation macro-F1 per epoch, best model's test confusion matrix
  m4_best_model/               everything the four scripts of m4_best_model/ write:
    scoreboard.csv / .md       every model, M2 to M4: validation and test macro-F1, test 95% CI, the final one marked
    best_model.json            the chosen model, how it was chosen, its validation and test scores
    history_resnet18.csv       training loss and validation macro-F1 after each epoch
    predictions_val.csv        the final model's class probabilities for each validation lesion
    predictions_test.csv       the same for each test lesion (the 3-class test predictions)
    confusion_matrix_test.csv  rows: truth, columns: prediction
    per_class_test.csv         precision, recall and F1 of each class
    errors_by_diagnosis.csv    mistakes per 11-class diagnosis and the answers the model gave
    errors_by_pigmentation.csv / errors_by_otsu_mask.csv   error rates inside each true class
    error_analysis.md          all of the above in one page
    misclassified_gallery.csv  the 10 misclassified test lesions shown in figures/misclassified_gallery.png
    fairness.csv / .md         one row per skin tone and per sex: group size, macro-F1, recall of Malignant and Benign
    fairness_malignant_by_diagnosis.csv   malignant lesions found, per diagnosis and skin-tone group
    class_weights_check.csv    the linear head with and without class weights, on validation
    class_weights_check_seeds.csv   the same with five seeds, on ResNet-18 and on ViT-B/16 features
    figures/                   learning curves, confusion matrix, misclassified gallery, fairness
  m5_final/                    everything train_11class.py writes:
    scores_11class.json        validation and test macro-F1: most likely class, and the leaderboard's cut at 0.5
    per_class_11class.csv      F1 at 0.5 of each of the 11 classes, on validation and on our test split
    history_11class.csv        training loss and validation macro-F1 after each epoch
    predictions_*_11class.csv  its probabilities for each validation and test lesion
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
  final_report.md              the final report (Markdown source of report.pdf)
  build_pdf.py                 Markdown -> PDF with the layout of the course reports
  milestone1_report.md         Milestone 1 report (2 pages)
  milestone1_report.pdf        the same report as PDF
  classical_baseline_report.md classical ML baseline report
  classical_baseline_report.pdf the same report as PDF
  session7_homework_report.md  Session 7 homework: final scoreboard and results (with links to the code)
  session7_homework_report.pdf the same report as PDF
milk10k/                       the dataset (not committed)
milk10k_test/                  the 479 blind test lesions of the leaderboard (not committed)
```

**Why this layout.** One folder per milestone, in the order the project was built, so the code of a milestone is in
one place. Everything that more than one milestone needs lives in one importable package, `shared/`: the scripts,
the notebooks and the tests all import from it, so no function is copy-pasted, and `shared/config.py` is the single
place where paths, the seed and sizes are set. Exploration and written answers stay in notebooks; the reproducible
pipeline is in scripts with named steps, so each step can be re-run alone. Generated files only ever go to
`outputs/`, never next to source code, so it is always clear what is code and what is a result. The tests use small
synthetic data, so they run anywhere without the dataset. The Session 1 and 2 scripts are kept as submitted; their
reusable parts (paths, loading, visualiser, figure style) were moved into `shared/`.

## 4. Data handling rules

* Not committed: the dataset (`milk10k/`, images and CSVs; it is in `.gitignore`). Anyone can download it from ISIC.
* Committed: code, the notebooks and their PDFs, and the small generated files in `outputs/` (split CSVs, JSON files,
  reports, figures), so results can be checked without re-running. The image features (`image_features.csv.gz`,
  4.8 MB, numbers only, no pixels) are committed so the baseline can be re-trained without the dataset.
* Generated files go to `outputs/` (Session 2 keeps its own `m1_data/session2/figures` and `m1_data/session2/results`).
* The two final models are committed (`model_m4.pt`, `model_11class.pt`, 45 MB each), so `predict.py` works on
  a fresh clone; the image and feature caches (`milk10k/cache/`) are not. The models were
  trained on CC BY-NC data: non-commercial use only. `submission.csv` and `leaderboard.png` sit in `m5_final/`, where
  the project brief asks for them.
* The 479 blind test images (`milk10k_test/`) are not committed and are never used for training or tuning.
* Seed: **42** (`shared/config.py`), used for the split, sampling and the DataLoaders.
* Splits created on **2026-10-05** with seed 42 (recorded in `outputs/splits/split_info.json`).

## 5. Key decisions and results

**Milestone 1: data pipeline.**

* **Label strategy.** Primary target `diagnosis_1` with 3 classes; Indeterminate is kept as its own class (123
  lesions, all actinic keratoses). For the 11-class stretch goal all classes are kept and the rare ones are handled
  with class weights. Saved in `outputs/label_map.json`.
* **Split design.** Lesion-level split with StratifiedGroupKFold (groups = lesion_id), stratified on the 11-class label
  refined by diagnosis_1: 70 / 15 / 15 = 3,667 / 786 / 787 lesions. No lesion in two splits, every lesion's two
  images in the same split, class shares within 0.21 pp of the overall shares.
* **Preprocessing.** All images are 600x450, so the input is 224x224 (Resize the shorter side to 224, then
  CenterCrop). Normalization from the train images only: mean (0.686, 0.528, 0.478), std (0.124, 0.132, 0.152).
  Augmentation: random resized crop, flips, quarter turns, brightness and contrast 0.1, no hue jitter. The networks
  of M3 to M5 use random flips only (the Session 7 recipe).
* **Imbalance.** Train ratio 29.2 : 1 (Malignant vs Indeterminate lesions). Class-weighted loss (weights in
  `outputs/class_weights.json`) and a WeightedRandomSampler are both implemented; one of them is used per run.
* **Views.** In the M1 DataLoaders each image is a training sample with its lesion's label; at evaluation the two
  predictions are averaged per lesion, and metrics are computed per lesion. The networks of M3 to M5 read the
  dermoscopic image only, one image per lesion.

Details and justification: [reports/milestone1_report.md](reports/milestone1_report.md).

**Milestone 2: classical ML baseline.** 131 hand-crafted features per image (colour, texture, ABCD shape) for both views plus age,
sex and site; 8 feature sets x 5 models tuned by CV on train and chosen on val. Chosen: both views + metadata,
logistic regression (C = 0.01). Test (787 lesions, scored once): balanced accuracy 0.721 (95% CI 0.648 to 0.782),
macro ROC-AUC 0.828, Malignant sensitivity 0.70 at specificity 0.79; always answering "Malignant" scores 0.333.
Dermoscopic texture and metadata matter most; the lesion mask fails on about half of the images (mostly pink,
non-pigmented lesions) and altered photos push predictions away from Malignant. 11 classes: balanced accuracy 0.420.
Details: [reports/classical_baseline_report.md](reports/classical_baseline_report.md).

**Milestone 3: CNNs and the scoreboard of Sessions 5 to 7.** On the dermoscopic image alone, an ImageNet ResNet-18 fine-tuned on
MILK10k is the best model family. Best model, chosen on validation: the whole network fine-tuned, macro-F1 0.682 on
validation and **0.597 on test** (95% CI 0.549 to 0.645); fine-tuning only `layer4` ties on validation (0.681) at half
the time per epoch and scores 0.634 on test. The CNN from scratch (0.531) and the frozen backbone with a linear head
(0.567) stay below the handcrafted-feature random forest (0.621); the best classical model needs both images and the
metadata to come close (0.667 validation, 0.599 test). Details: [reports/session7_homework_report.md](reports/session7_homework_report.md),
scoreboard in [outputs/session7/scoreboard.md](outputs/session7/scoreboard.md).

**Milestone 4: ViT-B/16 against ResNet-18, best model and analysis.** With both backbones frozen and the same new
last layer (Session 9's fair comparison), ViT-B/16 scores 0.574 on validation against 0.567 for ResNet-18 (test 0.592
against 0.537). The fine-tuned ResNet-18 of Session 7 keeps the highest validation macro-F1 (0.682) and is the final
model: **test macro-F1 0.597** (95% CI 0.549 to 0.645). The ViT was only tried frozen (86.6 million weights, 2-core
CPU), and on test it is level with the final model, so this does not show that a CNN beats a ViT. The final model's
errors share one pattern: brown, sharply bounded lesions are read as benign and pink ones as malignant, so 33 of 68
test melanomas are not called Malignant. Fairness: no difference found between sexes; on skin tones 0 to 2 it finds 22
of 41 malignant lesions (0.537) against 0.883 on tones 3 to 5. A check on the class weights shows that they overshoot:
the same linear head scores 0.567 with them and 0.703 without on validation, and over five seeds every run without
them beats every run with them. Scoreboard:
[outputs/m4_best_model/scoreboard.md](outputs/m4_best_model/scoreboard.md); error analysis:
[error_analysis.md](outputs/m4_best_model/error_analysis.md); fairness: [fairness.md](outputs/m4_best_model/fairness.md).

**Milestone 5: final pipeline and leaderboard.** The same architecture and recipe with 11 outputs
(`m5_final/train_11class.py`) scores a validation macro-F1 of 0.413 with the most likely class. The leaderboard cuts
each of the 11 probabilities at 0.5: scored that way, the model reaches 0.420 on validation and **0.432 on our own
test split** (95% interval 0.365 to 0.484). For about one lesion in four no class reaches 0.5; MAL_OTH (5 training
lesions) is never predicted. `submission.csv` is the plain softmax output for the 479 blind lesions. Official score,
approach name EADA-CV26-BC: pending upload ([m5_final/leaderboard.png](m5_final/leaderboard.png)). `submission.csv`
names its id column `lesion`, as the brief and the challenge page ask; the scoring code ISIC publishes reads
`lesion_id`, so `submission_lesion_id.csv` holds the same values under that name in case the first file is refused.
Report: [report.pdf](report.pdf); model card: [model_card.md](model_card.md).
