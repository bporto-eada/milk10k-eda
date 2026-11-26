# Model card: MILK10k skin-lesion classifier

Course project of Computer Vision and Speech Recognition (EADA), Bruno Curi, November 2026. Every number below comes
from a file in `outputs/` and can be reproduced with the commands at the end. Details and figures:
[report.pdf](report.pdf).

## What the model is

| | 3-class model | 11-class model |
|---|---|---|
| file | `m4_best_model/model_m4.pt` (45 MB) | `m5_final/model_11class.pt` (45 MB) |
| output | probabilities of Benign, Indeterminate, Malignant (`diagnosis_1`) | probabilities of AKIEC, BCC, BEN_OTH, BKL, DF, INF, MAL_OTH, MEL, NV, SCCKA, VASC |
| used for | the analysis of this project (scoreboard, errors, fairness) | the official MILK10k leaderboard |

Both are a **ResNet-18** (He et al., 2016) from torchvision, starting from its ImageNet-1k weights (IMAGENET1K_V1),
with a new last layer, fine-tuned as a whole network (11.2 million weights) on 3,667 MILK10k lesions: Adam, learning
rate 1e-4, batches of 64, random left-right and up-down flips, cross-entropy with class weights from the train split,
6 epochs, the epoch with the best validation macro-F1 kept. The 11-class model is the same recipe with other labels
and 11 outputs; its rarest class (MAL_OTH, 5 training lesions) has its loss weight capped at that of the next rarest.

**Input.** One dermoscopic RGB image of one lesion. It is resized so that its shorter side is 224 pixels, cropped to
the central 224x224 square and normalised with the ImageNet mean and standard deviation
(`shared.deep.val_tf`). Training, validation, test and `predict()` all use this one transform. The clinical
close-up, age, sex and body site are not used.

**How to use it.** `python m5_final/predict.py image.jpg` or, from Python,
`from m5_final.predict import predict; predict("image.jpg")`, which returns `{class: probability}`.

## Intended use

* Teaching and research on the MILK10k dataset: a worked example of a pipeline that is split, evaluated and reported
  honestly, and a baseline for the MILK10k leaderboard.
* Intended users: students and instructors of the course, and anyone studying the dataset.

## What it must not be used for

* **Diagnosis, triage, screening, or any decision about a patient.** It is not a medical device and has had no
  clinical validation.
* **Self-assessment** of a mole or spot by anyone.
* **Clinical photographs, phone pictures, or dermoscopic images from other devices or clinics.** It has only seen
  dermoscopic images of one study.
* **Commercial use.** It was trained on data under CC BY-NC.

The numbers below are the reason: a model that misses half of the melanomas it is shown cannot reassure anyone.

## Data

MILK10k (MILK study team, ISIC Archive): 5,240 lesions, each with a dermoscopic image and a clinical close-up, 600x450
JPEGs. 95.7% of the lesions were confirmed by histopathology, so the dataset holds lesions that someone decided to
biopsy: 69.4% are malignant and 48.1% are BCC. That is not the mix of a clinic or of the general population.

| split | lesions | Benign / Indeterminate / Malignant | role |
|---|---|---|---|
| train | 3,667 | 1,038 / 87 / 2,542 | fitting |
| validation | 786 | 222 / 18 / 546 | every choice (hyper-parameters, epoch, model) |
| test | 787 | 223 / 18 / 546 | one score per model |

The split is by lesion (StratifiedGroupKFold, seed 42): the two images of a lesion are never on two sides. The 479
lesions of the official leaderboard are a fourth set, without labels, used only for prediction.

## Results

Metric: **macro-F1 per lesion**, the plain mean of the F1 of each class, so a rare class counts as much as a common one.

**3-class model** (`outputs/m4_best_model/`). Always answering "Malignant" scores 0.273.

| | validation | test | test 95% CI |
|---|---|---|---|
| macro-F1 | 0.682 | **0.597** | 0.549 to 0.645 |

| class (test) | lesions | precision | recall | F1 |
|---|---|---|---|---|
| Benign | 223 | 0.752 | 0.695 | 0.723 |
| Indeterminate | 18 | 0.150 | 0.333 | 0.207 |
| Malignant | 546 | 0.865 | 0.857 | 0.861 |

It was chosen because it has the highest validation macro-F1 of the deep models. The same ResNet-18 with only `layer4`
fine-tuned is tied with it on validation (0.681) and scores 0.634 on test. A frozen ViT-B/16 with a new last layer
scores 0.574 / 0.592; frozen, with the same last layer, the ResNet-18 scores 0.567 / 0.537. The test intervals of all
of these overlap, and the random seed alone moves the validation score of such a head by up to 0.07.

**11-class model** (`outputs/m5_final/`). The leaderboard turns each of the 11 probabilities of a lesion into yes or
no at 0.5 and averages the F1 of the 11 classes. The scores below are computed the same way.

| | lesions | macro-F1, most likely class | macro-F1, cut at 0.5 |
|---|---|---|---|
| validation | 786 | 0.413 | 0.420 |
| our own test split, opened once | 787 | 0.419 | **0.432** (95% interval 0.365 to 0.484) |
| official MILK10k leaderboard (approach EADA-CV26-BC) | 479 | | pending upload |

F1 per class at 0.5 on our test split: VASC 0.750, NV 0.714, BCC 0.652, SCCKA 0.588, MEL 0.525, AKIEC 0.427, DF 0.400, BKL 0.328, INF 0.250, BEN_OTH 0.118, MAL_OTH 0.000. MAL_OTH is never predicted (5 training lesions, one test lesion). The
five classes with 1 to 7 test lesions make the macro-F1 uncertain. For about one lesion in four (27.3% on our test
split) no class reaches 0.5, and the leaderboard reads that as no diagnosis. `submission.csv` holds the model's plain
softmax probabilities; no thresholds were tuned.

## Results by group (3-class model, test lesions)

Skin tone is MILK10k's `skin_tone_class`, from 0 (very dark) to 5 (very light). The macro-F1 per group is in
`outputs/m4_best_model/fairness.md`; it is not comparable between groups, because 14 of the 18 Indeterminate test
lesions have skin tone 3 and some groups have none. The recall of Malignant and of Benign is comparable.

| group | lesions | macro-F1 | Malignant found (recall, 95% interval) | Benign found (recall) |
|---|---|---|---|---|
| all | 787 | 0.597 | 468 of 546 (0.857) | 155 of 223 (0.695) |
| skin tones 0 to 2 (darker) | 82 | 0.526 | **22 of 41 (0.537**, 0.390 to 0.683) | 29 of 40 (0.725) |
| skin tones 3 to 5 (lighter) | 705 | 0.602 | **446 of 505 (0.883**, 0.855 to 0.911) | 126 of 183 (0.689) |
| female | 269 | 0.538 | 140 of 165 (0.848, 0.788 to 0.897) | 75 of 99 (0.758) |
| male | 518 | 0.602 | 328 of 381 (0.861, 0.824 to 0.895) | 80 of 124 (0.645) |

* **Sex:** no difference found. The recall of Malignant is the same; the recall of Benign is 11 points higher for
  women, with overlapping intervals.
* **Skin tone:** the model is worse on darker skin. It finds 22 of 41 malignant lesions on tones 0 to 2 against 446 of
  505 on tones 3 to 5. Part of the gap is the mix of diagnoses (20 of the 41 malignant lesions on darker skin are
  melanomas, 49%, against 9.5% on lighter skin), part is inside the diagnoses (BCC: 12 of 18 found against 338 of 361;
  melanoma: 8 of 20 against 27 of 48). Only 82 test lesions have tones 0 to 2 and only 10 have tones 0 or 1, so the
  size of the gap is uncertain and nothing can be said about the darkest tones.

## Known failures and limits

* **Melanoma.** 33 of the 68 test melanomas are not called Malignant (25 are called Benign). The overall Malignant
  recall of 0.857 hides this because 69% of the malignant lesions are BCC, of which 350 of 379 are found.
* **A pigmentation shortcut.** Brown, sharply bounded lesions are read as benign and pink ones without a border as
  malignant. Benign lesions with little pigment are misclassified 55.7% of the time (more pigmented: 19.0%); malignant
  lesions with much pigment 22.2% of the time (less pigmented: 8.5%).
* **Indeterminate** (actinic keratosis only) is predicted 40 times on test and is right 6 times.
* **The class weights overshoot.** The loss gives Indeterminate a weight of 14. On a check with a linear layer on
  frozen ResNet-18 features, the weights raised the recall of Indeterminate from 9 to 16 of 18 validation lesions but
  made the layer call 102 lesions Indeterminate, and macro-F1 fell from 0.703 to 0.567
  (`m4_best_model/class_weights_check.py`). Over five seeds, every run without the weights beats every run with
  them. The models here keep the weights; retraining with milder ones is the first thing to try.
* **Confident mistakes.** Wrong answers have lower confidence on average (0.72 against 0.85), but 18 of them come
  with 0.95 or more. The probabilities are not calibrated to any clinic's prevalence.
* **Rare classes.** MAL_OTH has 9 lesions in the whole dataset and cannot be evaluated; DF, INF, VASC and BEN_OTH have
  6 to 8 lesions each in validation and in test, so their F1 moves a lot with one lesion.
* **Selection of the data.** Biopsied lesions from one study; no patient identifier, so two lesions of one patient can
  be in train and test.
* **One run.** Each model was trained once with one seed. The intervals measure the sampling noise of the test
  lesions, not the noise between training runs. Validation scores are optimistic (0.682 on validation, 0.597 on test).

## Reproduce

```
pip install -r requirements.txt
python m1_data/milestone1.py                 # split, label map, class weights (needs milk10k/)
python m4_best_model/compare_models.py       # the 3-class model, the ViT comparison, the scoreboard
python m4_best_model/error_analysis.py       # confusion matrix, misclassified images
python m4_best_model/fairness.py             # scores per skin tone and per sex
python m4_best_model/class_weights_check.py  # the linear head with and without class weights, and with 5 seeds
python m5_final/train_11class.py             # the 11-class model and its scores at the 0.5 cut
python m5_final/make_submission.py           # submission.csv (needs milk10k_test/)
python -m pytest tests                       # 35 tests, no dataset needed
```

Seed 42 everywhere. On the 2-core CPU used here the M4 run reproduced the Session 7 model prediction for prediction;
on other hardware small numerical differences can change a few borderline answers.

## Licence and citation

Data: CC BY-NC (non-commercial), attribution "MILK study team". The trained weights inherit that restriction.

* MILK study team. MILK10k Benchmark. ISIC Archive, 2025. doi:10.34970/262082
* Tschandl P, et al. MILK10k: A Hierarchical Multimodal Imaging-Learning Toolkit for Diagnosing Pigmented and
  Nonpigmented Skin Cancer and its Simulators. J Invest Dermatol. 2026;146(2):357-364.e7. doi:10.1016/j.jid.2025.06.1594
* He K, Zhang X, Ren S, Sun J. Deep Residual Learning for Image Recognition. CVPR 2016. ImageNet weights: torchvision.
