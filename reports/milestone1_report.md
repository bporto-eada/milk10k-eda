# Milestone 1 report: a leak-free data pipeline for MILK10k

Bruno Curi · Computer Vision and Speech Recognition, EADA · code: `milestone1.py` and `shared/` · every number
below comes from `python milestone1.py` (log in `outputs/milestone1_log.txt`).

## 1. Label strategy

**Primary target, diagnosis_1.** Lesions: Malignant 3,634 (69.4%), Benign 1,483 (28.3%), Indeterminate 123 (2.3%).
All 123 Indeterminate lesions are actinic keratoses: the AK half of AKIEC, whose other 180 lesions (SCC in situ) are
Malignant, so diagnosis_1 cannot be derived from an 11-class prediction (A1.1c). **Decision: keep Indeterminate as a
third class.** An AK is a pre-cancer that is treated (cryotherapy, topical therapy): neither "reassure" (Benign) nor
"excise" (Malignant). Merging it into Benign would teach the model to dismiss scaly red lesions, the look of SCC in situ;
merging it into Malignant erases a distinct treatment path; dropping it hides the hardest cases from the test set while
they still turn up in clinic. Its 87 training lesions get weight 14.05 in the loss and its recall is reported on its own.

**Stretch target, 11 classes.** DF 52, INF 50, VASC 47, BEN_OTH 44 and MAL_OTH 9 lesions. **Decision: keep all 11 and
handle the rare ones with class weights**, not an "other" group: such a group would mix benign classes (BEN_OTH, DF,
INF, VASC) with a malignant one (MAL_OTH), cutting the link to diagnosis_1, and DF and VASC look distinctive enough to
deserve their own output. The price is large weights (MAL_OTH 66.7 against BCC 0.19), which make training
noisier, and MAL_OTH cannot be evaluated reliably (A3.2), so its metrics are reported as not estimable. Mapping:
`outputs/label_map.json`.

## 2. Split design and verification

Lesions are split first and images follow their lesion. `split_lesions` runs StratifiedGroupKFold (groups = lesion_id,
strata = 11-class label refined by diagnosis_1, i.e. AKIEC split into its two values so both targets stay balanced) with
20 folds of 5%: 3 to test, 3 to validation, 14 to train. Seed 42, created 2026-10-05.

| split | lesions | images | share |
|---|---|---|---|
| train | 3,667 | 7,334 | 69.98% |
| val | 786 | 1,572 | 15.00% |
| test | 787 | 1,574 | 15.02% |

Verified in code: 0 lesions shared by any pair of splits, every lesion's two images in the same split, largest
deviation of a class share from the overall share 0.21 pp (11 classes) and 0.11 pp (diagnosis_1). Over 10 seeds
the sizes stay within 0.04 pp of 70/15/15 and the same seed gives the same split, but MAL_OTH reaches both val and test
in only 7 of 10 seeds.

## 3. Imbalance on the train split

diagnosis_1: Malignant 2,542, Benign 1,038, Indeterminate 87 lesions, a ratio of **29.2 : 1**; 11 classes: BCC
1,765 against MAL_OTH 5, **353 : 1**. Both remedies are implemented: a class-weighted cross-entropy with
weights n / (k · count) from the train split (Benign 1.18, Indeterminate 14.05, Malignant 0.48;
`class_weights.json`) and a WeightedRandomSampler. Over 20 train batches the sampler yields 197 / 218 / 225 images of
Benign / Indeterminate / Malignant, against 187 / 10 / 443 with plain shuffling. A training run uses one of the two; both
together would over-correct.

## 4. Data-quality issues and how they were handled

All 10,480 rows resolve to an image file, `Image.verify()` fails on none, and every image is 600x450 (B1). Every lesion
has one dermoscopic and one clinical image, and age, sex, site and diagnosis agree between them (B4). Issues:
(i) **anatom_site_general is missing for 37.3%** of images and the missing group looks like trunk lesions (NV share
+11 pp), so it becomes an "unknown" category rather than an imputed site; age (0.4% missing) gets the train median;
anatom_site_special (98% missing) is dropped. (ii) **Label leaks** never enter the model: diagnosis_2 to 4,
diagnosis_confirm_type (72% of histopathology lesions are malignant against 2% otherwise), concomitant_biopsy and
melanocytic. (iii) **Shortcut risk:** "altered" images are 4.5% of Benign, 17.5% of Indeterminate and 2.2% of
Malignant images, so image_manipulation is excluded and test results will be split by it. File size carries no class
signal (ROC-AUC 0.48 to 0.51, Part A) because every file shares one JPEG quality-75 setting. (iv) MAL_OTH has only 9
lesions in total.

## 5. Preprocessing and augmentation

**Resolution 224x224.** All images are 600x450, so 100% are larger than 224 and nothing is upsampled; 224 matches
ImageNet-pretrained backbones, and 450 px leaves room for a 384 run later. `eval_transform` resizes the shorter side to
224 and centre-crops, which keeps the lesion's proportions (a plain 224x224 resize would squash 4:3 into a square) at
the cost of the outer 12.5% on the left and right; it is deterministic (asserted). **Normalization** from the 7,334
train images only: mean (0.686, 0.528, 0.477), std (0.123, 0.132, 0.152). **Views:** option (a), each image is a sample with its lesion's label and
the two predictions are averaged per lesion (`aggregate_predictions`); (b) one view only would throw away half the data
and the second look a dermatologist uses; (c) both views per lesion is ready as `LesionDataset` for a two-branch model.
Splits are by lesion, so per-image evaluation would not leak, but it would count every lesion twice, so metrics are per
lesion. One train epoch loads in 40.9 s (batch 32, 2 workers).

| augmentation (train only) | parameters | why the label does not change |
|---|---|---|
| RandomResizedCrop | 224, area 0.8 to 1, aspect 3/4 to 4/3 | framing varies; 80% of the frame stays, so the centred lesion is never cut out |
| Horizontal and vertical flips | p = 0.5 each | skin has no canonical left, right, up or down |
| RandomQuarterTurn | 0, 90, 180, 270 degrees | orientation is arbitrary; lossless, no black corners |
| ColorJitter | brightness 0.1, contrast 0.1, no hue, no saturation | measured in A3.5: brightness 0.1 moves lesion brightness by 0.4x the Benign/Malignant gap; brightness 0.3 (the class example) moves it 1.2x and hue 0.05 shifts hue 1.3x the gap |

## 6. Why accuracy on MILK10k is not real-world accuracy

MILK10k is a sample of the lesions that were biopsied, not of the lesions a dermatologist sees. 95.7% of its lesions
were confirmed by histopathology, and a lesion is biopsied when a clinician suspects cancer, which is why 69% are
malignant and 48% are BCC, while most lesions shown in a clinic are benign nevi and keratoses. The 224 lesions that
entered without a biopsy confirm the mechanism: only 2% of them are malignant, because they were admitted for looking
harmless. The benign class is unusual too: it holds the benign lesions that worried someone enough to remove them, not
the easy ones. Two consequences follow for the final model. First, prevalence: accuracy, precision and negative
predictive value all depend on the class mix. A constant "Malignant" answer already scores 69% accuracy here (A3.4),
and in a clinic where cancer is rare the same sensitivity and specificity would turn most positive predictions into
false alarms. Second, spectrum: the test measures how well the model separates lesions that were already suspicious
enough to biopsy, not how it behaves in screening. So the final accuracy will only be read next to per-class recall
(Malignant above all), specificity and the confusion matrix, which depend far less on the class mix, and any use
outside this setting needs recalibration to the local prevalence and a test on consecutive, unselected lesions.

## 7. What could still go wrong

**Residual leakage:** MILK10k has no patient identifier, so two lesions of the same patient (same skin, camera and
session) can sit in train and test; near-duplicate photographs across lesions were not searched for. **Bias:** one
study team and set of devices, mostly middle skin tones (dataset description), older patients (median 65 for malignant
lesions) and biopsy-selected lesions. **Shortcuts:** "altered" photos, rulers, ink and gel bubbles must be audited with
per-subgroup results. **What a clinician should know:** the model learned from biopsied lesions of one source; its
probabilities need recalibration to the clinic's prevalence; "Indeterminate" means actinic keratosis only; MAL_OTH is
not evaluable; it has not been validated prospectively or on darker skin, so it is a second reader for triage and never
a replacement for a biopsy.
