# milk10k-eda

Course work for Computer Vision and Speech Recognition. The project builds towards a model that predicts the diagnosis of a skin lesion from its clinical and dermoscopic images, using the MILK10k dataset (ISIC Archive, https://api.isic-archive.com/doi/milk10k/).

## Layout

```
session1.py             session 1 EDA: samples per class and subclass, age and sex per class
session1_exercises.py   session 1 practice questions Q1 to Q4
session2/               session 2 homework, one module per part plus a runner
shared/                 paths and loading code used by both sessions
milk10k/                the dataset, downloaded and unzipped here (not committed)
```

Session 2 modules:

```
metadata_fields.py   part 1: field types, missing values, association with diagnosis_1, leakage and bias flags
colour_profile.py    part 2: intensity histograms and colour statistics per class and per image type
preprocess.py        part 3: resize, grayscale, scaling for one image or a batch, bad files are skipped
batch_loader.py      part 4: LesionBatches, iterates the dataset in shuffled batches with numeric labels
viewer.py            part 5: image grid, class balance chart, batch pixel diagnostics
run_session2.py      runs parts 1 to 5, writes figures/ and results/session2_results.txt
```

## Setup

```
python -m venv venv
source venv/bin/activate      (Windows: venv\Scripts\activate)
pip install -r requirements.txt
```

Download milk10k.zip from the ISIC link above and unzip it into `milk10k/` at the root of this repo, so that `milk10k/metadata.csv`, `milk10k/images/` and `milk10k/supplements/training_gt.csv` exist.

## Run

```
python session1.py
python session1_exercises.py
python session2/run_session2.py
```

All scripts resolve paths relative to their own location, so they also work from the VS Code run button.
