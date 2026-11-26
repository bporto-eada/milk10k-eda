"""Milestone 5: the leaderboard file for the 479 blind test lesions.

    python m5_final/make_submission.py        # writes m5_final/submission.csv (479 rows x 12 columns)

Needs the two test files of challenge.isic-archive.com/data (MILK10k section), unzipped into milk10k_test/
at the repo root (or the folder named by the environment variable MILK10K_TEST_DIR):

    milk10k_test/MILK10k_Test_Metadata.csv
    milk10k_test/MILK10k_Test_Input/IL_xxxxxxx/ISIC_xxxxxxx.jpg      (479 lesion folders, 958 images)

One row per lesion, predicted from its dermoscopic image, the image type the model was trained on.
The test images have no labels: the only thing we ever do with them is predict. No training, no tuning.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))        # the repo root, so that `shared` can be imported

import pandas as pd
import torch
from PIL import Image

from m5_final.predict import load_model
from shared import config
from shared.deep import val_tf                                       # the same transform as validation and predict()

CLASSES = config.SUBCLASSES                                          # the 11 classes in the official column order

# ---------------------------------------------------------------------------------------------------
# 1. The 479 lesions and their dermoscopic image
# HOW: the metadata has one row per image (958). Keeping the dermoscopic rows gives one row per lesion.

meta = pd.read_csv(config.TEST_DIR / "MILK10k_Test_Metadata.csv")
derm = meta[meta["image_type"] == "dermoscopic"].reset_index(drop=True)
files = [config.TEST_DIR / "MILK10k_Test_Input" / row.lesion_id / f"{row.isic_id}.jpg" for row in derm.itertuples()]

# ---------------------------------------------------------------------------------------------------
# 2. Predict
# HOW: the 11-class model, the validation transform, evaluation mode (set by load_model), no gradients.

model, classes = load_model(config.MODEL_11CLASS)
assert classes == CLASSES, "the model was not trained on the 11 classes in the official order"

batches = []
with torch.no_grad():
    for start in range(0, len(files), 32):                           # 32 images at a time
        images = [val_tf(Image.open(f).convert("RGB")) for f in files[start:start + 32]]
        logits = model(torch.stack(images))                          # raw scores, shape (32, 11)
        batches.append(torch.softmax(logits, dim=1))                 # probabilities: every value in [0, 1]
probabilities = torch.cat(batches).numpy()

# ---------------------------------------------------------------------------------------------------
# 3. The file, and the four checks of the project brief
# The upload system rejects a file that fails one of them, so we check before uploading.

sub = pd.DataFrame(probabilities, columns=CLASSES)
sub.insert(0, "lesion", derm["lesion_id"])

assert list(sub.columns) == ["lesion"] + CLASSES                     # exact header, exact order
assert len(sub) == 479 and sub["lesion"].is_unique                   # one row per lesion, not one per image
assert sub[CLASSES].isna().sum().sum() == 0                          # no empty cells
assert ((sub[CLASSES] >= 0) & (sub[CLASSES] <= 1)).all().all()       # probabilities, not raw logits

sub.to_csv(config.SUBMISSION_CSV, index=False, float_format="%.6f")
# The challenge page and the brief name the first column "lesion"; the scoring code that ISIC publishes
# reads "lesion_id". Same values under that name, to upload only if the first file is refused.
sub.rename(columns={"lesion": "lesion_id"}).to_csv(config.M5_DIR / "submission_lesion_id.csv", index=False,
                                                    float_format="%.6f")

yes = sub[CLASSES] >= 0.5                                            # what the leaderboard will read as "yes"
print(f"{config.SUBMISSION_CSV.name}: {len(sub)} lesions x {len(CLASSES)} classes, all checks passed")
print(f"lesions with one probability >= 0.5: {(yes.sum(axis=1) == 1).sum()}, with none: {(yes.sum(axis=1) == 0).sum()}")
print("diagnoses the leaderboard will read:", yes.sum().to_dict())
