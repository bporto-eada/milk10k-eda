"""Property tests for split_lesions (A3.2, B5). Run from the repo root:  python -m pytest tests"""

import numpy as np
import pandas as pd
import pytest

from shared.splits import assign_images, split_lesions, verify_splits

CLASSES = ["BCC", "NV", "MEL", "AKIEC", "DF", "MAL_OTH"]
MALIGNANT = {"BCC", "MEL", "MAL_OTH"}


def synthetic_lesions(n=1000, seed=0):
    """A lesion table shaped like MILK10k: imbalanced classes, AKIEC split over two diagnosis_1 values."""
    rng = np.random.default_rng(seed)
    dx = rng.choice(CLASSES, size=n, p=[0.5, 0.2, 0.12, 0.1, 0.06, 0.02])
    d1 = np.where(np.isin(dx, list(MALIGNANT)), "Malignant", "Benign")
    akiec = dx == "AKIEC"
    d1[akiec] = rng.choice(["Indeterminate", "Malignant"], size=akiec.sum())
    return pd.DataFrame({"lesion_id": [f"IL_{i:07d}" for i in range(n)], "dx": dx, "diagnosis_1": d1})


@pytest.mark.parametrize("seed", range(10))
def test_no_overlap_and_requested_sizes(seed):
    lesions = synthetic_lesions()
    train, val, test = split_lesions(lesions, val_size=0.15, test_size=0.15, seed=seed)
    assert not set(train) & set(val)
    assert not set(train) & set(test)
    assert not set(val) & set(test)
    assert len(train) + len(val) + len(test) == len(lesions)
    for ids, wanted in ((train, 0.70), (val, 0.15), (test, 0.15)):
        assert abs(len(ids) / len(lesions) - wanted) <= 0.01


def test_same_seed_same_output_and_new_seed_new_output():
    lesions = synthetic_lesions()
    assert split_lesions(lesions, 0.15, 0.15, seed=3) == split_lesions(lesions, 0.15, 0.15, seed=3)
    assert split_lesions(lesions, 0.15, 0.15, seed=3) != split_lesions(lesions, 0.15, 0.15, seed=4)


def test_images_follow_their_lesion():
    lesions = synthetic_lesions(400)
    images = pd.concat([lesions.assign(isic_id=lesions["lesion_id"] + "_derm", image_type="dermoscopic"),
                        lesions.assign(isic_id=lesions["lesion_id"] + "_clin", image_type="clinical: close-up")])
    table = assign_images(images, *split_lesions(lesions, 0.15, 0.15, seed=0))
    report = verify_splits(table)
    assert report["lesions_not_paired"] == 0
    assert all(n == 0 for n in report["overlaps"].values())
