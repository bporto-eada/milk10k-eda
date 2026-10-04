"""Leak-free train / val / test split at lesion level (A3.2, B5).

Lesions are split first; images follow their lesion afterwards, so the two views
of a lesion can never end up on both sides of a split.
"""

import warnings
from math import gcd

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import config

SPLITS = ("train", "val", "test")


def strata(lesions):
    """Stratification key: the 11-class label, refined by diagnosis_1.

    Only AKIEC holds two diagnosis_1 values (Indeterminate and Malignant), so this
    key keeps both the 11-class and the 3-class proportions balanced across splits.
    """
    return lesions["dx"].astype(str) + "|" + lesions["diagnosis_1"].astype(str)


def n_folds(val_size, test_size):
    """Fold count that turns both requested sizes into whole folds (15% and 15% -> 20 folds of 5%)."""
    v, t = round(val_size * 100), round(test_size * 100)
    return 100 // gcd(gcd(v, t), 100)


def split_lesions(lesions, val_size=config.VAL_SIZE, test_size=config.TEST_SIZE, seed=config.SEED):
    """Split lesions into train / val / test; return three sorted lists of lesion_id.

    StratifiedGroupKFold (grouped by lesion_id, stratified on strata()) cuts the table
    into k equal folds. Test and val receive whole folds, drawn with the seed; train
    gets the rest. Same seed -> same three lists.
    """
    k = n_folds(val_size, test_size)
    n_test, n_val = round(test_size * k), round(val_size * k)
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)   # the rarest class has fewer members than k
        folds = [part for _, part in cv.split(lesions, strata(lesions), groups=lesions["lesion_id"])]

    order = np.random.default_rng(seed).permutation(k)
    ids = lesions["lesion_id"].to_numpy()

    def take(fold_numbers):
        return np.unique(ids[np.concatenate([folds[f] for f in fold_numbers])]).tolist()

    test = take(order[:n_test])
    val = take(order[n_test:n_test + n_val])
    train = take(order[n_test + n_val:])
    return train, val, test


def assign_images(images, train, val, test):
    """Add a split column to the image table: every image follows its lesion."""
    split_of = {**dict.fromkeys(train, "train"), **dict.fromkeys(val, "val"), **dict.fromkeys(test, "test")}
    out = images.assign(split=images["lesion_id"].map(split_of))
    if out["split"].isna().any():
        raise ValueError(f"{out['split'].isna().sum()} images belong to no split")
    return out


def proportions(table, col, unit="lesion_id"):
    """Class share (%) per split and overall, counted in lesions."""
    lesions = table.drop_duplicates(unit).reset_index(drop=True)   # works whatever the index
    per_split = pd.crosstab(lesions[col], lesions["split"], normalize="columns").mul(100)
    per_split = per_split.reindex(columns=[s for s in SPLITS if s in per_split.columns])
    per_split["all"] = lesions[col].value_counts(normalize=True).mul(100)
    return per_split.fillna(0.0)


def verify_splits(images, label_cols=("dx", "diagnosis_1")):
    """Check an image table with a split column; raise AssertionError if a rule is broken.

    Returns the overlap counts, images-per-lesion check, split sizes and the class
    proportion tables with their largest deviation from the overall proportions.
    """
    ids = {s: set(images.loc[images["split"] == s, "lesion_id"]) for s in SPLITS}
    overlaps = {f"{a}-{b}": len(ids[a] & ids[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    assert all(n == 0 for n in overlaps.values()), f"lesions shared between splits: {overlaps}"

    per_lesion = images.groupby("lesion_id").agg(n_images=("isic_id", "size"), n_splits=("split", "nunique"))
    bad = per_lesion[(per_lesion["n_images"] != 2) | (per_lesion["n_splits"] != 1)]
    assert bad.empty, f"{len(bad)} lesions without exactly 2 images in one split"

    sizes = images.drop_duplicates("lesion_id")["split"].value_counts().reindex(SPLITS)
    report = {"overlaps": overlaps, "lesions_not_paired": len(bad),
              "sizes": pd.DataFrame({"lesions": sizes, "images": images["split"].value_counts().reindex(SPLITS),
                                     "share_pct": (sizes / sizes.sum() * 100).round(2)})}
    for col in label_cols:
        table = proportions(images, col)
        deviation = table[list(SPLITS)].sub(table["all"], axis=0).abs()
        report[col] = table.round(2)
        report[f"{col}_max_deviation_pp"] = round(float(deviation.to_numpy().max()), 2)
    return report
