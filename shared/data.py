"""Loading the label files and shaping them into the tables the project uses.

load_tables()          one row per image: metadata.csv plus the 11-class label dx
build_lesion_table()   one row per lesion: derm_id, clinical_id, labels, age, sex, site
missing_images()       isic_ids whose file is not on disk
require_images()       fail loudly when an image is missing, unless explicitly allowed
on_disk()              the session 2 filter that silently keeps only present images
"""

import warnings

import pandas as pd

from . import config, paths

LESION_COLUMNS = ["lesion_id", "derm_id", "clinical_id", "diagnosis_1", "dx", "age", "sex", "site"]


def load_metadata():
    """metadata.csv as published by ISIC: one row per image."""
    return pd.read_csv(config.METADATA_CSV)


def load_ground_truth():
    """training_gt.csv (one row per lesion, one-hot over the 11 classes) plus a dx column."""
    gt = pd.read_csv(config.LABELS_CSV)
    gt["dx"] = gt[config.SUBCLASSES].idxmax(axis=1)
    return gt


def load_tables():
    """Image-level table: every metadata.csv row with the lesion's 11-class label dx."""
    meta = load_metadata()
    gt = load_ground_truth()
    return meta.merge(gt[["lesion_id", "dx"]], on="lesion_id", how="left", validate="many_to_one")


def build_lesion_table(images):
    """One row per lesion, built with pivot and groupby (no loops over rows).

    derm_id and clinical_id are the isic_id of the lesion's two images. pivot raises
    if a lesion has two images of the same type, so a broken pair cannot slip through.
    """
    ids = images.pivot(index="lesion_id", columns="image_type", values="isic_id")
    ids = ids.rename(columns={t: f"{short}_id" for t, short in config.IMAGE_TYPES.items()})
    info = images.groupby("lesion_id").agg(
        diagnosis_1=("diagnosis_1", "first"),
        dx=("dx", "first"),
        age=("age_approx", "first"),
        sex=("sex", "first"),
        site=("anatom_site_general", "first"),
    )
    lesions = ids[["derm_id", "clinical_id"]].join(info).reset_index()
    lesions.columns.name = None
    return lesions[LESION_COLUMNS]


def missing_images(df, image_dir=None):
    """List the isic_ids in df that have no image file."""
    present = df["isic_id"].map(lambda i: paths.image_exists(i, image_dir))
    return df.loc[~present, "isic_id"].tolist()


def require_images(df, allow_missing=False, image_dir=None):
    """Return df unchanged if every image exists; otherwise raise FileNotFoundError.

    With allow_missing=True the rows without a file are dropped instead, with a warning,
    so skipping images is always an explicit choice.
    """
    missing = missing_images(df, image_dir)
    if missing and not allow_missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(df)} images are missing, first ones: {missing[:5]}. "
            "Fix the data folder (MILK10K_DIR) or pass allow_missing=True to drop them."
        )
    if missing:
        warnings.warn(f"dropping {len(missing)} rows whose image file is missing")
    return df[~df["isic_id"].isin(missing)].reset_index(drop=True)


def on_disk(df):
    """Session 2 helper: keep only the rows whose image exists (silently)."""
    present = df["isic_id"].map(paths.image_exists)
    return df[present].reset_index(drop=True)
