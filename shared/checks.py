"""Data checks shared by the Part A notebook (A1.1) and the pipeline (B1, B4).

Every function returns plain pandas objects, so the caller decides whether to
print them, save them in a report or assert on them.
"""

import pandas as pd
from PIL import Image

from . import config, paths

PER_LESION_FIELDS = ["age_approx", "sex", "anatom_site_general",
                     "diagnosis_1", "diagnosis_2", "diagnosis_3", "diagnosis_4"]


def compare_lesion_ids(meta, gt):
    """(a) lesion_ids that appear in only one of the two label files."""
    in_meta, in_gt = set(meta["lesion_id"]), set(gt["lesion_id"])
    return {"only_in_metadata": sorted(in_meta - in_gt), "only_in_gt": sorted(in_gt - in_meta)}


def one_hot_violations(gt):
    """(b) training_gt rows without exactly one positive class, or with values other than 0/1."""
    values = gt[config.SUBCLASSES]
    bad = (values.sum(axis=1) != 1) | ~values.isin([0, 1]).all(axis=1)
    return gt.loc[bad, ["lesion_id"] + config.SUBCLASSES]


def class_to_diagnosis_1(lesions):
    """(c) lesion counts: 11-class label (rows) by diagnosis_1 (columns)."""
    return pd.crosstab(lesions["dx"], lesions["diagnosis_1"])


def ambiguous_classes(table):
    """(c) rows of the class -> diagnosis_1 table that hit more than one diagnosis_1 value."""
    return table[(table > 0).sum(axis=1) > 1]


def hierarchy_violations(meta, child, parent):
    """(d) non-missing child values that sit under more than one parent value."""
    parents = meta.dropna(subset=[child]).groupby(child)[parent].nunique()
    return parents[parents > 1]


def inconsistent_lesions(meta, fields=PER_LESION_FIELDS):
    """(e) per field, the number of lesions whose two images disagree (missing counts as a value)."""
    distinct = meta.groupby("lesion_id")[fields].nunique(dropna=False)
    return (distinct > 1).sum()


def image_pair_violations(meta):
    """Lesions that do not have exactly two images, one dermoscopic and one clinical."""
    types = meta.groupby("lesion_id")["image_type"].agg(lambda s: tuple(sorted(s)))
    expected = tuple(sorted(config.IMAGE_TYPES))
    return types[types != expected]


def inspect_images(meta, image_dir=None):
    """B1: for every row, locate the file, run Image.verify() and record size and bytes.

    Problems are recorded in the error column instead of stopping the loop, so one
    pass lists every missing or unreadable file.
    """
    rows = []
    for isic_id, image_type in zip(meta["isic_id"], meta["image_type"]):
        row = {"isic_id": isic_id, "image_type": image_type,
               "width": pd.NA, "height": pd.NA, "file_bytes": pd.NA, "error": ""}
        try:
            path = paths.find_image(isic_id, image_dir)
            row["file_bytes"] = path.stat().st_size
            with Image.open(path) as img:
                row["width"], row["height"] = img.size
                img.verify()
        except Exception as err:  # missing file, truncated or corrupt JPEG
            row["error"] = f"{type(err).__name__}: {err}"
        rows.append(row)
    return pd.DataFrame(rows)


def size_summary(sizes):
    """One row per image type plus 'all': counts, unreadable files, min/median/max size."""
    groups = [("all", sizes)] + [(t, part) for t, part in sizes.groupby("image_type")]
    rows = []
    for name, part in groups:
        ok = part[part["error"] == ""]
        row = {"image_type": name, "n_images": len(part), "n_unreadable": int((part["error"] != "").sum())}
        for col, out in (("width", "width"), ("height", "height"), ("file_bytes", "file_kb")):
            values = pd.to_numeric(ok[col]) / (1024 if col == "file_bytes" else 1)
            row[f"{out}_min"] = round(values.min(), 1)
            row[f"{out}_median"] = round(values.median(), 1)
            row[f"{out}_max"] = round(values.max(), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def cramers_v(table):
    """Strength of association in a contingency table: 0 none, 1 perfect (same as session 2)."""
    from scipy.stats import chi2_contingency
    chi2 = chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    return (chi2 / (n * (min(table.shape) - 1))) ** 0.5
