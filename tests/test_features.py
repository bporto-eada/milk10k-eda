"""Tests for the classical baseline features and helpers, on synthetic images (no dataset needed).
Run from the repo root:  python -m pytest tests"""

import numpy as np
import pandas as pd
import pytest

from shared import classical
from shared import features as F

H, W = F.WORK_SIZE[1], F.WORK_SIZE[0]


def synthetic_image(shape="ellipse", lesion=(0.45, 0.28, 0.20), skin=(0.85, 0.65, 0.55), seed=0):
    """Skin-coloured background with a darker lesion near the centre, plus mild noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:H, :W]
    if shape == "ellipse":
        mask = ((xx - W / 2) / 70) ** 2 + ((yy - H / 2) / 50) ** 2 <= 1
    else:                                   # an ellipse with a lobe on one side: asymmetric
        mask = ((xx - W / 2) / 60) ** 2 + ((yy - H / 2) / 45) ** 2 <= 1
        mask |= ((xx - W / 2 - 60) / 35) ** 2 + ((yy - H / 2 + 30) / 25) ** 2 <= 1
    img = np.ones((H, W, 3), np.float32) * np.array(skin, np.float32)
    img[mask] = lesion
    img += rng.normal(0, 0.02, img.shape).astype(np.float32)
    return np.clip(img, 0, 1), mask


def test_segmentation_finds_the_lesion():
    rgb, truth = synthetic_image()
    _, mask = F.image_features(rgb)
    assert (mask & truth).sum() / (mask | truth).sum() > 0.95


def test_segmentation_ignores_black_corners():
    rgb, truth = synthetic_image()
    yy, xx = np.mgrid[:H, :W]
    corners = ((xx - W / 2) / (0.62 * W)) ** 2 + ((yy - H / 2) / (0.62 * H)) ** 2 > 1   # vignette
    rgb[corners] = 0.02
    mask, fallback = F.segment_lesion(F.color.rgb2lab(rgb))
    assert fallback == 0
    assert (mask & corners).sum() == 0
    assert (mask & truth).sum() / truth.sum() > 0.9


def test_fallback_on_a_blank_image():
    rgb = np.full((H, W, 3), 0.7, np.float32)
    feats, mask = F.image_features(rgb)
    assert feats["shp_fallback"] == 1
    assert mask.any()


def test_features_are_finite_and_stable_in_number():
    a, _ = F.image_features(synthetic_image(seed=1)[0])
    b, _ = F.image_features(synthetic_image("lobed", seed=2)[0])
    assert list(a) == list(b)
    assert all(np.isfinite(v) for v in a.values())
    assert {k[:4] for k in a} == {"col_", "tex_", "shp_"}


def test_colour_features_see_a_darker_lesion():
    feats, _ = F.image_features(synthetic_image()[0])
    assert feats["col_delta_L"] < -20
    assert feats["col_derm_dark_brown"] > 0.5


def test_asymmetry_is_low_for_an_ellipse_and_higher_for_a_lobed_lesion():
    _, ellipse = synthetic_image()
    _, lobed = synthetic_image("lobed")
    assert max(F.asymmetry(ellipse)) < 0.05
    assert max(F.asymmetry(lobed)) > 0.15


def test_lesion_table_pairs_the_two_views():
    ids = [f"IL_{i}" for i in range(4)]
    images = pd.DataFrame({
        "isic_id": [f"{i}_d" for i in ids] + [f"{i}_c" for i in ids],
        "lesion_id": ids * 2,
        "image_type": ["dermoscopic"] * 4 + ["clinical: close-up"] * 4,
        "diagnosis_1": ["Benign", "Malignant", "Malignant", "Indeterminate"] * 2,
        "dx": ["NV", "BCC", "MEL", "AKIEC"] * 2,
        "age_approx": [40, 70, np.nan, 80] * 2, "sex": ["male", "female"] * 4,
        "anatom_site_general": ["head/neck", np.nan, "upper extremity", "oral/genital"] * 2,
        "split": ["train", "train", "val", "test"] * 2})
    image_features = pd.DataFrame({"col_x": np.arange(8.0), "shp_y": np.arange(8.0) * 10},
                                  index=images["isic_id"])
    table = classical.lesion_feature_table(image_features, images)
    assert len(table) == 4
    row = table.set_index("lesion_id").loc["IL_1"]
    assert row["d_col_x"] == 1 and row["c_col_x"] == 5 and row["c_shp_y"] == 50
    assert row["meta_site_unknown"] == 1 and row["meta_male"] == 0
    assert np.isnan(table.set_index("lesion_id").loc["IL_2", "meta_age"])   # imputed later, train median


def test_feature_sets_select_the_right_columns():
    cols = ["d_col_a", "c_col_a", "d_tex_b", "c_shp_c", "meta_age"]
    table = pd.DataFrame(columns=cols)
    assert classical.feature_set_columns(table, "derm C+T+S") == ["d_col_a", "d_tex_b"]
    assert classical.feature_set_columns(table, "both colour") == ["d_col_a", "c_col_a"]
    assert classical.feature_set_columns(table, "metadata only") == ["meta_age"]
    assert classical.feature_set_columns(table, "both C+T+S+meta") == cols
    assert set(classical.feature_groups(cols)) == {"derm colour", "clinical colour", "derm texture",
                                                    "clinical shape", "metadata"}


def test_metrics_and_stratified_bootstrap():
    y = np.repeat([0, 1, 2], [30, 5, 65])
    pred = y.copy()
    pred[:5] = 2
    s = np.eye(3)[pred] * 0.8 + 0.1
    m = classical.evaluate(y, pred, s, ["Benign", "Indeterminate", "Malignant"])
    assert m["recall_Benign"] == pytest.approx(25 / 30)
    assert m["balanced_accuracy"] == pytest.approx((25 / 30 + 1 + 1) / 3)
    assert m["specificity_malignant"] == pytest.approx(30 / 35)
    ci = classical.bootstrap_ci(y, pred, s, ["Benign", "Indeterminate", "Malignant"], n=200, seed=1)
    lo, hi = ci["balanced_accuracy"]
    assert lo <= m["balanced_accuracy"] <= hi
