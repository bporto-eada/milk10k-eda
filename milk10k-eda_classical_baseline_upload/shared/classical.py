"""Classical ML baseline: lesion-level feature table, feature sets, models and metrics.

Each lesion becomes one row: the hand-crafted features of its dermoscopic image (prefix d_),
of its clinical close-up (prefix c_) and its metadata (meta_). Fusing the two views at the
feature level keeps the unit of evaluation the lesion, as decided in Milestone 1.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config
from .features import metadata_features

VIEW_PREFIX = {"dermoscopic": "d_", "clinical: close-up": "c_"}
GROUP_PREFIX = {"colour": "col_", "texture": "tex_", "shape": "shp_"}


# ---------------------------------------------------------------- lesion table
def lesion_feature_table(image_features, images):
    """One row per lesion: identifiers, labels, d_* and c_* image features, meta_* features.

    image_features  DataFrame indexed by isic_id (output of the features step)
    images          the split CSVs stacked, one row per image, with a split column
    """
    feats = image_features.loc[images["isic_id"]].reset_index(drop=True)
    table = pd.concat([images.reset_index(drop=True), feats], axis=1)
    feature_cols = list(image_features.columns)
    parts = []
    for image_type, prefix in VIEW_PREFIX.items():
        part = table.loc[table["image_type"] == image_type, ["lesion_id"] + feature_cols]
        parts.append(part.set_index("lesion_id").add_prefix(prefix))
    info = (images.groupby("lesion_id")
            .agg(split=("split", "first"), diagnosis_1=("diagnosis_1", "first"), dx=("dx", "first"),
                 age=("age_approx", "first"), sex=("sex", "first"), site=("anatom_site_general", "first")))
    if "image_manipulation" in images:
        info["altered"] = images.groupby("lesion_id")["image_manipulation"].agg(
            lambda s: bool((s == "altered").any()))
    lesions = info.join(parts[0], how="inner").join(parts[1], how="inner")
    meta = pd.DataFrame(metadata_features(lesions.reset_index()), index=lesions.index)
    return pd.concat([lesions, meta], axis=1).reset_index()


# ---------------------------------------------------------------- feature sets
def columns(table, views=("d_", "c_"), groups=("col_", "tex_", "shp_"), metadata=False):
    cols = [c for c in table.columns
            if any(c.startswith(v + g) for v in views for g in groups)]
    if metadata:
        cols += [c for c in table.columns if c.startswith("meta_")]
    return cols


FEATURE_SETS = {
    # one view, all image features
    "derm C+T+S": dict(views=("d_",)),
    "clinical C+T+S": dict(views=("c_",)),
    # both views, one group at a time
    "both colour": dict(groups=("col_",)),
    "both texture": dict(groups=("tex_",)),
    "both shape": dict(groups=("shp_",)),
    "metadata only": dict(groups=(), metadata=True),
    # both views, all image features, then with metadata
    "both C+T+S": dict(),
    "both C+T+S+meta": dict(metadata=True),
}


def feature_set_columns(table, name):
    return columns(table, **FEATURE_SETS[name])


def feature_groups(cols):
    """Column groups for grouped permutation importance: view x feature family, plus metadata."""
    names = {"d_": "derm", "c_": "clinical"}
    fams = {"col_": "colour", "tex_": "texture", "shp_": "shape"}
    groups = {}
    for c in cols:
        if c.startswith("meta_"):
            key = "metadata"
        else:
            key = f"{names[c[:2]]} {fams[c[2:6]]}"
        groups.setdefault(key, []).append(c)
    return groups


# ---------------------------------------------------------------- models
def _pipe(model, scale=True):
    steps = [("impute", SimpleImputer(strategy="median"))]   # age: TRAIN median (Milestone 1)
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def model_grid(seed=config.SEED):
    """name -> (pipeline, hyper-parameter grid). Class imbalance: class_weight='balanced'
    (n / (k * count), the Milestone 1 rule) wherever the model supports it."""
    return {
        "LogReg": (_pipe(LogisticRegression(class_weight="balanced", max_iter=5000)),
                   {"model__C": [0.01, 0.1, 1, 10]}),
        "kNN": (_pipe(KNeighborsClassifier()),
                {"model__n_neighbors": [5, 15, 31], "model__weights": ["uniform", "distance"]}),
        "SVM-RBF": (_pipe(SVC(kernel="rbf", class_weight="balanced", random_state=seed)),
                    {"model__C": [0.3, 1, 3, 10]}),
        "RandomForest": (_pipe(RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                      n_jobs=1, random_state=seed), scale=False),
                         {"model__max_depth": [None, 15], "model__min_samples_leaf": [1, 3, 10]}),
        "GradBoost": (_pipe(HistGradientBoostingClassifier(class_weight="balanced", max_iter=300,
                                                           early_stopping=True, n_iter_no_change=20,
                                                           random_state=seed), scale=False),
                      {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [15, 31]}),
    }


def dummy_models(seed=config.SEED):
    return {"majority (always Malignant)": DummyClassifier(strategy="most_frequent"),
            "stratified random": DummyClassifier(strategy="stratified", random_state=seed)}


def scores(model, X):
    """Per-class scores for ROC-AUC: probabilities if the model has them, else decision values."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except AttributeError:
            pass
    return model.decision_function(X)


# ---------------------------------------------------------------- metrics
def macro_auc(y, s, n_classes):
    """Mean one-vs-rest ROC-AUC over the classes present in y."""
    aucs = [roc_auc_score(y == k, s[:, k]) for k in range(n_classes) if 0 < (y == k).sum() < len(y)]
    return float(np.mean(aucs)) if aucs else np.nan


def evaluate(y, pred, s, classes, positive="Malignant"):
    """Lesion-level metrics. y, pred are integer labels in the order of classes."""
    k = len(classes)
    out = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", labels=range(k), zero_division=0),
        "macro_auc": macro_auc(y, s, k),
    }
    recall = recall_score(y, pred, labels=range(k), average=None, zero_division=0)
    precision = precision_score(y, pred, labels=range(k), average=None, zero_division=0)
    for i, c in enumerate(classes):
        out[f"recall_{c}"] = recall[i]
        out[f"precision_{c}"] = precision[i]
    if positive in classes:
        p = classes.index(positive)
        pos, called = y == p, pred == p
        out["sensitivity_malignant"] = called[pos].mean()
        out["specificity_malignant"] = (~called[~pos]).mean()
    return out


def bootstrap_ci(y, pred, s, classes, n=2000, seed=config.SEED,
                 keys=("balanced_accuracy", "macro_f1", "macro_auc",
                       "sensitivity_malignant", "specificity_malignant")):
    """95% percentile intervals from a stratified bootstrap of the test lesions (class counts fixed,
    so the 18 Indeterminate test lesions never vanish from a resample)."""
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(y == k) for k in range(len(classes))]
    draws = {key: [] for key in keys}
    for _ in range(n):
        idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in by_class if len(ix)])
        m = evaluate(y[idx], pred[idx], s[idx], classes)
        for key in keys:
            draws[key].append(m.get(key, np.nan))
    return {key: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
            for key, v in draws.items()}


def confusion(y, pred, classes):
    return pd.DataFrame(confusion_matrix(y, pred, labels=range(len(classes))),
                        index=[f"true {c}" for c in classes], columns=[f"pred {c}" for c in classes])


def grouped_permutation_importance(model, X, y, groups, n_repeats=10, seed=config.SEED):
    """Drop in balanced accuracy when all columns of a group are shuffled together (rows permuted
    jointly, so the columns inside a group keep their relation). Mean and std over repeats."""
    rng = np.random.default_rng(seed)
    base = balanced_accuracy_score(y, model.predict(X))
    rows = []
    for name, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[cols] = X[cols].to_numpy()[rng.permutation(len(X))]
            drops.append(base - balanced_accuracy_score(y, model.predict(Xp)))
        rows.append({"group": name, "n_features": len(cols),
                     "drop_mean": float(np.mean(drops)), "drop_std": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("drop_mean", ascending=False).reset_index(drop=True)


def refit(pipeline, params, X, y):
    model = clone(pipeline).set_params(**params)
    return model.fit(X, y)
