"""Classical ML baseline for MILK10k: hand-crafted features + scikit-learn classifiers.

    python classical_baseline.py                    run every step, in order
    python classical_baseline.py compare final      run only the named steps

Steps
  features   colour, texture and shape features of all 10,480 images (shared/features.py),
             saved to image_features.csv.gz; segmentation check and example figure
             (the only step that needs the images; classical_baseline_colab.ipynb runs it on Colab)
  audit      lesion-mask quality: manual verdicts on 120 images (segmentation_audit.csv) checked
             against an automatic flag, which then estimates the failure rate on all images
  compare    8 feature sets x 5 models: hyper-parameters tuned by 5-fold CV on the TRAIN lesions,
             each tuned configuration scored once on VAL; the best one is chosen on VAL
  final      the chosen configuration refitted on TRAIN+VAL and scored ONCE on TEST: metrics with
             bootstrap intervals, confusion matrix, ROC curves, triage threshold, subgroups,
             feature-group importance, majority and random baselines
  stretch    the chosen feature set and model family on the 11-class label dx

Target: diagnosis_1 (Benign / Indeterminate / Malignant), one row per lesion (the dermoscopic and
clinical features side by side). Splits: outputs/splits from Milestone 1, unchanged.
Primary metric: balanced accuracy (mean recall over the 3 classes); a constant "Malignant" answer
already scores 69% plain accuracy, so accuracy alone says little.
Every file goes to outputs/classical_baseline/; the console output is also appended to its log.txt.
"""

import json
import sys
import time
import warnings
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from shared import classical, config, figstyle, paths  # noqa: E402
from shared import features as feats  # noqa: E402

OUT = config.BASELINE_DIR
FIG = config.BASELINE_FIGURES_DIR
figstyle.FIGURES = FIG
SPLIT_FILES = {s: config.SPLITS_DIR / f"{s}.csv" for s in ("train", "val", "test")}

# class order = integer labels of the Milestone 1 label map (read from the file, no torch needed)
_LABEL_MAP = json.loads(config.LABEL_MAP_JSON.read_text())
PRIMARY_CLASSES = sorted(_LABEL_MAP[config.LABEL], key=_LABEL_MAP[config.LABEL].get)
DX_CLASSES = sorted(_LABEL_MAP[config.STRETCH_LABEL], key=_LABEL_MAP[config.STRETCH_LABEL].get)


# ---------------------------------------------------------------- helpers
class Tee:
    """Print to the console and append the same text to a log file."""

    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "a", encoding="utf-8")
        self.console = sys.stdout

    def write(self, text):
        self.console.write(text)
        self.file.write(text)
        self.file.flush()             # the log stays current during long steps

    def flush(self):
        self.console.flush()
        self.file.flush()

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, *exc):
        sys.stdout = self.console
        self.file.close()


def md_table(df, index=False, floats=3):
    frame = df.reset_index() if index else df
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.{floats}f}"
        return str(v)
    lines = ["| " + " | ".join(str(c) for c in frame.columns) + " |",
             "|" + "|".join("---" for _ in frame.columns) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path.relative_to(config.REPO_ROOT)}")


def write_json(path, obj):
    write(path, json.dumps(obj, indent=2, default=float))


def load_images():
    """The Milestone 1 split files stacked (one row per image), plus image_manipulation (used only
    to split the test results, never as an input). image_manipulation comes from metadata.csv, or
    from the copy the features step saves, so the later steps run without the dataset."""
    images = pd.concat([pd.read_csv(f).assign(split=s) for s, f in SPLIT_FILES.items()], ignore_index=True)
    source = config.METADATA_CSV if config.METADATA_CSV.exists() else config.MANIPULATION_CSV
    if source.exists():
        manip = pd.read_csv(source, usecols=["isic_id", "image_manipulation"])
        images = images.merge(manip, on="isic_id", how="left", validate="one_to_one")
    return images


def load_lesions():
    if not config.IMAGE_FEATURES_CSV.exists():
        raise FileNotFoundError("image_features.csv.gz not found: run  python classical_baseline.py features")
    image_features = pd.read_csv(config.IMAGE_FEATURES_CSV, index_col="isic_id")
    return classical.lesion_feature_table(image_features, load_images())


def encode(series, classes):
    return series.map({c: i for i, c in enumerate(classes)}).to_numpy()


def cv(seed=config.SEED):
    return StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=seed)


def tune(pipeline, grid, X, y):
    search = GridSearchCV(pipeline, grid, cv=cv(), n_jobs=-1, refit="bal_acc",
                          scoring={"bal_acc": "balanced_accuracy", "macro_f1": "f1_macro"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # rare classes below n_splits in the 11-class run
        search.fit(X, y)
    best = search.best_index_
    res = search.cv_results_
    return search, {"cv_bal_acc": res["mean_test_bal_acc"][best], "cv_bal_acc_std": res["std_test_bal_acc"][best],
                    "cv_macro_f1": res["mean_test_macro_f1"][best]}


# ---------------------------------------------------------------- features
def _one(isic_id):
    try:
        return isic_id, feats.features_of_file(paths.find_image(isic_id)), ""
    except Exception as err:                 # recorded, never silently skipped
        return isic_id, {}, f"{type(err).__name__}: {err}"


def step_features():
    images = load_images()
    ids = images["isic_id"].tolist()
    print(f"extracting features of {len(ids):,} images at {feats.WORK_SIZE[0]}x{feats.WORK_SIZE[1]} ...")
    start = time.time()
    results = Parallel(n_jobs=-1, batch_size=32)(delayed(_one)(i) for i in ids)
    errors = [(i, e) for i, _, e in results if e]
    print(f"done in {time.time() - start:.0f} s; failures: {len(errors)}")
    for i, e in errors[:10]:
        print("  ", i, e)
    table = pd.DataFrame([f for _, f, e in results if not e], index=[i for i, _, e in results if not e])
    table.index.name = "isic_id"
    config.IMAGE_FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.IMAGE_FEATURES_CSV, float_format="%.6g")
    if "image_manipulation" in images:
        images[["isic_id", "image_manipulation"]].to_csv(config.MANIPULATION_CSV, index=False)
    print(f"wrote {config.IMAGE_FEATURES_CSV.relative_to(config.REPO_ROOT)}: "
          f"{table.shape[0]:,} images x {table.shape[1]} features "
          f"(colour {sum(c.startswith('col_') for c in table)}, texture {sum(c.startswith('tex_') for c in table)}, "
          f"shape {sum(c.startswith('shp_') for c in table)})")

    # segmentation check: how often the fallback ellipse was needed, and lesion size, per view and class
    seg = images.set_index("isic_id").join(table[["shp_fallback", "shp_area_frac"]], how="inner")
    summary = (seg.groupby(["image_type", "diagnosis_1"])
               .agg(images=("shp_fallback", "size"), fallback_pct=("shp_fallback", lambda s: 100 * s.mean()),
                    median_lesion_area_pct=("shp_area_frac", lambda s: 100 * s.median()))
               .reset_index())
    print(summary.to_string(index=False))
    write(OUT / "segmentation_summary.md",
          "# Lesion segmentation check\n\nOtsu threshold on smoothed CIELAB L*, the dark component with most "
          "pixels in the central ellipse (shared/features.py, segment_lesion). The fallback is a central "
          "ellipse, used when no plausible component was found.\n\n" + md_table(summary, floats=1) + "\n")
    figure_segmentation(images)


def figure_segmentation(images, per_class=2, seed=config.SEED):
    """Examples of the lesion mask: one row per class, per_class lesions, both views of each."""
    lesions = images.drop_duplicates("lesion_id")
    fig, axes = plt.subplots(len(PRIMARY_CLASSES), 2 * per_class, figsize=(2.3 * 2 * per_class, 1.95 * 3))
    for r, cls in enumerate(PRIMARY_CLASSES):
        picks = lesions[lesions["diagnosis_1"] == cls].sample(n=per_class, random_state=seed)
        for p, (_, lesion) in enumerate(picks.iterrows()):
            pair = images[images["lesion_id"] == lesion["lesion_id"]].sort_values("image_type", ascending=False)
            for v, (_, im) in enumerate(pair.iterrows()):
                rgb = feats.load_rgb(paths.find_image(im["isic_id"]))
                _, mask = feats.image_features(rgb)
                ax = axes[r, 2 * p + v]
                ax.imshow(rgb)
                ax.contour(mask, levels=[0.5], colors=["#00e5ff"], linewidths=1.0)
                ax.set_title(f"{cls} ({lesion['dx']}), {im['image_type'].split(':')[0]}", fontsize=7)
                ax.axis("off")
    fig.tight_layout(pad=0.4)
    figstyle.finish(fig, "segmentation_examples.png")


# ---------------------------------------------------------------- compare
def step_compare():
    lesions = load_lesions()
    train, val = lesions[lesions["split"] == "train"], lesions[lesions["split"] == "val"]
    y_train, y_val = encode(train["diagnosis_1"], PRIMARY_CLASSES), encode(val["diagnosis_1"], PRIMARY_CLASSES)
    print(f"lesions: train {len(train):,}, val {len(val):,}")

    rows = []
    for fs in classical.FEATURE_SETS:
        cols = classical.feature_set_columns(lesions, fs)
        for name, (pipeline, grid) in classical.model_grid().items():
            start = time.time()
            search, cv_scores = tune(pipeline, grid, train[cols], y_train)
            model = search.best_estimator_
            m = classical.evaluate(y_val, model.predict(val[cols]), classical.scores(model, val[cols]),
                                   PRIMARY_CLASSES)
            params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
            rows.append({"feature_set": fs, "n_features": len(cols), "model": name, "params": json.dumps(params),
                         **cv_scores, **{f"val_{k}": v for k, v in m.items()}})
            print(f"{fs:<18} {name:<13} {len(cols):>4} feats  CV bal.acc {cv_scores['cv_bal_acc']:.3f}"
                  f"  VAL bal.acc {m['balanced_accuracy']:.3f}  macro-F1 {m['macro_f1']:.3f}"
                  f"  AUC {m['macro_auc']:.3f}  ({time.time() - start:.0f} s)")

    for name, dummy in classical.dummy_models().items():
        dummy.fit(train[["meta_age"]], y_train)
        m = classical.evaluate(y_val, dummy.predict(val[["meta_age"]]), dummy.predict_proba(val[["meta_age"]]),
                               PRIMARY_CLASSES)
        rows.append({"feature_set": "none", "n_features": 0, "model": name, "params": "{}",
                     **{f"val_{k}": v for k, v in m.items()}})

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "comparison.csv", index=False)
    print(f"wrote {(OUT / 'comparison.csv').relative_to(config.REPO_ROOT)}")

    real = table[table["feature_set"] != "none"]
    best = real.sort_values(["val_balanced_accuracy", "val_macro_auc"], ascending=False).iloc[0]
    choice = {"feature_set": best["feature_set"], "model": best["model"], "params": json.loads(best["params"]),
              "selected_on": "highest VAL balanced accuracy (ties: VAL macro ROC-AUC)",
              "cv_bal_acc": best["cv_bal_acc"], "val_balanced_accuracy": best["val_balanced_accuracy"],
              "val_macro_f1": best["val_macro_f1"], "val_macro_auc": best["val_macro_auc"],
              "date": str(date.today()), "seed": config.SEED}
    write_json(OUT / "best_config.json", choice)
    print(f"\nchosen: {choice['feature_set']} + {choice['model']} {choice['params']}  "
          f"VAL balanced accuracy {choice['val_balanced_accuracy']:.3f}")

    show = table[["feature_set", "model", "n_features", "cv_bal_acc", "val_balanced_accuracy", "val_macro_f1",
                  "val_macro_auc", "val_recall_Benign", "val_recall_Indeterminate", "val_recall_Malignant"]]
    show = show.rename(columns={"cv_bal_acc": "CV bal.acc", "val_balanced_accuracy": "VAL bal.acc",
                                "val_macro_f1": "VAL macro-F1", "val_macro_auc": "VAL macro-AUC",
                                "val_recall_Benign": "rec Ben", "val_recall_Indeterminate": "rec Ind",
                                "val_recall_Malignant": "rec Mal"})
    write(OUT / "comparison.md", "# Feature sets x models, selection on VAL\n\nHyper-parameters by "
          f"{config.CV_FOLDS}-fold CV on the train lesions (balanced accuracy); each tuned model scored once "
          "on the val lesions.\n\n" + md_table(show) + "\n")
    figure_comparison(real)


def figure_comparison(real):
    pivot = real.pivot(index="feature_set", columns="model", values="val_balanced_accuracy")
    pivot = pivot.loc[list(classical.FEATURE_SETS), list(classical.model_grid())]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    im = ax.imshow(pivot.to_numpy(), cmap="Blues", vmin=np.floor(pivot.min().min() * 20) / 20,
                   vmax=pivot.max().max())
    ax.set_xticks(range(pivot.shape[1]), pivot.columns, fontsize=8, rotation=20, ha="right")
    ax.set_yticks(range(pivot.shape[0]), pivot.index, fontsize=8)
    best = np.unravel_index(np.nanargmax(pivot.to_numpy()), pivot.shape)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iat[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    fontweight="bold" if (i, j) == best else "normal",
                    color="white" if v > pivot.to_numpy().mean() + 0.5 * pivot.to_numpy().std() else "black")
    ax.set_title("Validation balanced accuracy (diagnosis_1, 3 classes); chance = 0.333", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    figstyle.finish(fig, "comparison_val_balanced_accuracy.png")


# ---------------------------------------------------------------- final
def step_final():
    lesions = load_lesions()
    choice = json.loads((OUT / "best_config.json").read_text())
    pipeline, _ = classical.model_grid()[choice["model"]]
    params = {f"model__{k}": v for k, v in choice["params"].items()}
    cols = classical.feature_set_columns(lesions, choice["feature_set"])

    train = lesions[lesions["split"] == "train"]
    val = lesions[lesions["split"] == "val"]
    dev = lesions[lesions["split"].isin(["train", "val"])]
    test = lesions[lesions["split"] == "test"]
    y_dev, y_test = encode(dev["diagnosis_1"], PRIMARY_CLASSES), encode(test["diagnosis_1"], PRIMARY_CLASSES)
    print(f"final model: {choice['feature_set']} + {choice['model']} {choice['params']}, "
          f"fitted on train+val ({len(dev):,} lesions), scored once on test ({len(test):,} lesions)")

    model = classical.refit(pipeline, params, dev[cols], y_dev)
    pred, s = model.predict(test[cols]), classical.scores(model, test[cols])
    metrics = classical.evaluate(y_test, pred, s, PRIMARY_CLASSES)
    ci = classical.bootstrap_ci(y_test, pred, s, PRIMARY_CLASSES, n=config.BOOTSTRAP_RESAMPLES)
    cm = classical.confusion(y_test, pred, PRIMARY_CLASSES)
    print(cm.to_string())
    for k, v in metrics.items():
        print(f"  {k:<26} {v:.3f}" + (f"   95% CI {ci[k][0]:.3f} to {ci[k][1]:.3f}" if k in ci else ""))

    # baselines on the same test lesions
    base_rows = []
    for name, dummy in classical.dummy_models().items():
        dummy.fit(dev[["meta_age"]], y_dev)
        bm = classical.evaluate(y_test, dummy.predict(test[["meta_age"]]), dummy.predict_proba(test[["meta_age"]]),
                                PRIMARY_CLASSES)
        base_rows.append({"model": name, **bm})
    base_rows.append({"model": f"{choice['model']} on {choice['feature_set']}", **metrics})
    baselines = pd.DataFrame(base_rows)

    # triage operating point: Malignant threshold reaching 90% sensitivity, chosen on out-of-fold
    # predictions of train+val (never on test), then applied to the test lesions
    mal = PRIMARY_CLASSES.index("Malignant")
    oof = cross_val_predict(clone(pipeline).set_params(**params), dev[cols], y_dev, cv=cv(),
                            method="predict_proba" if hasattr(model, "predict_proba") else "decision_function",
                            n_jobs=-1)
    pos = oof[y_dev == mal, mal]
    threshold = float(np.quantile(pos, 1 - config.TRIAGE_SENSITIVITY))
    called = s[:, mal] >= threshold
    triage = {"threshold": threshold, "chosen_on": "out-of-fold scores of train+val",
              "target_sensitivity": config.TRIAGE_SENSITIVITY,
              "test_sensitivity": float(called[y_test == mal].mean()),
              "test_specificity": float((~called[y_test != mal]).mean()),
              "test_benign_flagged_pct": float(100 * called[y_test == 0].mean()),
              "test_indeterminate_flagged_pct": float(100 * called[y_test == 1].mean())}
    print("triage threshold:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in triage.items()})

    # subgroups: lesions with an 'altered' image (shortcut check from Milestone 1)
    subgroups = []
    if "altered" in test:
        for flag, part in test.groupby("altered"):
            idx = test.index.get_indexer(part.index)
            sm = classical.evaluate(y_test[idx], pred[idx], s[idx], PRIMARY_CLASSES)
            subgroups.append({"subgroup": "has an altered image" if flag else "instrument only",
                              "lesions": len(part), "balanced_accuracy": sm["balanced_accuracy"],
                              "accuracy": sm["accuracy"], "recall_Benign": sm["recall_Benign"],
                              "recall_Indeterminate": sm["recall_Indeterminate"],
                              "recall_Malignant": sm["recall_Malignant"]})
    subgroups = pd.DataFrame(subgroups)

    # which feature groups the chosen model relies on: model fitted on train, shuffled on val
    y_train, y_val = encode(train["diagnosis_1"], PRIMARY_CLASSES), encode(val["diagnosis_1"], PRIMARY_CLASSES)
    train_model = classical.refit(pipeline, params, train[cols], y_train)
    importance = classical.grouped_permutation_importance(train_model, val[cols], y_val,
                                                          classical.feature_groups(cols))
    print(importance.to_string(index=False))

    results = {"choice": choice, "test_lesions": int(len(test)), "metrics": metrics,
               "ci95": ci, "confusion": cm.to_dict(), "triage": triage}
    write_json(OUT / "test_results.json", results)
    baselines.to_csv(OUT / "test_vs_baselines.csv", index=False)
    importance.to_csv(OUT / "feature_group_importance.csv", index=False)
    if len(subgroups):
        subgroups.to_csv(OUT / "test_subgroups.csv", index=False)

    lines = ["# Test results of the chosen classical baseline\n",
             f"Configuration: **{choice['feature_set']} + {choice['model']}** {choice['params']}, chosen on VAL, "
             f"refitted on train+val ({len(dev):,} lesions) and scored once on the {len(test):,} test lesions. "
             f"95% intervals: stratified bootstrap, {config.BOOTSTRAP_RESAMPLES:,} resamples.\n",
             "| metric | test | 95% CI |", "|---|---|---|"]
    for k, v in metrics.items():
        lines.append(f"| {k} | {v:.3f} | " + (f"{ci[k][0]:.3f} to {ci[k][1]:.3f}" if k in ci else "") + " |")
    lines += ["\n## Confusion matrix (lesions)\n", md_table(cm, index=True),
              "\n## Against trivial baselines (same test lesions)\n",
              md_table(baselines[["model", "accuracy", "balanced_accuracy", "macro_f1", "macro_auc",
                                  "recall_Benign", "recall_Indeterminate", "recall_Malignant"]]),
              "\n## Triage operating point\n",
              f"Malignant threshold for {config.TRIAGE_SENSITIVITY:.0%} sensitivity, chosen on out-of-fold "
              f"predictions of train+val: on test, sensitivity {triage['test_sensitivity']:.3f}, specificity "
              f"{triage['test_specificity']:.3f}; {triage['test_benign_flagged_pct']:.1f}% of benign and "
              f"{triage['test_indeterminate_flagged_pct']:.1f}% of indeterminate lesions are flagged.\n",
              "\n## Feature-group importance (model fitted on train, groups shuffled on val)\n",
              md_table(importance)]
    if len(subgroups):
        lines += ["\n## Subgroups: lesions with an 'altered' image\n", md_table(subgroups)]
    write(OUT / "test_results.md", "\n".join(lines) + "\n")

    figure_confusion(cm)
    figure_roc(y_test, s, triage)
    figure_importance(importance)


def figure_confusion(cm):
    counts = cm.to_numpy()
    share = counts / counts.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{counts[i, j]}\n{share[i, j]:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if share[i, j] > 0.55 else "black")
    ax.set_xticks(range(3), PRIMARY_CLASSES, fontsize=8)
    ax.set_yticks(range(3), PRIMARY_CLASSES, fontsize=8)
    ax.set_xlabel("predicted"), ax.set_ylabel("true")
    ax.set_title("Test confusion matrix (lesions; row %)", fontsize=9)
    figstyle.finish(fig, "test_confusion_matrix.png")


def figure_roc(y, s, triage):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for k, c in enumerate(PRIMARY_CLASSES):
        fpr, tpr, _ = roc_curve(y == k, s[:, k])
        ax.plot(fpr, tpr, color=figstyle.colour(c), label=f"{c} vs rest, AUC {roc_auc_score(y == k, s[:, k]):.3f}")
    ax.plot([0, 1], [0, 1], color="#bbbbbb", linestyle="--", linewidth=0.8)
    ax.scatter([1 - triage["test_specificity"]], [triage["test_sensitivity"]], color=figstyle.colour("Malignant"),
               zorder=5, s=30, label="triage threshold (from train+val)")
    ax.set_xlabel("1 - specificity"), ax.set_ylabel("sensitivity")
    ax.set_title("Test ROC curves, one vs rest", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    figstyle.tidy(ax)
    figstyle.finish(fig, "test_roc.png")


def figure_importance(importance):
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    order = importance.iloc[::-1]
    ax.barh(order["group"], order["drop_mean"], xerr=order["drop_std"], color="#3b6ea5")
    ax.set_xlabel("drop in VAL balanced accuracy when the group is shuffled")
    ax.set_title("Which feature groups the chosen model uses", fontsize=9)
    figstyle.tidy(ax)
    figstyle.finish(fig, "feature_group_importance.png")


# ---------------------------------------------------------------- stretch
def step_stretch():
    lesions = load_lesions()
    choice = json.loads((OUT / "best_config.json").read_text())
    pipeline, grid = classical.model_grid()[choice["model"]]
    cols = classical.feature_set_columns(lesions, choice["feature_set"])
    classes = DX_CLASSES
    train, val, test = (lesions[lesions["split"] == s] for s in ("train", "val", "test"))
    dev = pd.concat([train, val])
    y_train, y_val, y_dev, y_test = (encode(t["dx"], classes) for t in (train, val, dev, test))

    search, cv_scores = tune(pipeline, grid, train[cols], y_train)
    val_m = classical.evaluate(y_val, search.best_estimator_.predict(val[cols]),
                               classical.scores(search.best_estimator_, val[cols]), classes)
    model = classical.refit(pipeline, search.best_params_, dev[cols], y_dev)
    pred = model.predict(test[cols])
    test_m = classical.evaluate(y_test, pred, classical.scores(model, test[cols]), classes)
    majority = classical.evaluate(y_test, np.full_like(y_test, np.bincount(y_dev).argmax()),
                                  np.eye(len(classes))[np.full_like(y_test, np.bincount(y_dev).argmax())], classes)
    print(f"11-class dx, {choice['feature_set']} + {choice['model']}: CV bal.acc {cv_scores['cv_bal_acc']:.3f}, "
          f"VAL {val_m['balanced_accuracy']:.3f}, TEST {test_m['balanced_accuracy']:.3f} "
          f"(majority class {majority['balanced_accuracy']:.3f}; chance {1 / len(classes):.3f})")

    per_class = pd.DataFrame({"class": classes,
                              "test_lesions": [int((y_test == k).sum()) for k in range(len(classes))],
                              "recall": [test_m[f"recall_{c}"] for c in classes],
                              "precision": [test_m[f"precision_{c}"] for c in classes]})
    per_class["note"] = np.where(per_class["test_lesions"] < 10, "fewer than 10 test lesions: not estimable", "")
    summary = {"feature_set": choice["feature_set"], "model": choice["model"],
               "params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
               **cv_scores, "val_balanced_accuracy": val_m["balanced_accuracy"],
               **{f"test_{k}": test_m[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auc")},
               "majority_test_balanced_accuracy": majority["balanced_accuracy"],
               "majority_test_accuracy": majority["accuracy"]}
    write_json(OUT / "stretch_dx_results.json", summary)
    per_class.to_csv(OUT / "stretch_dx_per_class.csv", index=False)
    write(OUT / "stretch_dx.md",
          "# Stretch target: 11-class dx\n\n"
          f"Same feature set and model family as the chosen 3-class baseline ({choice['feature_set']} + "
          f"{choice['model']}), hyper-parameters re-tuned by CV on train, refitted on train+val, scored once on test.\n\n"
          f"Test: accuracy {test_m['accuracy']:.3f}, balanced accuracy {test_m['balanced_accuracy']:.3f}, "
          f"macro-F1 {test_m['macro_f1']:.3f}, macro ROC-AUC {test_m['macro_auc']:.3f}. Majority class (BCC): "
          f"accuracy {majority['accuracy']:.3f}, balanced accuracy {majority['balanced_accuracy']:.3f}.\n\n"
          + md_table(per_class) + "\n")


# ---------------------------------------------------------------- audit
AUDIT_CSV = OUT / "segmentation_audit.csv"
VERDICTS = {"G": "good", "P": "partial", "F": "failed", "B": "fallback ellipse"}
SUSPECT_BORDER = 0.05      # a mask covering more than 5% of the image border is flagged


def suspect_mask(table):
    """Automatic flag for a doubtful lesion mask: it runs along the image border (lesions are centred,
    so a mask on the border has usually caught skin, shadow or ink), or the fallback ellipse was used."""
    return (table["shp_border_touch"] > SUSPECT_BORDER) | (table["shp_fallback"] == 1)


def step_audit():
    """Mask quality. segmentation_audit.csv holds a visual verdict (G good, P partial, F failed,
    B fallback ellipse) for 120 images: 20 lesions per diagnosis_1 class drawn with seed 7 in the Colab
    notebook, both views, masks drawn as in figures/segmentation_examples.png."""
    image_features = pd.read_csv(config.IMAGE_FEATURES_CSV, index_col="isic_id")
    audit = pd.read_csv(AUDIT_CSV).join(image_features[["shp_border_touch", "shp_fallback"]], on="isic_id")
    audit["view"] = np.where(audit["image_type"] == "dermoscopic", "dermoscopic", "clinical")
    counts = pd.crosstab([audit["diagnosis_1"], audit["view"]], audit["verdict"].map(VERDICTS))
    counts = counts.reindex(columns=list(VERDICTS.values()), fill_value=0)
    bad = audit["verdict"].isin(["F", "B"])
    flag = suspect_mask(audit)
    check = {"images": len(audit), "bad_pct": 100 * bad.mean(), "agreement": (flag == bad).mean(),
             "sensitivity": flag[bad].mean(), "specificity": (~flag[~bad]).mean()}
    print(counts.to_string())
    print({k: round(float(v), 3) for k, v in check.items()})

    images = load_images().set_index("isic_id").join(image_features[["shp_border_touch", "shp_fallback"]])
    images["view"] = np.where(images["image_type"] == "dermoscopic", "dermoscopic", "clinical")
    images["suspect"] = suspect_mask(images)
    by_class = (100 * images.groupby(["diagnosis_1", "view"])["suspect"].mean()).unstack().round(1)
    by_dx = (100 * images.groupby("dx")["suspect"].mean()).round(1).sort_values()
    print(by_class.to_string())
    write_json(OUT / "segmentation_audit.json",
               {"manual": check, "flag_rule": f"border share > {SUSPECT_BORDER} or fallback",
                "flagged_pct_all_images": 100 * float(images["suspect"].mean()),
                "flagged_pct_by_class_view": by_class.to_dict(), "flagged_pct_by_dx": by_dx.to_dict()})
    write(OUT / "segmentation_audit.md",
          "# Lesion-mask audit\n\n"
          f"Manual verdicts on {len(audit)} images (20 lesions per class, both views):\n\n"
          + md_table(counts, index=True) + "\n\n"
          f"Automatic flag (mask on more than {SUSPECT_BORDER:.0%} of the image border, or fallback ellipse) "
          f"against the manual verdicts (bad = failed or fallback): agreement {check['agreement']:.3f}, "
          f"sensitivity {check['sensitivity']:.3f}, specificity {check['specificity']:.3f}.\n\n"
          f"Flagged share of all {len(images):,} images: {100 * images['suspect'].mean():.1f}%. By class and view (%):\n\n"
          + md_table(by_class, index=True, floats=1) + "\n\nBy 11-class label (%):\n\n"
          + md_table(by_dx.rename("flagged_pct").to_frame(), index=True, floats=1) + "\n")


STEPS = {"features": step_features, "audit": step_audit, "compare": step_compare, "final": step_final,
         "stretch": step_stretch}


def main(argv):
    wanted = argv or list(STEPS)
    unknown = [w for w in wanted if w not in STEPS]
    if unknown:
        raise SystemExit(f"unknown step(s) {unknown}; choose from {list(STEPS)}")
    with Tee(OUT / "log.txt"):
        print(f"\n===== classical_baseline.py {' '.join(wanted)}  ({date.today()}) =====")
        for name in wanted:
            print(f"\n--- {name} ---")
            start = time.time()
            STEPS[name]()
            print(f"[{name}: {time.time() - start:.0f} s]")


if __name__ == "__main__":
    main(sys.argv[1:])
