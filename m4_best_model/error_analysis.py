"""Milestone 4, part 2: error analysis of the final model on the test lesions.

    python m4_best_model/error_analysis.py        (a few seconds; run compare_models.py first)

It only reads the test predictions that compare_models.py saved. Nothing is trained or chosen here.

  1. Confusion matrix, and precision / recall / F1 of each class.
  2. Which diagnoses (the 11 classes) are behind the mistakes.
  3. The misclassified images themselves: the most confident mistakes in both directions.
  4. What the mistakes have in common: pigmentation, and whether the lesion has a clear outline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))        # the repo root, so that `shared` can be imported

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from shared import config, paths
from shared.deep import dermoscopic, to_uint8
from shared.labels import PRIMARY_CLASSES as CLASSES                 # ["Benign", "Indeterminate", "Malignant"]
from shared.transforms import load_rgb

OUT = config.M4_OUT
FIG = OUT / "figures"

# ---------------------------------------------------------------------------------------------------
# 0. The test predictions, with what we know about each lesion
# HOW: three tables joined on lesion_id / isic_id. validate="one_to_one" stops the script if a lesion
#      appears twice, which would silently duplicate rows.

pred = pd.read_csv(OUT / "predictions_test.csv")                     # lesion_id, true, pred, p_Benign, ...
info = dermoscopic("test")[["lesion_id", "isic_id", "dx"]]           # dx = the 11-class diagnosis
extra = pd.read_csv(config.SUPPLEMENT_INPUT_CSV)                     # published with MILK10k: one row per image
extra = extra[extra["image_type"] == "dermoscopic"][["lesion_id", "MONET_pigmented"]]
mask = pd.read_csv(config.IMAGE_FEATURES_CSV, usecols=["isic_id", "shp_border_touch", "shp_fallback"])   # Session 5

t = (pred.merge(info, on="lesion_id", validate="one_to_one")
         .merge(extra, on="lesion_id", validate="one_to_one")
         .merge(mask, on="isic_id", validate="one_to_one"))
t["wrong"] = t["pred"] != t["true"]
t["confidence"] = t[[f"p_{c}" for c in CLASSES]].max(axis=1)         # the probability of the predicted class
print(f"{t['wrong'].sum()} of {len(t)} test lesions misclassified ({t['wrong'].mean():.1%})")

# ---------------------------------------------------------------------------------------------------
# 1. Confusion matrix and the scores of each class
# HOW: rows = the truth, columns = the prediction. Precision: of the lesions called X, how many are X.
#      Recall: of the lesions that are X, how many were called X. F1: their harmonic mean.

cm = confusion_matrix(t["true"], t["pred"], labels=CLASSES)
pd.DataFrame(cm, index=[f"true {c}" for c in CLASSES], columns=[f"pred {c}" for c in CLASSES]).to_csv(
    OUT / "confusion_matrix_test.csv")
precision, recall, f1, n = precision_recall_fscore_support(t["true"], t["pred"], labels=CLASSES, zero_division=0)
per_class = pd.DataFrame({"class": CLASSES, "test lesions": n, "precision": precision, "recall": recall, "F1": f1})
per_class.to_csv(OUT / "per_class_test.csv", index=False)
print(per_class.round(3).to_string(index=False))

share = cm / cm.sum(axis=1, keepdims=True)                           # each row as a share of its true class
fig, ax = plt.subplots(figsize=(4.2, 3.6))
ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        ax.text(j, i, f"{cm[i, j]}\n{share[i, j]:.0%}", ha="center", va="center", fontsize=9,
                color="white" if share[i, j] > 0.55 else "black")
ax.set_xticks(range(len(CLASSES)), CLASSES, fontsize=8)
ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=8, rotation=90, va="center")
ax.set_xlabel("predicted")
ax.set_ylabel("true")
ax.set_title(f"final model, {len(t)} test lesions", fontsize=9)
fig.savefig(FIG / "confusion_matrix_test.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------------------------------
# 2. Which diagnoses are behind the mistakes
# HOW: every test lesion also has its 11-class diagnosis (dx); count the answers the model gave for each.

answers = pd.crosstab(t["dx"], t["pred"]).reindex(columns=CLASSES, fill_value=0)       # diagnosis x answer
answers.columns = [f"called {c}" for c in CLASSES]
by_dx = t.groupby("dx")["wrong"].agg(lesions="size", wrong="sum", error_rate="mean").join(answers)
by_dx = by_dx.sort_values("wrong", ascending=False).reset_index()
by_dx.to_csv(OUT / "errors_by_diagnosis.csv", index=False)
print(by_dx.round(3).to_string(index=False))

# ---------------------------------------------------------------------------------------------------
# 3. Open the misclassified images
# PURPOSE: look at real mistakes, not only at numbers.
# HOW: the two big confusions are Malignant called Benign and Benign called Malignant. Take the 5 most
#      confident mistakes of each and show the 224x224 crop the network received.

missed = t[(t["true"] == "Malignant") & (t["pred"] == "Benign")].nlargest(5, "confidence")
alarms = t[(t["true"] == "Benign") & (t["pred"] == "Malignant")].nlargest(5, "confidence")
gallery = pd.concat([missed, alarms], ignore_index=True)
gallery[["lesion_id", "isic_id", "dx", "true", "pred", "confidence", "MONET_pigmented"]].to_csv(
    OUT / "misclassified_gallery.csv", index=False)

fig, axes = plt.subplots(2, 5, figsize=(9.5, 4.5))
for k, (ax, row) in enumerate(zip(axes.ravel(), gallery.itertuples())):
    image = to_uint8(load_rgb(paths.find_image(row.isic_id)))        # the same crop as in training and prediction
    ax.imshow(image.permute(1, 2, 0).numpy())                        # (3, H, W) -> (H, W, 3) for matplotlib
    ax.set_title(f"{k + 1}. {row.dx} ({row.true})\ncalled {row.pred}, {row.confidence:.2f}", fontsize=8.5)
    ax.axis("off")
fig.subplots_adjust(wspace=0.06, hspace=0.3)
fig.savefig(FIG / "misclassified_gallery.png", dpi=170, bbox_inches="tight")
plt.close(fig)

# Is the model less sure when it is wrong? On average yes, but its most confident mistakes are very confident.
mean_confidence = t.groupby("wrong")["confidence"].mean()            # index False = correct answers, True = mistakes
sure_and_wrong = int((t["wrong"] & (t["confidence"] >= 0.95)).sum())
confidence_line = (f"Mean confidence: {mean_confidence[False]:.2f} on correct answers, {mean_confidence[True]:.2f} on "
                   f"mistakes. Mistakes made with a confidence of 0.95 or more: {sure_and_wrong}.")
print(confidence_line)

# ---------------------------------------------------------------------------------------------------
# 4. What the mistakes have in common
# PURPOSE: test with numbers what the gallery suggests: brown lesions with a clear outline are called
#          Benign, pink lesions without an outline are called Malignant.
# HOW: (a) MONET_pigmented is a 0-1 "how pigmented is this image" score that comes with MILK10k: split the
#          test lesions at its median and compare the error rates inside each true class.
#      (b) Session 5's Otsu mask fails when there is no dark, sharply bounded lesion to outline (the mask
#          touches more than 5% of the image border, or the fallback ellipse was used). Compare the error
#          rates where the mask failed and where it worked.

t["pigmentation"] = np.where(t["MONET_pigmented"] >= t["MONET_pigmented"].median(), "more pigmented half",
                             "less pigmented half")
t["otsu mask"] = np.where((t["shp_border_touch"] > 0.05) | (t["shp_fallback"] == 1), "mask failed", "mask plausible")
two_classes = t[t["true"] != "Indeterminate"]                        # 18 Indeterminate lesions: too few to split
tables = {}
for column in ("pigmentation", "otsu mask"):
    tables[column] = (two_classes.groupby(["true", column])["wrong"]
                      .agg(lesions="size", wrong="sum", error_rate="mean").reset_index())
    tables[column].to_csv(OUT / f"errors_by_{column.replace(' ', '_')}.csv", index=False)
    print(tables[column].round(3).to_string(index=False))


def markdown(frame):
    """A DataFrame as a Markdown table, numbers with 3 decimals."""
    text = frame.copy()
    for column in text.columns:
        if pd.api.types.is_float_dtype(text[column]):
            text[column] = text[column].map("{:.3f}".format)
    lines = ["| " + " | ".join(map(str, text.columns)) + " |", "|---" * len(text.columns) + "|"]
    return "\n".join(lines + ["| " + " | ".join(map(str, row)) + " |" for row in text.itertuples(index=False)])


(OUT / "error_analysis.md").write_text("\n\n".join([
    "# Error analysis of the final model (written by m4_best_model/error_analysis.py)\n\n"
    f"{len(t)} test lesions, {t['wrong'].sum()} misclassified ({t['wrong'].mean():.1%}).",
    "## Confusion matrix (rows: truth)\n\n" + markdown(pd.DataFrame(cm, columns=[f"pred {c}" for c in CLASSES])
                                                       .assign(true=CLASSES).set_index("true").reset_index()),
    "## Per class\n\n" + markdown(per_class),
    "## Mistakes by 11-class diagnosis\n\n" + markdown(by_dx),
    "## The 10 images of figures/misclassified_gallery.png\n\n" + markdown(gallery[["isic_id", "dx", "true", "pred",
                                                                                   "confidence", "MONET_pigmented"]])
    + "\n\n" + confidence_line,
    "## Error rate by pigmentation, inside each true class\n\n" + markdown(tables["pigmentation"]),
    "## Error rate by Otsu mask, inside each true class\n\n" + markdown(tables["otsu mask"]),
]) + "\n")
