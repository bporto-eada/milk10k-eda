"""Milestone 4, part 3: fairness check of the final model on the test lesions.

    python m4_best_model/fairness.py        (a few seconds; run compare_models.py first)

Question: does the model work equally well for every skin tone and for both sexes?
It only reads the saved test predictions. Nothing is trained or chosen here.

Skin tone is MILK10k's skin_tone_class: 0 = very dark ... 5 = very light.

What to be careful about (and how this script handles it):
  * small groups: every row shows its number of lesions, and groups under 50 lesions are flagged;
  * macro-F1 averages the classes that occur in a group, as truth or as prediction. Skin tones 2 and 5
    have no Indeterminate lesion, but the model says "Indeterminate" for a few of their lesions: that
    class then scores 0 and pulls the group's macro-F1 down. Such rows are flagged. The recall of
    Malignant and of Benign does not have that problem: it is the number to compare between groups.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))        # the repo root, so that `shared` can be imported

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from shared import config
from shared.deep import dermoscopic
from shared.labels import PRIMARY_CLASSES as CLASSES                 # ["Benign", "Indeterminate", "Malignant"]

OUT = config.M4_OUT
SMALL = 50                                                           # fewer lesions than this: do not interpret

# ---------------------------------------------------------------------------------------------------
# 0. Test predictions + the group of each lesion

pred = pd.read_csv(OUT / "predictions_test.csv")                     # lesion_id, true, pred, ...
info = dermoscopic("test")[["lesion_id", "sex", "dx"]]
tone = pd.read_csv(config.SUPPLEMENT_INPUT_CSV)                      # skin_tone_class comes with MILK10k
tone = tone[tone["image_type"] == "dermoscopic"][["lesion_id", "skin_tone_class"]]
t = pred.merge(info, on="lesion_id", validate="one_to_one").merge(tone, on="lesion_id", validate="one_to_one")
t["tone group"] = np.where(t["skin_tone_class"] <= 2, "darker (0 to 2)", "lighter (3 to 5)")


def recall_of(part, label):
    """Of the lesions of `part` that truly are `label`: (how many the model found, how many there are)."""
    truly = part[part["true"] == label]
    return int((truly["pred"] == label).sum()), len(truly)


def interval(found, total, n=config.BOOTSTRAP_RESAMPLES):
    """95% interval of a recall: draw the `total` lesions again with replacement, 2,000 times."""
    if total == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(config.SEED)
    hits = np.array([1] * found + [0] * (total - found))             # 1 = found, 0 = missed
    draws = [rng.choice(hits, size=total, replace=True).mean() for _ in range(n)]
    return np.percentile(draws, [2.5, 97.5])


# ---------------------------------------------------------------------------------------------------
# 1. One row per group
# HOW: the loop of the course (group, size, macro-F1), plus the size of each class and the two recalls.

rows = []
for column in ("skin_tone_class", "tone group", "sex"):
    for group, part in t.groupby(column):
        counts = [int((part["true"] == c).sum()) for c in CLASSES]   # lesions of each class in this group
        found_m, n_m = recall_of(part, "Malignant")
        found_b, n_b = recall_of(part, "Benign")
        found_i, n_i = recall_of(part, "Indeterminate")
        low_m, high_m = interval(found_m, n_m)
        low_b, high_b = interval(found_b, n_b)
        notes = []
        if len(part) < SMALL:
            notes.append("fewer than 50 lesions")
        if 0 in counts:                                              # e.g. no Indeterminate lesion in the group
            notes.append("a class has no lesion here: macro-F1 not comparable")
        rows.append({
            "attribute": column, "group": group, "lesions": len(part),
            "Benign / Indeterminate / Malignant": " / ".join(map(str, counts)),
            "macro-F1": f1_score(part["true"], part["pred"], average="macro", zero_division=0),
            "Malignant found": f"{found_m} of {n_m}",
            "recall Malignant": found_m / n_m if n_m else np.nan,
            "recall Malignant 95% low": low_m, "recall Malignant 95% high": high_m,
            "Benign found": f"{found_b} of {n_b}",
            "recall Benign": found_b / n_b if n_b else np.nan,
            "recall Benign 95% low": low_b, "recall Benign 95% high": high_b,
            "Indeterminate found": f"{found_i} of {n_i}",
            "called Indeterminate": int((part["pred"] == "Indeterminate").sum()),    # right or wrong
            "note": "; ".join(notes),
        })
table = pd.DataFrame(rows)
table.to_csv(OUT / "fairness.csv", index=False)
print(table.round(3).to_string(index=False))

# ---------------------------------------------------------------------------------------------------
# 2. Why is the recall of Malignant lower on darker skin?
# PURPOSE: two explanations are possible, and they are not the same problem.
#          (a) the darker group holds other diagnoses (melanoma is missed far more often than BCC);
#          (b) the same diagnosis is found less often on darker skin.
# HOW: for the malignant lesions, count per diagnosis and tone group how many were found.

malignant = t[t["true"] == "Malignant"]
by_dx = (malignant.assign(found=malignant["pred"] == "Malignant")
         .groupby(["tone group", "dx"])["found"].agg(lesions="size", found="sum").reset_index())
by_dx["share of the group's malignant lesions"] = by_dx["lesions"] / by_dx.groupby("tone group")["lesions"].transform("sum")
by_dx["recall"] = by_dx["found"] / by_dx["lesions"]
by_dx.to_csv(OUT / "fairness_malignant_by_diagnosis.csv", index=False)
print(by_dx.round(3).to_string(index=False))

# ---------------------------------------------------------------------------------------------------
# 3. Figure: recall of Malignant per group

groups = table[table["attribute"].isin(["skin_tone_class", "sex"])].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(6.6, 3.0))
colours = ["#b9c4d0" if n < SMALL else "#2f5d8a" for n in groups["lesions"]]       # pale = small group
ax.bar(range(len(groups)), groups["recall Malignant"], color=colours, width=0.65)
ax.errorbar(range(len(groups)), groups["recall Malignant"],
            yerr=[groups["recall Malignant"] - groups["recall Malignant 95% low"],
                  groups["recall Malignant 95% high"] - groups["recall Malignant"]],
            fmt="none", ecolor="black", capsize=3, lw=1)
ax.set_xticks(range(len(groups)), [f"{g}\n{m}" for g, m in zip(groups["group"], groups["Malignant found"])], fontsize=7)
ax.set_ylabel("recall of Malignant (test)", fontsize=8)
ax.set_ylim(0, 1.05)
ax.set_title("skin tone 0 (very dark) to 5 (very light), then sex; under each bar: malignant lesions found;\n"
             "pale bars: groups with fewer than 50 lesions; lines: 95% bootstrap interval", fontsize=7.5)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(OUT / "figures" / "fairness.png", dpi=160, bbox_inches="tight")
plt.close(fig)


def markdown(frame):
    """A DataFrame as a Markdown table, numbers with 3 decimals."""
    text = frame.copy()
    for column in text.columns:
        if pd.api.types.is_float_dtype(text[column]):
            text[column] = text[column].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    lines = ["| " + " | ".join(map(str, text.columns)) + " |", "|---" * len(text.columns) + "|"]
    return "\n".join(lines + ["| " + " | ".join(map(str, row)) + " |" for row in text.itertuples(index=False)])


(OUT / "fairness.md").write_text(
    "# Fairness check of the final model (written by m4_best_model/fairness.py)\n\n"
    f"{len(t)} test lesions. Skin tone: MILK10k's skin_tone_class, 0 = very dark to 5 = very light. "
    "macro-F1 averages the classes that occur in a group (as truth or as prediction), so groups with and "
    "without Indeterminate lesions are not comparable on it; the recall of Malignant and of Benign is. "
    "95% interval: 2,000 bootstrap resamples of the group's malignant (or benign) lesions.\n\n" + markdown(table)
    + "\n\n## Malignant lesions by diagnosis and skin-tone group\n\n" + markdown(by_dx) + "\n")
