"""Milestone 5: the 11-class model for the official leaderboard (Path A of the project brief).

    python m5_final/train_11class.py        (about 30 minutes on a 2-core CPU)

Our analysis (scoreboard, errors, fairness) used 3 classes. The leaderboard scores 11. So the best
architecture of Milestone 4 is trained once more with 11 outputs. Compared with the 3-class model, only two
things change:

    the labels      dx, the 11 classes of training_gt.csv, in its column order (= the submission's order)
    the last layer  11 outputs instead of 3

Same split, same recipe (shared.deep.fit: 6 epochs, learning rate 1e-4, flips, best validation epoch).

The leaderboard does not take the most likely class. It turns each of the 11 probabilities of a lesion into
yes / no at 0.5 and averages the F1 of the 11 classes. So this script reports the validation macro-F1 in
that same way, and at the very end scores our own test split, once.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))        # the repo root, so that `shared` can be imported

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torchvision.models import ResNet18_Weights, resnet18

from shared import config
from shared.deep import dermoscopic, fit, load_images, macro_f1, predict_proba
from shared.labels import load_class_weights

OUT = config.M5_OUT
OUT.mkdir(parents=True, exist_ok=True)
CLASSES = config.SUBCLASSES                                          # AKIEC, BCC, BEN_OTH, ... in the official order
THRESHOLD = 0.5                                                      # the leaderboard's cut

# ---------------------------------------------------------------------------------------------------
# 0. Data: the same lesions, now with their 11-class label

X = load_images()
tables = {name: dermoscopic(name) for name in X}
to_int = {c: k for k, c in enumerate(CLASSES)}                       # {"AKIEC": 0, "BCC": 1, ...}
y = {name: torch.tensor(tables[name][config.STRETCH_LABEL].map(to_int).to_numpy()) for name in X}

# Class weights n / (11 * n_c) from the train split (class_weights.json, Milestone 1).
# MAL_OTH has only 5 training lesions, so its weight is 66.7: one such lesion would count more than the
# rest of its batch together. We cap it at the weight of the next rarest class (BEN_OTH, 10.75).
class_weights = load_class_weights(config.STRETCH_LABEL)
class_weights = class_weights.clamp(max=float(class_weights.sort().values[-2]))
print(dict(zip(CLASSES, class_weights.round(decimals=2).tolist())))

# ---------------------------------------------------------------------------------------------------
# 1. Same network, 11 outputs
# HOW: identical to the 3-class model of compare_models.py, except the size of the last layer.

torch.manual_seed(config.SEED)
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)             # pretrained on ImageNet
model.fc = nn.Linear(512, len(CLASSES))                              # new last layer: 512 features -> 11 classes
history, best = fit(model, list(model.parameters()), X["train"], y["train"], X["val"], y["val"], class_weights,
                    epochs=config.FINE_TUNE_EPOCHS, lr=config.FINE_TUNE_LR)
history.to_csv(OUT / "history_11class.csv", index=False)
torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "target": config.STRETCH_LABEL}, config.MODEL_11CLASS)

# ---------------------------------------------------------------------------------------------------
# 2. Scores the way the leaderboard computes them
# PURPOSE: a local estimate of the official score.
# HOW: P holds the 11 probabilities of each lesion (softmax, each row adds up to 1). Y is the truth as
#      11 columns of 0 / 1. "P >= 0.5" turns every probability into yes / no, f1_score then computes the
#      F1 of each class (column) and average="macro" takes their plain mean.
#      zero_division=0: a class with no "yes" at all has no precision (0 / 0); its F1 counts as 0.


def leaderboard_scores(split):
    """Score one split the leaderboard's way. Returns (summary, table with one row per class, P, Y)."""
    P = predict_proba(model, X[split]).numpy()                       # probabilities, shape (n, 11)
    Y = np.eye(len(CLASSES), dtype=int)[y[split].numpy()]            # one-hot truth, shape (n, 11)
    yes = P >= THRESHOLD                                             # the leaderboard's yes / no
    per_class = f1_score(Y, yes, average=None, zero_division=0)      # one F1 per class
    summary = {
        "lesions": len(P),
        "macro_f1_argmax": macro_f1(y[split], P.argmax(1)),          # usual way: the most likely class
        "macro_f1_at_0.5": float(f1_score(Y, yes, average="macro", zero_division=0)),   # the leaderboard's way
        "lesions_without_any_yes": float((yes.sum(axis=1) == 0).mean()),   # no probability reaches 0.5
    }
    table = pd.DataFrame({"class": CLASSES, "lesions": Y.sum(axis=0), "predicted yes": yes.sum(axis=0),
                          "F1 at 0.5": per_class})
    probs = pd.DataFrame(P, columns=[f"p_{c}" for c in CLASSES])
    probs.insert(0, "lesion_id", tables[split]["lesion_id"])
    probs.insert(1, "true", tables[split][config.STRETCH_LABEL])
    probs.to_csv(OUT / f"predictions_{split}_11class.csv", index=False)
    return summary, table, P, Y


val, val_table, _, _ = leaderboard_scores("val")
print("validation:", val)

# ---------------------------------------------------------------------------------------------------
# 3. Our own test split, once
# PURPOSE: validation chose the epoch, so its score is a little optimistic. The test lesions were not
#          used for anything: their score is the fairest estimate of the official one.

test, test_table, P, Y = leaderboard_scores("test")

# How much would this score move with other test lesions? Draw the 787 lesions again with replacement
# 2,000 times and keep the middle 95% of the scores (the bootstrap of Session 7's scoreboard).
rng = np.random.default_rng(config.SEED)
draws = []
for _ in range(config.BOOTSTRAP_RESAMPLES):
    idx = rng.integers(0, len(P), len(P))                            # 787 lesions drawn at random, with repeats
    draws.append(f1_score(Y[idx], P[idx] >= THRESHOLD, average="macro", zero_division=0))
test["macro_f1_at_0.5_95_interval"] = [float(v) for v in np.percentile(draws, [2.5, 97.5])]
print("test:", test)

per_class = val_table.merge(test_table, on="class", suffixes=(" (val)", " (test)"))
per_class.insert(1, "train lesions", np.bincount(y["train"].numpy(), minlength=len(CLASSES)))
per_class.to_csv(OUT / "per_class_11class.csv", index=False)
print(per_class.round(3).to_string(index=False))
(OUT / "scores_11class.json").write_text(json.dumps(
    {"model": "ResNet-18 fine-tuned: whole network, 11 outputs", "best_epoch": int(best["epoch"]),
     "threshold": THRESHOLD, "validation": val, "test": test}, indent=2))
