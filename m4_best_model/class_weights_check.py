"""Milestone 4, two small checks on the linear head of compare_models.py.

    python m4_best_model/class_weights_check.py        (about a minute; run compare_models.py first)

  1. Does the class-weighted loss help? The same linear layer is trained on the saved ResNet-18 features
     twice: once WITH the class weights in the loss, once WITHOUT. Nothing else changes.
  2. How much does a score move when only the random seed changes? The same head is trained with five
     seeds, on the ResNet-18 features and on the ViT-B/16 features, with and without class weights.
     One run is one sample: a difference smaller than the spread between seeds can be noise.

Scored on VALIDATION only. These are questions about how to train, so the test set is not opened for them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))        # the repo root, so that `shared` can be imported

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import recall_score
from torch.utils.data import DataLoader, TensorDataset

from shared import config
from shared.deep import dermoscopic, macro_f1
from shared.labels import load_class_weights, load_label_map

# ---------------------------------------------------------------------------------------------------
# 0. The saved features and the labels
# HOW: compare_models.py saved the features of the two frozen backbones: 512 numbers per lesion for
#      ResNet-18, 768 for ViT-B/16. Each file holds {"train": ..., "val": ..., "test": ...}.

cache = config.DATA_DIR / "cache"
FEATURES = {"ResNet-18": torch.load(cache / "features_resnet18.pt"),
            "ViT-B/16": torch.load(cache / "features_vit_b_16.pt")}
to_int = load_label_map()[config.LABEL]                              # {"Benign": 0, "Indeterminate": 1, "Malignant": 2}
CLASSES = sorted(to_int, key=to_int.get)
y = {name: torch.tensor(dermoscopic(name)[config.LABEL].map(to_int).to_numpy()) for name in ("train", "val")}
class_weights = load_class_weights(config.LABEL)                     # n / (3 * n_c) on train: [1.18, 14.05, 0.48]
LOSSES = [("with class weights", class_weights), ("without class weights", None)]


def train_head(F, weight, seed):
    """The head of compare_models.py. `weight` (the loss weights) and `seed` are the only things that change."""
    torch.manual_seed(seed)                                          # fixes the starting weights and the batch order
    head = nn.Linear(F["train"].shape[1], len(CLASSES))              # 512 or 768 features -> 3 classes
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(weight=weight)                     # weight=None: every lesion counts the same
    loader = DataLoader(TensorDataset(F["train"], y["train"]), batch_size=64, shuffle=True)
    for epoch in range(25):
        for fb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(head(fb), yb)
            loss.backward()
            optimizer.step()
    return head


def predict_val(head, F):
    """The class the head predicts for each validation lesion."""
    with torch.no_grad():
        return head(F["val"]).argmax(1)


# ---------------------------------------------------------------------------------------------------
# 1. With and without class weights (ResNet-18 features, the seed of the project)
# HOW: recall of a class = of the lesions that truly are that class, the share the model found.

rows = []
for name, weight in LOSSES:
    pred = predict_val(train_head(FEATURES["ResNet-18"], weight, config.SEED), FEATURES["ResNet-18"])
    recall = recall_score(y["val"], pred, average=None, labels=[0, 1, 2], zero_division=0)
    row = {"loss": name, "val macro-F1": macro_f1(y["val"], pred)}
    for k, c in enumerate(CLASSES):
        row[f"recall {c}"] = recall[k]
    called = pred == to_int["Indeterminate"]                         # the lesions the head calls Indeterminate
    row["lesions called Indeterminate"] = int(called.sum())          # 18 validation lesions truly are
    row["of which right"] = int((called & (y["val"] == to_int["Indeterminate"])).sum())
    rows.append(row)

table = pd.DataFrame(rows)
table.to_csv(config.M4_OUT / "class_weights_check.csv", index=False)
print(table.round(3).to_string(index=False))

# ---------------------------------------------------------------------------------------------------
# 2. Five seeds, both backbones, both losses
# PURPOSE: see how much of a difference between two scores is just the random seed.
# HOW: train the same head five times; only the seed changes. Report the mean, the lowest and the highest
#      validation macro-F1 of the five runs.

SEEDS = [config.SEED, 0, 1, 2, 3]                                    # 42 is the seed of every other script
rows = []
for backbone, F in FEATURES.items():
    for name, weight in LOSSES:
        scores = [macro_f1(y["val"], predict_val(train_head(F, weight, seed), F)) for seed in SEEDS]
        rows.append({"backbone (frozen)": backbone, "loss": name, "mean of 5 seeds": np.mean(scores),
                     "lowest": min(scores), "highest": max(scores),
                     "seeds 42, 0, 1, 2, 3": " ".join(f"{s:.3f}" for s in scores)})

seeds = pd.DataFrame(rows)
seeds.to_csv(config.M4_OUT / "class_weights_check_seeds.csv", index=False)
print(seeds.round(3).to_string(index=False))
