"""Milestone 4, part 1: a ViT next to our CNN, the scoreboard, and the best deep model.

    python m4_best_model/compare_models.py        (about 55 minutes on a 2-core CPU, no GPU needed)

Same lesions (the split of Milestone 1), same target (diagnosis_1: Benign / Indeterminate / Malignant) and
same metric (macro-F1 per lesion) as every model since Milestone 2.

  1. Train the best CNN of Session 7 again and save its weights. Session 7 did not save them, and
     predict() needs them.
  2. Compare ViT-B/16 with ResNet-18 the fair way of Session 9: both frozen, the same new last layer, the
     same training. Only the backbone changes.
  3. One scoreboard with every model, Milestones 2 to 4.
  4. The deep model with the best VALIDATION macro-F1 is the final model. Its TEST score is computed once,
     here, and its predictions are saved for the error analysis and the fairness check.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))        # the repo root, so that `shared` can be imported

import matplotlib
matplotlib.use("Agg")                                                # draw to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import ResNet18_Weights, ViT_B_16_Weights, resnet18, vit_b_16

from shared import config
from shared.deep import dermoscopic, fit, load_images, macro_f1, predict_proba, prep, test_interval
from shared.labels import load_class_weights, load_label_map

OUT = config.M4_OUT
(OUT / "figures").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------------------------------
# 0. Data
# PURPOSE: the same lesions as every earlier model.
# HOW: the split files of Milestone 1, one dermoscopic image per lesion, labels as integers.

X = load_images()                                                    # uint8 tensors (n, 3, 224, 224)
tables = {name: dermoscopic(name) for name in X}                     # one row per lesion, same order as X
to_int = load_label_map()[config.LABEL]                              # {"Benign": 0, "Indeterminate": 1, "Malignant": 2}
CLASSES = sorted(to_int, key=to_int.get)
y = {name: torch.tensor(tables[name][config.LABEL].map(to_int).to_numpy()) for name in X}
class_weights = load_class_weights(config.LABEL)                     # n / (3 * n_c) on train: [1.18, 14.05, 0.48]
print({name: len(y[name]) for name in y})                            # {'train': 3667, 'val': 786, 'test': 787}

# ---------------------------------------------------------------------------------------------------
# 1. The best CNN of Session 7, trained again and saved
# PURPOSE: have the weights of our best CNN in a file.
# HOW: exactly Session 7's strategy 3: ResNet-18 with ImageNet weights, a new last layer, every layer
#      trained, 6 epochs, learning rate 1e-4, the best validation epoch kept (early stopping).

BEST_CNN = "ResNet-18 fine-tuned: whole network"                     # its name on the Session 7 scoreboard

torch.manual_seed(config.SEED)                                       # same seed as Session 7 -> same model
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)             # pretrained on ImageNet (1.2 million photos)
model.fc = nn.Linear(512, len(CLASSES))                              # new last layer: 512 features -> 3 classes
history, best = fit(model, list(model.parameters()),                 # list(model.parameters()): train every layer
                    X["train"], y["train"], X["val"], y["val"], class_weights,
                    epochs=config.FINE_TUNE_EPOCHS, lr=config.FINE_TUNE_LR)
history.to_csv(OUT / "history_resnet18.csv", index=False)

# The file holds the weights and the class names, so predict() needs nothing else
torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "target": config.LABEL}, config.MODEL_M4)

# Learning curves: the training loss keeps falling while validation goes up and down -> overfitting
fig, ax = plt.subplots(figsize=(4.6, 2.9))
ax.plot(history["epoch"] + 1, history["train_loss"], marker="o", color="#c8553d", label="training loss")
ax.plot(history["epoch"] + 1, history["val_macro_f1"], marker="o", color="#2f5d8a", label="validation macro-F1")
ax.axvline(best["epoch"] + 1, color="grey", ls="--", lw=1)           # the epoch that early stopping kept
ax.set_xlabel("epoch")
ax.legend(fontsize=7, frameon=False)
ax.set_title("ResNet-18 fine-tuned: dashed line = the epoch that was kept", fontsize=8)
fig.savefig(OUT / "figures" / "learning_curves.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------------------------------
# 2. ViT-B/16 against ResNet-18: both frozen, the same new last layer (Session 9)
# PURPOSE: a fair comparison of the two backbones. Only the backbone changes.
# HOW: remove the head of each pretrained network, so it returns its features (512 numbers for ResNet-18,
#      768 for ViT-B/16); compute them once for every image; train the same linear layer on each.
#      Why frozen: retraining the 86 million weights of a ViT on a CPU would take hours.


@torch.no_grad()                                                     # frozen: no gradients are needed
def embed(backbone, images, batch_size=128):
    """Images -> one row of features per image."""
    backbone.eval()
    return torch.cat([backbone(prep(images[i:i + batch_size])) for i in range(0, len(images), batch_size)])


def features(name, backbone):
    """The features of the three splits. The slow step (15 to 20 minutes for the ViT), so it runs once and is saved."""
    file = config.DATA_DIR / "cache" / f"features_{name}.pt"         # milk10k/cache/ is never committed
    if not file.exists():
        torch.save({split: embed(backbone, X[split]) for split in X}, file)
    return torch.load(file)


def train_head(F, epochs=25, lr=1e-3):
    """A new last layer trained on frozen features. Session 7's recipe, identical for both backbones."""
    torch.manual_seed(config.SEED)
    head = nn.Linear(F["train"].shape[1], len(CLASSES))              # 512 or 768 features -> 3 classes
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)           # only the head is trained
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    loader = DataLoader(TensorDataset(F["train"], y["train"]), batch_size=64, shuffle=True)
    for epoch in range(epochs):
        for fb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(head(fb), yb)
            loss.backward()
            optimizer.step()
    return head


resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
resnet.fc = nn.Identity()                                            # no head: the network returns 512 features
vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
vit.heads = nn.Identity()                                            # no head: it returns the 768 numbers of its class token

FROZEN_CNN, FROZEN_VIT = "ResNet-18 frozen + linear head", "ViT-B/16 frozen + linear head"
frozen = {}
for name, key, backbone in [(FROZEN_CNN, "resnet18", resnet), (FROZEN_VIT, "vit_b_16", vit)]:
    F = features(key, backbone)                                      # saved as features_resnet18.pt / features_vit_b_16.pt
    head = train_head(F)
    with torch.no_grad():
        frozen[name] = {split: head(F[split]).argmax(1) for split in ("val", "test")}
    frozen[name]["weights"] = sum(p.numel() for p in head.parameters())
    print(f"{name}: {F['train'].shape[1]} features, val macro-F1 {macro_f1(y['val'], frozen[name]['val']):.3f}")

# ---------------------------------------------------------------------------------------------------
# 3. One scoreboard: Milestones 2 to 4
# PURPOSE: every model on the same validation and test lesions, with the same metric.
# HOW: the rows of Milestones 2 and 3 exactly as Session 7 reported them, plus the ViT row of today.
#      The two ResNet-18 rows computed again today are printed next to Session 7's as a check.

probs = {split: predict_proba(model, X[split]) for split in ("val", "test")}      # the fine-tuned CNN of step 1
pred = {split: probs[split].argmax(1) for split in probs}

board = pd.read_csv(config.OUTPUT_DIR / "session7" / "scoreboard.csv").drop(columns="rank")
board.insert(0, "milestone", board.pop("session").astype(str).map({"5": "M2", "6": "M3", "7": "M3", "ref": "ref"}))
for name, val_pred, test_pred in [(BEST_CNN, pred["val"], pred["test"]),
                                  (FROZEN_CNN, frozen[FROZEN_CNN]["val"], frozen[FROZEN_CNN]["test"])]:
    then = board[board["model"] == name].iloc[0]
    print(f"{name}: today val {macro_f1(y['val'], val_pred):.3f} test {macro_f1(y['test'], test_pred):.3f} | "
          f"Session 7 val {then['val macro-F1']:.3f} test {then['test macro-F1']:.3f}")

low, high = test_interval(y["test"], frozen[FROZEN_VIT]["test"])
vit_row = {"milestone": "M4", "model": FROZEN_VIT, "input": "derm image, 224x224",
           "trainable weights": frozen[FROZEN_VIT]["weights"],
           "val macro-F1": macro_f1(y["val"], frozen[FROZEN_VIT]["val"]),
           "test macro-F1": macro_f1(y["test"], frozen[FROZEN_VIT]["test"]),
           "test 95% CI": f"{low:.3f} to {high:.3f}"}
board = pd.concat([board, pd.DataFrame([vit_row])], ignore_index=True)
board = board.sort_values("val macro-F1", ascending=False).reset_index(drop=True)

# ---------------------------------------------------------------------------------------------------
# 4. The best deep model, chosen on validation, scored on test once
# PURPOSE: one final model, chosen without looking at the test set.
# HOW: the network with the highest validation macro-F1. Its test score is only reported, never used to choose.

networks = board[board["model"].str.contains("ResNet|ViT|CNN")]      # the deep models (rows are sorted by validation)
best_name = networks.iloc[0]["model"]
assert best_name == BEST_CNN, f"{best_name} is now the best on validation: save that model instead"
board["final"] = (board["model"] == best_name).map({True: "best -> final", False: ""})

low, high = test_interval(y["test"], pred["test"])
result = {"model": best_name, "chosen_by": "highest validation macro-F1 among the deep models",
          "best_epoch": int(best["epoch"]), "val_macro_f1": macro_f1(y["val"], pred["val"]),
          "test_macro_f1": macro_f1(y["test"], pred["test"]), "test_95_ci": [float(low), float(high)],
          "test_accuracy": float((pred["test"] == y["test"]).float().mean())}
(OUT / "best_model.json").write_text(json.dumps(result, indent=2))
print(result)

# Save the 3-class predictions of the final model: one row per lesion, with the probability of each class
for split in ("val", "test"):
    table = pd.DataFrame(probs[split].numpy(), columns=[f"p_{c}" for c in CLASSES])
    table.insert(0, "lesion_id", tables[split]["lesion_id"])
    table.insert(1, "true", [CLASSES[k] for k in y[split]])
    table.insert(2, "pred", [CLASSES[k] for k in pred[split]])
    table.to_csv(OUT / f"predictions_{split}.csv", index=False)

# The scoreboard as CSV and as a Markdown table that GitHub shows
board.to_csv(OUT / "scoreboard.csv", index=False)
text = board.copy()
text["trainable weights"] = text["trainable weights"].map(lambda v: "" if pd.isna(v) else f"{v:,.0f}")
for column in ("val macro-F1", "test macro-F1"):
    text[column] = text[column].map("{:.3f}".format)
lines = ["| " + " | ".join(text.columns) + " |", "|---" * len(text.columns) + "|"]
lines += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in text.itertuples(index=False)]
(OUT / "scoreboard.md").write_text(
    "# Scoreboard, Milestones 2 to 4 (written by m4_best_model/compare_models.py)\n\n"
    f"Macro-F1 per lesion on the same {len(y['val'])} validation and {len(y['test'])} test lesions, target "
    "diagnosis_1 (3 classes). M2 and M3 rows as reported in Session 7; the M4 row is new. Sorted by validation, "
    "which is what every choice was made on. Each test score was computed once. 95% CI: bootstrap of the test "
    "lesions.\n\n" + "\n".join(lines) + "\n")
print(text.to_string(index=False))
