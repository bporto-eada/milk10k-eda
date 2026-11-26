"""Deep-learning helpers of Milestones 4 and 5.

These are the functions of m3_deep/session7_cnns.ipynb (prep, augment, predict, fit, test_interval), moved into one
module so that training, validation, the leaderboard file and predict() all run the same code and, above
all, the same preprocessing.

val_tf          THE inference transform: validation, test, the leaderboard images and predict() all use it
dermoscopic     one row per lesion of a split: its dermoscopic image
load_images     every image of the three splits in memory, decoded once
prep            stored pixels -> network input (the last two steps of val_tf)
augment         random flips, for training batches only
predict_proba   class probabilities, in evaluation mode and without gradients
fit             the training loop of Session 7; keeps the weights of the best validation epoch
macro_f1        the metric of the whole project
test_interval   95% interval of a test score (bootstrap of the test lesions)
"""

import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torchvision import transforms as T

from . import config, paths
from .data import require_images
from .transforms import load_rgb

# ImageNet statistics, one value per colour channel (red, green, blue). The pretrained networks were trained
# on images normalised with these numbers, so they must see our images normalised the same way.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# PURPOSE: the ONE transform an image goes through whenever it is not a training batch.
# HOW: shorter side to 224, then the central 224x224 square (Milestone 1: we do not squeeze the 600x450
#      photo into a square, that would change the shape of the lesion); ToTensor puts the values in [0, 1];
#      Normalize subtracts the ImageNet mean and divides by the ImageNet std.
# Nothing here is random: the same image always gives the same tensor.
val_tf = T.Compose([
    T.Resize(config.IMAGE_SIZE),
    T.CenterCrop(config.IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

# The first two steps of val_tf only. PILToTensor keeps the pixels as uint8 (0-255): 4 times less memory
# than floats, so all 5,240 images fit in RAM. prep() below does the last two steps of val_tf.
to_uint8 = T.Compose([T.Resize(config.IMAGE_SIZE), T.CenterCrop(config.IMAGE_SIZE), T.PILToTensor()])


def dermoscopic(name):
    """A split CSV (one row per image) -> one row per lesion: its dermoscopic image."""
    split = pd.read_csv(config.SPLITS_DIR / f"{name}.csv")       # the split of Milestone 1, never re-made here
    split = split[split["image_type"] == "dermoscopic"].reset_index(drop=True)
    assert split["lesion_id"].is_unique, f"{name}: a lesion has two dermoscopic images"
    return require_images(split)                                 # stops here if an image file is missing


def load_images():
    """{"train": X, "val": X, "test": X} with X a uint8 tensor (n, 3, 224, 224); row i = row i of dermoscopic().

    PURPOSE: make training faster by not reading 5,240 JPEG files again in every epoch.
    HOW: if a saved file for exactly these lesions exists, load it; if not, build it once and save it in
         milk10k/cache/ (that folder is never committed).
    """
    file = config.DATA_DIR / "cache" / f"derm_{config.IMAGE_SIZE}_images.pt"
    tables = {name: dermoscopic(name) for name in ("train", "val", "test")}
    data = torch.load(file) if file.exists() else {}
    if any(data.get(f"ids_{name}") != tables[name]["isic_id"].tolist() for name in tables):
        data = {}                                                # no file yet, or it holds other lesions
        for name, table in tables.items():
            data[f"X_{name}"] = torch.stack([to_uint8(load_rgb(paths.find_image(i))) for i in table["isic_id"]])
            data[f"ids_{name}"] = table["isic_id"].tolist()      # lets the next run check the file against the split
        file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, file)
    return {name: data[f"X_{name}"] for name in tables}


def prep(xb):
    """Stored pixels (uint8, 0-255) -> what the network expects. Same numbers as val_tf gives."""
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)                   # view(1, 3, 1, 1): one value per channel,
    std = torch.tensor(STD).view(1, 3, 1, 1)                     # in a shape that applies to a whole batch
    return (xb.float() / 255 - mean) / std                       # [0, 255] -> [0, 1] -> normalised


def augment(xb):
    """Random flips: each image ends up in one of 4 versions. The labels do not change.

    A lesion has no "up" or "left": a flipped lesion is still the same diagnosis, so flips are safe.
    Used on training batches only. Validation and test images are never augmented.
    """
    n = xb.size(0)                                               # number of images in the batch
    flip_h = torch.rand(n, 1, 1, 1) < 0.5                        # one coin per image: True for half of them
    xb = torch.where(flip_h, xb.flip(3), xb)                     # flip(3) mirrors the width: left-right
    flip_v = torch.rand(n, 1, 1, 1) < 0.5                        # a second, independent coin
    return torch.where(flip_v, xb.flip(2), xb)                   # flip(2) mirrors the height: up-down


@torch.no_grad()                                                 # no gradients: we only predict, nothing is learned
def predict_proba(model, X, batch_size=128):
    """Class probabilities, shape (n, n_classes), for stored images X."""
    model.eval()                                                 # evaluation mode: BatchNorm uses its stored statistics
    logits = torch.cat([model(prep(X[i:i + batch_size])) for i in range(0, len(X), batch_size)])
    return torch.softmax(logits, dim=1)                          # scores -> probabilities that add up to 1


def macro_f1(y_true, y_pred):
    """F1 of each class, then their plain average: a rare class counts as much as a common one.

    zero_division=0: a class that is never predicted has no precision (0 / 0); its F1 then counts as 0.
    """
    return float(f1_score(np.asarray(y_true), np.asarray(y_pred), average="macro", zero_division=0))


def fit(model, params, X_train, y_train, X_val, y_val, class_weights, epochs, lr, batch_size=64):
    """Train the weights `params` of `model`. Returns (history, best); the model ends with its best epoch.

    PURPOSE: one training loop for every network (3 classes and 11 classes).
    HOW: Session 6's loop over batches, class weights in the loss, validation macro-F1 after each epoch,
         and early stopping: the weights of the best validation epoch are the ones we keep.
    """
    optimizer = torch.optim.Adam(params, lr=lr)                  # params: only the weights we want to train
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)          # a mistake on a rare class costs more
    history, best = [], {"f1": -1}
    for epoch in range(epochs):
        model.train()                                            # training mode
        t0, total = time.perf_counter(), 0.0
        order = torch.randperm(len(X_train))                     # a new random order in each epoch
        for i in range(0, len(X_train), batch_size):
            idx = order[i:i + batch_size]                        # the next 64 images
            xb = augment(prep(X_train[idx]))                     # normalise, then random flips
            yb = y_train[idx]

            # The training step from Session 6
            optimizer.zero_grad()                                # forget the gradients of the previous batch
            loss = loss_fn(model(xb), yb)                        # forward pass + loss
            loss.backward()                                      # backpropagation: compute the gradients
            optimizer.step()                                     # update the weights
            total += loss.item() * len(idx)

        # End of the epoch: score on validation and remember the best weights so far
        f1 = macro_f1(y_val, predict_proba(model, X_val).argmax(1))
        seconds = time.perf_counter() - t0
        history.append({"epoch": epoch, "train_loss": total / len(X_train), "val_macro_f1": f1, "seconds": seconds})
        print(f"epoch {epoch}  loss {total / len(X_train):.3f}  val macro-F1 {f1:.3f}  ({seconds:.0f} s)", flush=True)
        if f1 > best["f1"]:
            best = {"f1": f1, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
    model.load_state_dict(best["state"])                         # early stopping: go back to the best epoch
    return pd.DataFrame(history), best


def test_interval(y_true, y_pred, n=config.BOOTSTRAP_RESAMPLES, seed=config.SEED):
    """95% interval of the test macro-F1: how much the score would move with other test lesions.

    HOW: draw the test lesions again with replacement 2,000 times (inside each class, so the class counts
         stay the same), compute macro-F1 each time, and keep the middle 95% of the 2,000 scores.
    """
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y_true), np.asarray(y_pred)
    by_class = [np.flatnonzero(y == k) for k in np.unique(y)]
    draws = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in by_class])
        draws.append(macro_f1(y[idx], p[idx]))
    return np.percentile(draws, [2.5, 97.5])
