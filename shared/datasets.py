"""PyTorch datasets and loaders (A3.6 and B8).

ImageDataset           one item per image -> (image_tensor, label, isic_id); upgrade of session2 LesionBatches
LesionDataset          one item per lesion -> dict with one tensor per view, label and lesion_id
aggregate_predictions  image-level class probabilities -> one prediction per lesion
make_loaders           train / val / test DataLoaders: seeds, transforms, sampler
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from . import config, paths
from .data import require_images
from .labels import build_label_map, class_weights, load_label_map, sample_weights
from .transforms import eval_transform, load_rgb, train_transform


def seed_everything(seed=config.SEED):
    """Seed Python, NumPy and PyTorch so a run can be repeated exactly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(worker_id):
    """Give every DataLoader worker its own reproducible NumPy and random seed."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


class ImageDataset(Dataset):
    """One image per item -> (image_tensor, label, isic_id).

    Upgrade of the session 2 loader (LesionBatches): it reads a split CSV, turns labels
    into integers with label_map.json, uses a torchvision transform, and fails loudly
    on a missing file unless allow_missing=True.
    """

    def __init__(self, split, transform, label_col=config.LABEL, label_map=None,
                 image_dir=None, allow_missing=False):
        table = pd.read_csv(split) if isinstance(split, (str, Path)) else split
        self.table = require_images(table, allow_missing=allow_missing, image_dir=image_dir)
        mapping = (label_map or load_label_map())[label_col]
        unknown = set(self.table[label_col]) - set(mapping)
        if unknown:
            raise ValueError(f"labels not in label_map for {label_col}: {sorted(unknown)}")
        self.label_col = label_col
        self.classes = sorted(mapping, key=mapping.get)
        self.labels = self.table[label_col].map(mapping).to_numpy()
        self.ids = self.table["isic_id"].tolist()
        self.paths = [paths.find_image(i, image_dir) for i in self.ids]
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        image = self.transform(load_rgb(self.paths[i]))
        return image, int(self.labels[i]), self.ids[i]


class LesionDataset(Dataset):
    """One lesion per item -> {view: tensor for each requested view, "label": int, "lesion_id": str}.

    lesion_table needs lesion_id, label_col and one <view>_id column per view (derm_id,
    clinical_id: the isic_id of each image). The transform runs separately on every view,
    so each view gets its own random augmentation. Missing files raise at construction.
    """

    def __init__(self, lesion_table, img_dir, label_col, transform, views=("derm", "clinical"),
                 label_map=None):
        self.table = lesion_table.reset_index(drop=True)
        self.views = tuple(views)
        unknown = set(self.views) - set(config.IMAGE_TYPES.values())
        if unknown:
            allowed = sorted(config.IMAGE_TYPES.values())
            raise ValueError(f"unknown views {sorted(unknown)}, use {allowed}")
        mapping = label_map or build_label_map().get(label_col)
        if mapping is None:
            mapping = {c: i for i, c in enumerate(sorted(self.table[label_col].unique()))}
        self.class_to_index = mapping
        self.labels = self.table[label_col].map(mapping).to_numpy()
        self.lesion_ids = self.table["lesion_id"].tolist()
        self.transform = transform
        self.paths = {view: [self._locate(lesion, isic_id, view, img_dir)
                             for lesion, isic_id in zip(self.lesion_ids, self.table[f"{view}_id"])]
                      for view in self.views}

    @staticmethod
    def _locate(lesion_id, isic_id, view, img_dir):
        try:
            return paths.find_image(isic_id, img_dir)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"lesion {lesion_id}: {view} image {isic_id} is missing in {paths.shown(img_dir)}") from None

    def __len__(self):
        return len(self.lesion_ids)

    def __getitem__(self, i):
        item = {view: self.transform(load_rgb(self.paths[view][i])) for view in self.views}
        item["label"] = int(self.labels[i])
        item["lesion_id"] = self.lesion_ids[i]
        return item


def aggregate_predictions(image_probs, image_table, classes=None):
    """Average the class probabilities of each lesion's images -> one row per lesion.

    image_probs is (n_images, n_classes), aligned row by row with image_table (which needs
    lesion_id). Returns a DataFrame indexed by lesion_id with the mean probability per class,
    n_images and pred (the index of the largest mean probability).
    """
    probs = np.asarray(image_probs, dtype=float)
    if probs.ndim != 2 or len(probs) != len(image_table):
        raise ValueError(f"expected ({len(image_table)}, n_classes) probabilities, got {probs.shape}")
    cols = [str(c) for c in classes] if classes is not None else [f"p{k}" for k in range(probs.shape[1])]
    frame = pd.DataFrame(probs, columns=cols)
    frame["lesion_id"] = image_table["lesion_id"].to_numpy()
    lesions = frame.groupby("lesion_id")[cols].mean()
    lesions["n_images"] = frame.groupby("lesion_id").size()
    lesions["pred"] = lesions[cols].to_numpy().argmax(axis=1)
    return lesions


def make_loaders(label_col=config.LABEL, balance="sampler", batch_size=config.BATCH_SIZE,
                 num_workers=config.NUM_WORKERS, seed=config.SEED, image_dir=None):
    """Train / val / test DataLoaders over the split CSVs.

    train: train_transform, and either a WeightedRandomSampler with class-balanced weights
    (balance="sampler") or a plain seeded shuffle (balance="shuffle").
    val and test: eval_transform, fixed order, no augmentation.
    """
    seed_everything(seed)
    label_map = load_label_map()
    datasets = {name: ImageDataset(config.SPLITS_DIR / f"{name}.csv",
                                   train_transform() if name == "train" else eval_transform(),
                                   label_col, label_map, image_dir)
                for name in ("train", "val", "test")}
    common = {"batch_size": batch_size, "num_workers": num_workers, "worker_init_fn": seed_worker}
    train_set = datasets["train"]
    if balance == "sampler":
        labels = train_set.table[label_col]
        weights = sample_weights(labels, class_weights(labels, train_set.classes))
        sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=len(train_set),
                                        replacement=True, generator=torch.Generator().manual_seed(seed))
        train = DataLoader(train_set, sampler=sampler, **common)
    elif balance == "shuffle":
        train = DataLoader(train_set, shuffle=True, generator=torch.Generator().manual_seed(seed), **common)
    else:
        raise ValueError("balance must be 'sampler' or 'shuffle'")
    return {"train": train,
            "val": DataLoader(datasets["val"], shuffle=False, **common),
            "test": DataLoader(datasets["test"], shuffle=False, **common)}
