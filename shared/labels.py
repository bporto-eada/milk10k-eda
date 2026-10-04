"""Label strategy (B3) and class-imbalance weights (B8).

Decision (justified in reports/milestone1_report.md):
  primary target diagnosis_1 keeps Indeterminate as a third class;
  stretch target dx keeps all 11 classes, rare ones are handled with weights.
"""

import json

import torch

from . import config

PRIMARY_CLASSES = ["Benign", "Indeterminate", "Malignant"]


def build_label_map():
    """{target column: {class name: integer label}}. Later milestones load this from label_map.json."""
    return {
        config.LABEL: {name: i for i, name in enumerate(PRIMARY_CLASSES)},
        config.STRETCH_LABEL: {name: i for i, name in enumerate(config.SUBCLASSES)},
    }


def save_label_map(path=config.LABEL_MAP_JSON):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_label_map(), indent=2))
    return path


def load_label_map(path=config.LABEL_MAP_JSON):
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found: run  python milestone1.py labels")
    return json.loads(path.read_text())


def class_weights(labels, classes):
    """Balanced weights n / (k * count_c), computed on the TRAIN labels only.

    A class with half the average frequency gets weight 2. Returned in the order of
    classes, which is the integer-label order of label_map.json.
    """
    counts = labels.value_counts().reindex(classes, fill_value=0)
    if (counts == 0).any():
        raise ValueError(f"no training examples for {list(counts[counts == 0].index)}")
    return len(labels) / (len(classes) * counts)


def sample_weights(labels, weights):
    """Per-example weight for WeightedRandomSampler: the weight of the example's class."""
    return labels.map(weights).to_numpy(dtype=float)


def load_class_weights(label_col=config.LABEL, path=config.CLASS_WEIGHTS_JSON):
    """Class weights from class_weights.json as a tensor in label order."""
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found: run  python milestone1.py loaders")
    entry = json.loads(path.read_text())[label_col]
    return torch.tensor(entry["weights"], dtype=torch.float32)


def weighted_loss(label_col=config.LABEL):
    """Cross-entropy that counts each class by its weight from class_weights.json."""
    return torch.nn.CrossEntropyLoss(weight=load_class_weights(label_col))
