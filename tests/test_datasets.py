"""Tests for shared/datasets.py on a tiny synthetic image folder (no MILK10k data needed)."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T

from shared.datasets import ImageDataset, LesionDataset, aggregate_predictions

SMALL = T.Compose([T.Resize((32, 32)), T.ToTensor()])


def fake_lesions(folder, n=20):
    """Write 2 small JPEGs per lesion into folder and return the lesion table."""
    rows = []
    for i in range(n):
        row = {"lesion_id": f"IL_{i:03d}", "derm_id": f"ISIC_d{i:03d}", "clinical_id": f"ISIC_c{i:03d}",
               "diagnosis_1": ["Benign", "Indeterminate", "Malignant"][i % 3]}
        for key in ("derm_id", "clinical_id"):
            Image.new("RGB", (60, 45), (i * 10 % 255, 80, 120)).save(folder / f"{row[key]}.jpg")
        rows.append(row)
    return pd.DataFrame(rows)


def test_lesion_batch_shapes_and_ids(tmp_path):
    lesions = fake_lesions(tmp_path)
    dataset = LesionDataset(lesions, tmp_path, "diagnosis_1", SMALL)
    batch = next(iter(DataLoader(dataset, batch_size=8)))
    assert batch["derm"].shape == (8, 3, 32, 32) and batch["clinical"].shape == (8, 3, 32, 32)
    expected = lesions.set_index("lesion_id").loc[batch["lesion_id"], "diagnosis_1"].map(dataset.class_to_index)
    assert batch["label"].tolist() == expected.tolist()


def test_missing_image_names_the_lesion(tmp_path):
    lesions = fake_lesions(tmp_path)
    (tmp_path / "ISIC_c004.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="IL_004"):
        LesionDataset(lesions, tmp_path, "diagnosis_1", SMALL)


def test_image_dataset_fails_loudly_unless_allowed(tmp_path):
    lesions = fake_lesions(tmp_path, n=6)
    images = pd.DataFrame({"isic_id": list(lesions["derm_id"]) + ["ISIC_missing"],
                           "diagnosis_1": list(lesions["diagnosis_1"]) + ["Benign"]})
    label_map = {"diagnosis_1": {"Benign": 0, "Indeterminate": 1, "Malignant": 2}}
    with pytest.raises(FileNotFoundError):
        ImageDataset(images, SMALL, label_map=label_map, image_dir=tmp_path)
    with pytest.warns(UserWarning):
        dataset = ImageDataset(images, SMALL, label_map=label_map, image_dir=tmp_path, allow_missing=True)
    image, label, isic_id = dataset[0]
    assert len(dataset) == 6 and image.shape == (3, 32, 32) and isinstance(label, int)
    assert isic_id == "ISIC_d000"


def test_aggregate_predictions_averages_the_two_views():
    table = pd.DataFrame({"lesion_id": ["A", "B", "A", "B"]})
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]])
    out = aggregate_predictions(probs, table, classes=["Benign", "Malignant"])
    assert out.loc["A", "Benign"] == pytest.approx(0.6) and out.loc["A", "pred"] == 0
    assert out.loc["B", "Malignant"] == pytest.approx(0.7) and out.loc["B", "pred"] == 1
    assert out["n_images"].tolist() == [2, 2]
    shuffled = aggregate_predictions(probs[[3, 1, 0, 2]], table.iloc[[3, 1, 0, 2]], classes=["Benign", "Malignant"])
    pd.testing.assert_frame_equal(out, shuffled)
