"""Milestones 4 and 5 on synthetic images and untrained networks: no dataset and no download needed."""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet18

from shared import deep


def synthetic_image(seed=0, size=(600, 450)):
    """A random RGB image with the size of a MILK10k photo."""
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))


def test_val_tf_is_deterministic():
    """The same image must always give the same tensor: no randomness outside training."""
    image = synthetic_image()
    assert torch.equal(deep.val_tf(image), deep.val_tf(image))
    assert deep.val_tf(image).shape == (3, 224, 224)


def test_training_and_prediction_see_the_same_numbers():
    """Training reads stored pixels through prep(); predict() reads val_tf. Both must agree (no train / serve skew)."""
    image = synthetic_image(1)
    served = deep.val_tf(image)
    trained = deep.prep(deep.to_uint8(image).unsqueeze(0))[0]
    assert torch.allclose(served, trained, atol=1e-6)


def test_augment_only_flips():
    """A flipped batch has the same pixels in another place, and the same shape."""
    torch.manual_seed(0)
    batch = torch.rand(16, 3, 8, 8)
    flipped = deep.augment(batch)
    assert flipped.shape == batch.shape
    assert torch.allclose(flipped.sum(dim=(1, 2, 3)), batch.sum(dim=(1, 2, 3)), atol=1e-4)


def test_fit_learns_and_keeps_the_best_epoch():
    """The loop on a toy problem: bright images are class 1, dark images class 0."""
    torch.manual_seed(0)
    dark = torch.randint(0, 100, (60, 3, 8, 8), dtype=torch.uint8)
    bright = torch.randint(156, 256, (60, 3, 8, 8), dtype=torch.uint8)
    X, y = torch.cat([dark, bright]), torch.tensor([0] * 60 + [1] * 60)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 2))
    history, best = deep.fit(model, list(model.parameters()), X, y, X, y, torch.ones(2), epochs=3, lr=0.01)
    assert len(history) == 3 and best["f1"] == history["val_macro_f1"].max() > 0.9
    probabilities = deep.predict_proba(model, X)
    assert probabilities.shape == (120, 2) and torch.allclose(probabilities.sum(dim=1), torch.ones(120), atol=1e-5)
    assert deep.macro_f1(y, probabilities.argmax(1)) == best["f1"]       # the model holds its best epoch


def test_predict_gives_one_probability_per_class(tmp_path):
    """predict() on a saved (untrained) ResNet-18: the right classes, adding up to 1, same as the validation path."""
    from m5_final.predict import predict

    torch.manual_seed(0)
    classes = ["Benign", "Indeterminate", "Malignant"]
    model = resnet18()
    model.fc = nn.Linear(512, 3)
    torch.save({"state_dict": model.state_dict(), "classes": classes}, tmp_path / "model.pt")
    image = synthetic_image(2)
    image.save(tmp_path / "lesion.png")

    probabilities = predict(tmp_path / "lesion.png", tmp_path / "model.pt")
    assert list(probabilities) == classes
    assert abs(sum(probabilities.values()) - 1) < 1e-5
    model.eval()
    with torch.no_grad():
        expected = torch.softmax(model(deep.val_tf(image).unsqueeze(0)), dim=1)[0]
    assert np.allclose(list(probabilities.values()), expected.numpy(), atol=1e-6)
