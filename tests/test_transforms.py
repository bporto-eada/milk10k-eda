"""Tests for shared/transforms.py: shapes, determinism of eval_transform, randomness of train_transform."""

import numpy as np
import torch
from PIL import Image

from shared.transforms import RandomQuarterTurn, denormalize, eval_transform, train_transform

NORM = ([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])


def sample_image(seed=0, size=(600, 450)):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))


def test_eval_transform_is_deterministic():
    image, transform = sample_image(), eval_transform(norm=NORM)
    assert torch.equal(transform(image), transform(image))


def test_output_shape_and_dtype():
    image = sample_image()
    for transform in (eval_transform(norm=NORM), train_transform(norm=NORM)):
        x = transform(image)
        assert x.shape == (3, 224, 224) and x.dtype == torch.float32


def test_train_transform_is_random():
    torch.manual_seed(0)
    image, transform = sample_image(), train_transform(norm=NORM)
    assert not torch.equal(transform(image), transform(image))


def test_quarter_turn_only_moves_pixels():
    image = sample_image(size=(64, 64))
    torch.manual_seed(1)
    turned = RandomQuarterTurn()(image)
    assert turned.size == image.size
    assert np.array_equal(np.sort(np.asarray(turned), axis=None), np.sort(np.asarray(image), axis=None))


def test_denormalize_undoes_normalize():
    image = sample_image()
    x = eval_transform(norm=NORM)(image)
    restored = denormalize(x, NORM)
    assert restored.shape == (224, 224, 3)
    assert 0.0 <= restored.min() and restored.max() <= 1.0
