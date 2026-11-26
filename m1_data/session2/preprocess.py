from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from shared import paths

LUMA = np.array([0.299, 0.587, 0.114])


def to_grayscale(arr):
    return np.round(np.dot(arr[..., :3], LUMA)).astype(np.uint8)


def prepare_image(source, size=(224, 224), mode="rgb", scaling="unit"):
    if isinstance(source, (str, Path)):
        img = Image.open(source)
    else:
        img = Image.fromarray(np.asarray(source))
    img = img.convert("RGB")
    original = (img.height, img.width)

    if size is not None:
        img = img.resize((size[1], size[0]))
    arr = np.asarray(img)
    if mode == "gray":
        arr = to_grayscale(arr)

    if scaling == "unit":
        arr = arr.astype(np.float32) / 255.0
    elif scaling == "standardize":
        arr = arr.astype(np.float32)
        arr = (arr - arr.mean(axis=(0, 1))) / (arr.std(axis=(0, 1)) + 1e-6)

    details = {
        "original_hw": original,
        "shape": arr.shape,
        "mode": mode,
        "scaling": scaling,
        "min": round(float(arr.min()), 3),
        "max": round(float(arr.max()), 3),
    }
    return arr, details


def prepare_many(items, **options):
    if isinstance(items, pd.DataFrame):
        ids = list(items["isic_id"])
        sources = [None] * len(ids)
    else:
        sources = list(items)
        ids = [Path(s).stem for s in sources]

    arrays, kept, failed = [], [], []
    for item_id, src in zip(ids, sources):
        try:
            location = paths.find_image(item_id) if src is None else src
            arr, _ = prepare_image(location, **options)
            arrays.append(arr)
            kept.append(item_id)
        except Exception as err:
            failed.append((item_id, type(err).__name__))

    shapes = {a.shape for a in arrays}
    if arrays and len(shapes) == 1:
        arrays = np.stack(arrays)
    return arrays, kept, failed
