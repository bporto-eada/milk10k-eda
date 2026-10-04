"""Image transforms (B6, B7) and the train-only normalization statistics.

eval_transform   deterministic: shorter side to IMAGE_SIZE, central square, tensor, normalize
train_transform  random crop/zoom, flips, quarter turns, mild brightness/contrast, then the same
AUGMENTATIONS    the augmentation table: parameters and medical justification
"""

import json
from datetime import date

import torch
from PIL import Image
from torchvision import transforms as T

from . import config, paths

CROP_SCALE = (0.8, 1.0)   # at least 80% of the frame stays: a centred lesion is never cut out
JITTER = {"brightness": 0.1, "contrast": 0.1}   # no hue or saturation jitter, see A3.5


def load_rgb(path):
    """Open an image file as a fully loaded RGB PIL image (the file is closed again)."""
    with Image.open(path) as img:
        return img.convert("RGB")


class RandomQuarterTurn:
    """Rotate a PIL image by 0, 90, 180 or 270 degrees, chosen with torch's RNG.

    Quarter turns are lossless: no interpolation and no black corners, unlike a free rotation.
    """

    def __call__(self, img):
        k = int(torch.randint(0, 4, (1,)))
        return img.rotate(90 * k, expand=True) if k else img

    def __repr__(self):
        return f"{self.__class__.__name__}()"


AUGMENTATIONS = [
    {"augmentation": "RandomResizedCrop",
     "parameters": f"output {config.IMAGE_SIZE}x{config.IMAGE_SIZE}, area {CROP_SCALE[0]} to {CROP_SCALE[1]}, aspect 3/4 to 4/3",
     "justification": "zoom and framing vary between photographers; at least 80% of the frame stays, so the centred lesion is not cut out"},
    {"augmentation": "RandomHorizontalFlip", "parameters": "p = 0.5",
     "justification": "a lesion has no left or right side; the mirror image has the same diagnosis"},
    {"augmentation": "RandomVerticalFlip", "parameters": "p = 0.5",
     "justification": "the dermatoscope or camera can be held either way up"},
    {"augmentation": "RandomQuarterTurn", "parameters": "0, 90, 180 or 270 degrees",
     "justification": "orientation is arbitrary; quarter turns add no interpolation or black corners"},
    {"augmentation": "ColorJitter",
     "parameters": f"brightness {JITTER['brightness']}, contrast {JITTER['contrast']}, saturation 0, hue 0",
     "justification": "exposure differs between devices; hue is never touched because colour (red, brown, blue-grey) is diagnostic"},
]


def eval_transform(size=config.IMAGE_SIZE, norm=None):
    """Deterministic: Resize(size) on the shorter side, CenterCrop(size), ToTensor, Normalize."""
    mean, std = norm or load_normalization()
    return T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(), T.Normalize(mean, std)])


def train_transform(size=config.IMAGE_SIZE, norm=None):
    """Random, label-safe augmentation for training (see AUGMENTATIONS), then ToTensor and Normalize."""
    mean, std = norm or load_normalization()
    return T.Compose([
        T.RandomResizedCrop(size, scale=CROP_SCALE, ratio=(3 / 4, 4 / 3)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        RandomQuarterTurn(),
        T.ColorJitter(**JITTER),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def compute_normalization(isic_ids, size=config.IMAGE_SIZE, image_dir=None):
    """Per-channel mean and std over every pixel of the given (TRAIN) images.

    Images are resized and centre-cropped exactly like eval_transform, without augmentation,
    with pixel values in [0, 1]. Running sums in float64 avoid loading all images at once.
    """
    prep = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor()])
    total = torch.zeros(3, dtype=torch.float64)
    total_sq = torch.zeros(3, dtype=torch.float64)
    n_pixels = 0
    for isic_id in isic_ids:
        x = prep(load_rgb(paths.find_image(isic_id, image_dir))).double()
        total += x.sum(dim=(1, 2))
        total_sq += (x * x).sum(dim=(1, 2))
        n_pixels += x.shape[1] * x.shape[2]
    mean = total / n_pixels
    std = (total_sq / n_pixels - mean ** 2).sqrt()
    return [round(v, 4) for v in mean.tolist()], [round(v, 4) for v in std.tolist()]


def save_normalization(mean, std, n_images, path=config.NORMALIZATION_JSON):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "mean": mean, "std": std, "n_train_images": n_images, "image_size": config.IMAGE_SIZE,
        "computed_on": "train split only, after Resize + CenterCrop, pixel values in [0, 1]",
        "date": date.today().isoformat()}, indent=2))
    return path


def load_normalization(path=config.NORMALIZATION_JSON):
    """(mean, std) from normalization.json; fails loudly if the file was not produced yet."""
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found: run  python milestone1.py preprocess")
    stats = json.loads(path.read_text())
    return stats["mean"], stats["std"]


def denormalize(batch, norm=None):
    """Undo Normalize for display: (N,3,H,W) or (3,H,W) tensor -> HWC float array(s) in [0, 1]."""
    mean, std = norm or load_normalization()
    x = batch.detach().cpu().float()
    single = x.ndim == 3
    if single:
        x = x.unsqueeze(0)
    x = x * torch.tensor(std).view(1, 3, 1, 1) + torch.tensor(mean).view(1, 3, 1, 1)
    out = x.clamp(0, 1).permute(0, 2, 3, 1).numpy()
    return out[0] if single else out
