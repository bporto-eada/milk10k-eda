"""Where each image lives on disk.

All locations come from shared.config. LABEL and SUBCLASSES are re-exported
because the session 2 scripts import them from here.
"""

from pathlib import Path

from .config import IMAGE_DIR, EXTENSIONS, LABEL, REPO_ROOT, SUBCLASSES  # noqa: F401


def shown(path):
    """A path for messages: relative to the repo root when it is inside the repo."""
    path = Path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def find_image(isic_id, image_dir=None):
    """Return the path of the image file for isic_id. Raise FileNotFoundError if there is none."""
    folder = Path(image_dir) if image_dir is not None else IMAGE_DIR
    for ext in EXTENSIONS:
        candidate = folder / f"{isic_id}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{isic_id}: no image file in {shown(folder)}")


def image_exists(isic_id, image_dir=None):
    """True if an image file for isic_id exists (any of the accepted extensions)."""
    folder = Path(image_dir) if image_dir is not None else IMAGE_DIR
    return any((folder / f"{isic_id}{ext}").exists() for ext in EXTENSIONS)
