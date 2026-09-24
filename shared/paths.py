from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "milk10k"
IMAGE_DIR = DATA_DIR / "images"
METADATA_CSV = DATA_DIR / "metadata.csv"
LABELS_CSV = DATA_DIR / "supplements" / "training_gt.csv"

LABEL = "diagnosis_1"
SUBCLASSES = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF",
              "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
EXTENSIONS = (".jpg", ".jpeg", ".png")


def find_image(isic_id):
    for ext in EXTENSIONS:
        candidate = IMAGE_DIR / f"{isic_id}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{isic_id}: no image file in {IMAGE_DIR}")


def image_exists(isic_id):
    return any((IMAGE_DIR / f"{isic_id}{ext}").exists() for ext in EXTENSIONS)
