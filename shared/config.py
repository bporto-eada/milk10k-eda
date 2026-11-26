"""The one place for every setting the project uses.

Data location, output folders, seed, label columns, split sizes, image size and
loader settings all live here. Every other module imports them from this file,
so nothing else in the repo builds a path to the data on its own.

The data folder is read from the environment variable MILK10K_DIR. When it is
not set, the default is the folder milk10k/ at the root of the repo. If only the
images live elsewhere (as in the class notebook), set MILK10K_IMAGES_DIR instead:

    milk10k/metadata.csv
    milk10k/images/ISIC_xxxxxxx.jpg
    milk10k/supplements/training_gt.csv
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---- data (never committed) ----
DATA_DIR = Path(os.environ.get("MILK10K_DIR", REPO_ROOT / "milk10k")).expanduser()
IMAGE_DIR = Path(os.environ.get("MILK10K_IMAGES_DIR", DATA_DIR / "images")).expanduser()
METADATA_CSV = DATA_DIR / "metadata.csv"
LABELS_CSV = DATA_DIR / "supplements" / "training_gt.csv"
EXTENSIONS = (".jpg", ".jpeg", ".png")

# ---- generated files (small ones are committed, see README) ----
OUTPUT_DIR = REPO_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
SPLITS_DIR = OUTPUT_DIR / "splits"
LESIONS_CSV = OUTPUT_DIR / "lesions.csv"
LABEL_MAP_JSON = OUTPUT_DIR / "label_map.json"
CLASS_WEIGHTS_JSON = OUTPUT_DIR / "class_weights.json"
NORMALIZATION_JSON = OUTPUT_DIR / "normalization.json"
IMAGE_SIZE_SUMMARY_CSV = OUTPUT_DIR / "image_size_summary.csv"
IMAGE_SIZES_CSV = OUTPUT_DIR / "image_sizes.csv"
BASELINE_DIR = OUTPUT_DIR / "classical_baseline"            # classical ML baseline (m2_baseline/classical_baseline.py)
BASELINE_FIGURES_DIR = BASELINE_DIR / "figures"
IMAGE_FEATURES_CSV = BASELINE_DIR / "image_features.csv.gz"   # one row of hand-crafted features per image
MANIPULATION_CSV = BASELINE_DIR / "image_manipulation.csv"     # isic_id -> image_manipulation, test subgroups only

# ---- Milestones 4 and 5: best model, final pipeline ----
M4_DIR = REPO_ROOT / "m4_best_model"
M5_DIR = REPO_ROOT / "m5_final"
M4_OUT = OUTPUT_DIR / "m4_best_model"                       # scoreboard, predictions, error analysis, fairness
M5_OUT = OUTPUT_DIR / "m5_final"                            # 11-class results
MODEL_M4 = M4_DIR / "model_m4.pt"                           # the best 3-class model (committed)
MODEL_11CLASS = M5_DIR / "model_11class.pt"                 # the 11-class leaderboard model (committed)
SUBMISSION_CSV = M5_DIR / "submission.csv"
SUPPLEMENT_INPUT_CSV = DATA_DIR / "supplements" / "training_input.csv"   # skin tone and MONET scores per image
# The 479 blind test lesions of the ISIC leaderboard: MILK10k_Test_Metadata.csv and MILK10k_Test_Input/
TEST_DIR = Path(os.environ.get("MILK10K_TEST_DIR", REPO_ROOT / "milk10k_test")).expanduser()
# Fine-tuning recipe of Session 7's best network, used again for the 3-class and for the 11-class model
FINE_TUNE_EPOCHS = 6
FINE_TUNE_LR = 1e-4

# ---- labels ----
LABEL = "diagnosis_1"            # primary target: Benign / Indeterminate / Malignant
STRETCH_LABEL = "dx"             # 11-class label taken from training_gt.csv
SUBCLASSES = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF",
              "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
RARE_CLASSES = ["MAL_OTH", "DF", "INF", "VASC", "BEN_OTH"]   # fewer than ~50 lesions
IMAGE_TYPES = {"dermoscopic": "derm", "clinical: close-up": "clinical"}

# ---- reproducibility and splits ----
SEED = 42
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# ---- model input and loading ----
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2

# ---- classical ML baseline ----
CV_FOLDS = 5                     # stratified folds on the train lesions for hyper-parameter search
BOOTSTRAP_RESAMPLES = 2000       # stratified bootstrap of the test lesions for 95% intervals
TRIAGE_SENSITIVITY = 0.90        # Malignant sensitivity the triage threshold must reach
