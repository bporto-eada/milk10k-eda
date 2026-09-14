"""MILK10k EDA, session 1.

Runs against the local copy of the dataset in eda/milk10k/ (already downloaded
and unzipped next to this script). No download step, no network needed.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# All paths resolve relative to this file, so the script works no matter
# which folder you run it from (workspace root, eda/, VS Code Run button).
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "milk10k"
IMG_DIR = DATA_DIR / "images"
METADATA_PATH = DATA_DIR / "metadata.csv"
GT_PATH = DATA_DIR / "supplements" / "training_gt.csv"

CLASS_COLS = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF",
              "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]


def check_dataset() -> None:
    """Stop early with a clear message if the local dataset is not where expected."""
    missing = [p for p in (IMG_DIR, METADATA_PATH, GT_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Local milk10k dataset not found. Missing: "
            + ", ".join(str(p) for p in missing)
        )
    print("Using dataset at:", DATA_DIR)


def count_images() -> int:
    n_images = len(list(IMG_DIR.glob("*.jpg")))
    print("Number of images:", n_images)
    return n_images


def load_metadata() -> pd.DataFrame:
    df = pd.read_csv(METADATA_PATH)
    print("\nmetadata.csv shape:", df.shape)
    print(df.head())
    print("\ncolumns:", list(df.columns))
    print("\ndiagnosis_1 classes:", df["diagnosis_1"].unique())
    return df


def load_ground_truth() -> pd.DataFrame:
    gt = pd.read_csv(GT_PATH)
    print("\ntraining_gt.csv shape:", gt.shape)
    print(gt.head())
    print("\nlabels per row (should all be 1):")
    print(gt[CLASS_COLS].sum(axis=1).value_counts())
    gt["subclass"] = gt[CLASS_COLS].idxmax(axis=1)
    return gt


def merge_labels(df: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    data = df.merge(gt[["lesion_id", "subclass"]], on="lesion_id", how="left")
    print("\nmerged shape:", data.shape)
    print("images with no subclass label:", data["subclass"].isna().sum())
    return data


def plot_class_distribution(data: pd.DataFrame) -> None:
    counts = data["diagnosis_1"].value_counts()
    ax = counts.plot(kind="bar", color="steelblue", figsize=(6, 4), rot=0)
    ax.set_title("Samples per class (diagnosis_1)")
    ax.set_xlabel("")
    ax.set_ylabel("number of images")
    ax.bar_label(ax.containers[0])
    plt.tight_layout()

    print("\nsamples per class:")
    print(counts)
    print((counts / counts.sum() * 100).round(1))  # % share


def plot_subclass_distribution(data: pd.DataFrame) -> None:
    sub_counts = data["subclass"].value_counts()
    ax = sub_counts.plot(kind="bar", color="indianred", figsize=(8, 4), rot=45)
    ax.set_title("Samples per subclass (11 diagnoses)")
    ax.set_xlabel("")
    ax.set_ylabel("number of images")
    ax.bar_label(ax.containers[0])
    plt.tight_layout()

    print("\nsamples per subclass:")
    print(sub_counts)


def analyze_age(data: pd.DataFrame) -> None:
    stats = (data.groupby("diagnosis_1")["age_approx"]
             .describe()[["count", "min", "25%", "50%", "75%", "max", "mean"]]
             .round(1))
    print("\nage per class:")
    print(stats)
    print("missing ages:", data["age_approx"].isna().sum())

    data.boxplot(column="age_approx", by="diagnosis_1", figsize=(6, 4), grid=False)
    plt.title("Age distribution per class")
    plt.suptitle("")
    plt.xlabel("")
    plt.ylabel("age_approx")
    plt.tight_layout()


def analyze_sex(data: pd.DataFrame) -> None:
    sex_ct = pd.crosstab(data["diagnosis_1"], data["sex"])
    sex_pct = (pd.crosstab(data["diagnosis_1"], data["sex"], normalize="index")
               .mul(100).round(1))
    print("\nsex per class:")
    print(sex_ct)
    print(sex_pct)
    print("missing sex:", data["sex"].isna().sum())

    ax = sex_ct.plot(kind="bar", figsize=(6, 4), rot=0)
    ax.set_title("Sex distribution per class")
    ax.set_xlabel("")
    ax.set_ylabel("number of images")
    for c in ax.containers:
        ax.bar_label(c)
    plt.tight_layout()


def main() -> None:
    check_dataset()
    count_images()
    df = load_metadata()
    gt = load_ground_truth()
    data = merge_labels(df, gt)

    plot_class_distribution(data)
    plot_subclass_distribution(data)
    analyze_age(data)
    analyze_sex(data)

    plt.show()  # opens all four figures at once


if __name__ == "__main__":
    main()
