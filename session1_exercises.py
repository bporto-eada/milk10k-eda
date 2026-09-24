from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from session1 import check_dataset, IMG_DIR, METADATA_PATH

FIG_DIR = Path(__file__).resolve().parent / "figures"
EXPECTED_TYPES = ["clinical: close-up", "dermoscopic"]


def show_examples(df, diagnosis, n=6, seed=1):
    picked = df[df["diagnosis_1"] == diagnosis].sample(n=n, random_state=seed)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3))
    for ax, (_, row) in zip(axes, picked.iterrows()):
        with Image.open(IMG_DIR / f"{row['isic_id']}.jpg") as img:
            ax.imshow(img)
        ax.set_title(f"{diagnosis}\n{row['image_type']}", fontsize=7)
        ax.set_axis_off()
    fig.suptitle(f"{n} random {diagnosis} images", fontsize=10)
    plt.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    plt.savefig(FIG_DIR / f"ex1_examples_{diagnosis.lower()}.png", dpi=150, bbox_inches="tight")


def q1_top_malignant_subcategory(df):
    malignant = df[df["diagnosis_1"] == "Malignant"]
    shares = malignant["diagnosis_2"].value_counts(normalize=True)
    print(shares.head(10).round(3))
    top, share = shares.index[0], shares.iloc[0]
    print(f"\nQ1: the most frequent Malignant subcategory is '{top}', "
          f"{share:.1%} of all Malignant images ({int(share * len(malignant))} images)")


def q2_typical_image_size(df, n=200, seed=1):
    sizes = []
    for isic_id in df["isic_id"].sample(n, random_state=seed):
        with Image.open(IMG_DIR / f"{isic_id}.jpg") as img:
            sizes.append(img.size)
    sizes = pd.DataFrame(sizes, columns=["width", "height"])
    print(f"Q2: based on {n} random images")
    print(sizes.agg(["mean", "min", "max"]).round(1))
    print("distinct (width, height) pairs:", sizes.value_counts().to_dict())


def q3_visual_comparison(df):
    show_examples(df, "Malignant")
    show_examples(df, "Benign")
    print("Q3: figures saved to figures/ex1_examples_malignant.png and figures/ex1_examples_benign.png")
    print("What stands out: malignant lesions look pinker and less symmetric with a rougher border, "
          "benign lesions are mostly an even brown with a smoother outline.")


def q4_lesion_pair_check(df, lesion_id=None):
    if lesion_id is None:
        lesion_id = df["lesion_id"].iloc[0]
    pair = df[df["lesion_id"] == lesion_id]
    print(f"Lesion {lesion_id}:")
    print(pair[["isic_id", "image_type"]].to_string(index=False))
    found = sorted(pair["image_type"])
    print("Q4:", "confirmed" if found == EXPECTED_TYPES else "not as expected", found)

    per_lesion = df.groupby("lesion_id")["image_type"].apply(lambda s: sorted(s) == EXPECTED_TYPES)
    print(f"Lesions with exactly one clinical and one dermoscopic image: "
          f"{per_lesion.sum()} of {len(per_lesion)}")
    print("Missing share per field:")
    print(df[["age_approx", "sex", "anatom_site_general"]].isna().mean().round(3))


def main():
    check_dataset()
    df = pd.read_csv(METADATA_PATH)

    print("\n== Question 1 ==")
    q1_top_malignant_subcategory(df)
    print("\n== Question 2 ==")
    q2_typical_image_size(df)
    print("\n== Question 3 ==")
    q3_visual_comparison(df)
    print("\n== Question 4 ==")
    q4_lesion_pair_check(df)

    plt.show()


if __name__ == "__main__":
    main()
