"""Milestone 1 data pipeline (Part B).

    python milestone1.py                  run every step, in order
    python milestone1.py splits loaders   run only the named steps

Steps
  integrity   B1  every row has an image file; Image.verify() on all; image size table
  eda         B2  class distributions, Session 2 findings re-checked, 3x4 class gallery
  labels      B3  label_map.json
  quality     B4  data_quality_report.md
  splits      B5  lesions.csv, splits/{train,val,test}.csv and their verification
  preprocess  B6  resolution evidence, normalization.json from the train split only
  augment     B7  eval_transform determinism check, augmentation figure and table
  loaders     B8  class_weights.json, the three DataLoaders and their sanity checks

Every file goes to outputs/ (locations in shared/config.py). The console output is
also appended to outputs/milestone1_log.txt.
"""

import json
import sys
import time
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from shared import checks, config, data, labels, paths, splits, transforms, viewer  # noqa: E402
from shared.datasets import make_loaders, seed_everything  # noqa: E402
from shared.figstyle import colour, finish  # noqa: E402

SPLIT_COLUMNS = ["isic_id", "lesion_id", "image_type", "diagnosis_1", "dx",
                 "age_approx", "sex", "anatom_site_general"]


# ---------------------------------------------------------------- helpers
class Tee:
    """Print to the console and append the same text to a log file."""

    def __init__(self, path):
        self.file = open(path, "a", encoding="utf-8")
        self.console = sys.stdout

    def write(self, text):
        self.console.write(text)
        self.file.write(text)

    def flush(self):
        self.console.flush()
        self.file.flush()

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, *exc):
        sys.stdout = self.console
        self.file.close()


def md_table(df, index=True):
    """A DataFrame as a Markdown table (no extra package needed)."""
    frame = df.reset_index() if index else df
    lines = ["| " + " | ".join(str(c) for c in frame.columns) + " |",
             "|" + "|".join("---" for _ in frame.columns) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path.relative_to(config.REPO_ROOT)}")


def mean_colours(table):
    """Mean R, G, B of every image in table. JPEG draft mode decodes at 1/4 scale, which is fast
    and leaves the image means practically unchanged."""
    rows = []
    for isic_id in table["isic_id"]:
        with Image.open(paths.find_image(isic_id)) as img:
            img.draft("RGB", (img.width // 4, img.height // 4))
            rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
        rows.append(rgb.reshape(-1, 3).mean(axis=0))
    return pd.DataFrame(rows, columns=["mean_R", "mean_G", "mean_B"], index=table.index)


def lesion_tables():
    images = data.load_tables()
    return images, data.build_lesion_table(images)


# ---------------------------------------------------------------- B1
def step_integrity():
    meta = data.load_metadata()
    missing = data.missing_images(meta)
    print(f"metadata rows: {len(meta):,}   rows whose image file is missing: {len(missing)}")
    for isic_id in missing:
        print("  missing:", isic_id)

    sizes = checks.inspect_images(meta)
    sizes.to_csv(config.IMAGE_SIZES_CSV, index=False)
    bad = sizes[sizes["error"] != ""]
    print(f"Image.verify() run on {len(sizes):,} files: {len(bad)} unreadable")
    if len(bad):
        print(bad[["isic_id", "error"]].to_string(index=False))

    summary = checks.size_summary(sizes)
    summary.to_csv(config.IMAGE_SIZE_SUMMARY_CSV, index=False)
    print(summary.to_string(index=False))

    try:
        data.require_images(pd.DataFrame({"isic_id": ["ISIC_0000000"]}))
    except FileNotFoundError as err:
        print("loader fails loudly on a missing file:", err)


# ---------------------------------------------------------------- B2
SESSION2 = {   # values printed by session2/run_session2.py (session2/results/session2_results.txt)
    "red": "mean R 177.5 Malignant vs 165.1 Benign (72 images)",
    "type": "mean B gap: image type 16.7 vs class 7.1 (72 images)",
    "manipulation": "Cramer's V 0.139",
    "confirm": "Cramer's V 0.311",
    "site": "37.3% missing",
    "age": "median 55 / 70 / 65 (Benign / Indet. / Malignant)",
    "image_type": "Cramer's V 0.000",
}


def session2_findings(images, lesions):
    colours = mean_colours(images)
    full = images.join(colours)
    red = full.groupby("diagnosis_1")["mean_R"].mean()
    blue_type = full.groupby("image_type")["mean_B"].mean()
    blue_class = full.groupby("diagnosis_1")["mean_B"].mean()
    type_gap = abs(blue_type["dermoscopic"] - blue_type["clinical: close-up"])
    class_gap = abs(blue_class["Malignant"] - blue_class["Benign"])
    v_manip = checks.cramers_v(pd.crosstab(images["image_manipulation"], images["diagnosis_1"]))
    lesion_meta = images.drop_duplicates("lesion_id")
    v_confirm = checks.cramers_v(pd.crosstab(lesion_meta["diagnosis_confirm_type"], lesion_meta["diagnosis_1"]))
    site_missing = images["anatom_site_general"].isna().mean() * 100
    nv_gap = (lesions.loc[lesions["site"].isna(), "dx"].eq("NV").mean()
              - lesions.loc[lesions["site"].notna(), "dx"].eq("NV").mean()) * 100
    age = lesions.groupby("diagnosis_1")["age"].median()
    v_type = checks.cramers_v(pd.crosstab(images["image_type"], images["diagnosis_1"]))

    rows = [
        ("Malignant images are redder than Benign ones", SESSION2["red"],
         f"mean R {red['Malignant']:.1f} vs {red['Benign']:.1f} (all {len(full):,} images)",
         red["Malignant"] > red["Benign"],
         "colour carries class signal: keep RGB input and never jitter hue (A3.5)"),
        ("Image type shifts colour more than the class does", SESSION2["type"],
         f"mean B gap: type {type_gap:.1f} vs class {class_gap:.1f} (all images)",
         type_gap > class_gap,
         "both views share one model: normalise with train stats over both types, evaluate per lesion"),
        ("image_manipulation ('altered') is linked to the diagnosis", SESSION2["manipulation"],
         f"Cramer's V {v_manip:.3f}", v_manip >= 0.1,
         "possible shortcut: never a model input; report test results for altered images separately"),
        ("diagnosis_confirm_type / concomitant_biopsy predict the label", SESSION2["confirm"],
         f"Cramer's V {v_confirm:.3f}", v_confirm >= 0.1,
         "label leakage: excluded from the inputs (data_quality_report.md)"),
        ("anatom_site_general is often missing", SESSION2["site"],
         f"{site_missing:.1f}% missing; NV share +{nv_gap:.1f} pp when missing", site_missing > 30,
         "missing is informative (mostly trunk): if site is used, encode 'unknown', never impute the mode"),
        ("Indeterminate and Malignant patients are older", SESSION2["age"],
         f"median {age['Benign']:.0f} / {age['Indeterminate']:.0f} / {age['Malignant']:.0f}",
         age["Benign"] < age["Malignant"],
         "age is a legitimate clinical input for a later metadata branch; impute 0.4% missing with train median"),
        ("image_type says nothing about the label", SESSION2["image_type"],
         f"Cramer's V {v_type:.3f}", v_type < 0.05,
         "not a shortcut: every lesion has one image of each type, so every class has both views"),
    ]
    table = pd.DataFrame(rows, columns=["Session 2 finding", "Session 2", "full dataset",
                                        "still true?", "consequence for the pipeline"])
    table["still true?"] = table["still true?"].map({True: "yes", False: "no"})
    return table


def plot_mapping(table):
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.imshow(np.log1p(table.to_numpy()), cmap="Blues", aspect="auto")
    for (i, j), v in np.ndenumerate(table.to_numpy()):
        if v:
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=8,
                    color="white" if v > 400 else "black")
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.set_title("Lesions per 11-class label and diagnosis_1\nonly AKIEC spans two diagnosis_1 values",
                 fontsize=10)
    finish(fig, "b2_dx_to_diagnosis_1.png")


def gallery(lesions):
    """One dermoscopic example per class: 11 classes, AKIEC once per diagnosis_1 value = 12 panels."""
    picks = (lesions.groupby(["dx", "diagnosis_1"]).sample(1, random_state=config.SEED)
             .sort_values(["dx", "diagnosis_1"]))
    images = [np.asarray(transforms.load_rgb(paths.find_image(i))) for i in picks["derm_id"]]
    captions = [f"{dx}\n{d1}" for dx, d1 in zip(picks["dx"], picks["diagnosis_1"])]
    viewer.image_grid(images, captions, ncols=4,
                      title="One dermoscopic example per class (AKIEC shown for both of its diagnosis_1 values)",
                      filename="b2_gallery_by_class.png")


def step_eda():
    images, lesions = lesion_tables()
    viewer.class_balance_chart(lesions, "diagnosis_1", "b2_classes_diagnosis_1.png", unit="lesions")
    dx_colour = {dx: colour(d1.iloc[0] if d1.nunique() == 1 else "mixed")
                 for dx, d1 in lesions.groupby("dx")["diagnosis_1"]}
    viewer.class_balance_chart(lesions, "dx", "b2_classes_dx_log.png", log=True,
                               colours=dx_colour, unit="lesions")
    mapping = checks.class_to_diagnosis_1(lesions)
    mapping.to_csv(config.OUTPUT_DIR / "dx_to_diagnosis_1.csv")
    plot_mapping(mapping)
    print("lesions per class and diagnosis_1:")
    print(mapping.to_string())

    findings = session2_findings(images, lesions)
    print(findings.to_string(index=False))
    write(config.OUTPUT_DIR / "session2_findings.md",
          "# Session 2 findings re-checked on the full dataset\n\n"
          "Produced by `python milestone1.py eda`.\n\n" + md_table(findings, index=False) + "\n")
    gallery(lesions)
    print("figures: b2_classes_diagnosis_1.png, b2_classes_dx_log.png, b2_dx_to_diagnosis_1.png, "
          "b2_gallery_by_class.png")


# ---------------------------------------------------------------- B3
def step_labels():
    labels.save_label_map()
    print(json.dumps(labels.build_label_map(), indent=2))
    _, lesions = lesion_tables()
    print("lesions per diagnosis_1:", lesions["diagnosis_1"].value_counts().to_dict())
    rare = lesions["dx"].value_counts().loc[config.RARE_CLASSES]
    print("rare 11-class labels (lesions):", rare.to_dict())


# ---------------------------------------------------------------- B4
MISSING_DECISIONS = {
    "age_approx": "impute with the TRAIN median if age becomes a model input",
    "anatom_site_general": "keep as its own category 'unknown': missing is informative (mostly trunk)",
    "anatom_site_special": "drop the column: 98% missing, only refines acral and genital sites",
    "diagnosis_3": "leave NaN: label field, never a model input",
    "diagnosis_4": "leave NaN: label field, never a model input",
    "melanocytic": "leave NaN: derived from the diagnosis, never a model input",
}

NOT_INPUTS = [
    ("diagnosis_2, diagnosis_3, diagnosis_4", "finer levels of the diagnosis: they contain the label"),
    ("dx", "the 11-class label (an input only if you want to leak the answer)"),
    ("diagnosis_confirm_type", "histopathology lesions are 72% Malignant, clinically assessed ones 2%"),
    ("concomitant_biopsy", "same information as diagnosis_confirm_type (identical 10,032 / 448 split)"),
    ("melanocytic", "only filled in for melanocytic lesions, so it follows from the diagnosis"),
    ("image_manipulation", "not a label field, but linked to the class: shortcut risk"),
    ("attribution, copyright_license", "one value for every image: no information"),
    ("isic_id, lesion_id", "identifiers"),
]


def crosstab_section(images, col):
    counts = pd.crosstab(images[col], images["diagnosis_1"], margins=True, margins_name="total")
    share = pd.crosstab(images[col], images["diagnosis_1"], normalize="columns").mul(100).round(1)
    v = checks.cramers_v(pd.crosstab(images[col], images["diagnosis_1"]))
    return counts, share, v


def step_quality():
    images, lesions = lesion_tables()
    meta = data.load_metadata()
    gt = data.load_ground_truth()

    missing = meta.isna().mean().mul(100).round(1)
    missing = missing[missing > 0]
    decisions = pd.DataFrame({"missing_%": missing,
                              "decision": [MISSING_DECISIONS.get(c, "leave NaN, not used") for c in missing.index]})
    decisions.index.name = "column"

    pairs = checks.image_pair_violations(meta)
    per_lesion = meta.groupby("lesion_id").size()
    ids = checks.compare_lesion_ids(meta, gt)
    one_hot = checks.one_hot_violations(gt)
    mapping = checks.class_to_diagnosis_1(lesions)
    ambiguous = checks.ambiguous_classes(mapping)
    inconsistent = checks.inconsistent_lesions(meta)
    h32 = checks.hierarchy_violations(meta, "diagnosis_3", "diagnosis_2")
    h21 = checks.hierarchy_violations(meta, "diagnosis_2", "diagnosis_1")
    consistency = pd.DataFrame([
        ("images per lesion is exactly 2", f"{(per_lesion == 2).sum():,} of {len(per_lesion):,} lesions"),
        ("one dermoscopic + one clinical image", f"{len(per_lesion) - len(pairs):,} of {len(per_lesion):,} lesions"),
        ("same lesion_id set in both label files", f"{len(ids['only_in_metadata'])} only in metadata, "
                                                   f"{len(ids['only_in_gt'])} only in training_gt"),
        ("exactly one positive class per gt row", f"{len(one_hot)} violations"),
        ("each diagnosis_3 under one diagnosis_2", f"{len(h32)} violations"),
        ("each diagnosis_2 under one diagnosis_1", f"{len(h21)} violations"),
        ("age, sex, site, diagnosis equal on both images", f"{int(inconsistent.sum())} disagreements"),
        ("each 11-class label maps to one diagnosis_1",
         "; ".join(f"{dx}: " + ", ".join(f"{c} {n}" for c, n in row[row > 0].items())
                   for dx, row in ambiguous.iterrows()) or "all do"),
    ], columns=["check", "result"])

    manip_counts, manip_share, manip_v = crosstab_section(images, "image_manipulation")
    type_counts, type_share, type_v = crosstab_section(images, "image_type")
    altered = manip_share.loc["altered"]

    report = f"""# Data-quality report, Milestone 1

Generated by `python milestone1.py quality` on {date.today().isoformat()}. It extends the
Session 2 audit (`session2/results/session2_results.txt`, field types and missing shares per
field), which is not repeated here.

## 1. Columns with missing values: handling decision

{md_table(decisions)}

## 2. Label consistency

{md_table(consistency, index=False)}

## 3. Shortcut check

### image_manipulation by diagnosis_1 (images)

{md_table(manip_counts)}

Share of each class (% of its images):

{md_table(manip_share)}

Cramer's V = {manip_v:.3f}. 'Altered' images are {altered['Benign']:.1f}% of Benign images,
{altered['Indeterminate']:.1f}% of Indeterminate and {altered['Malignant']:.1f}% of Malignant ones,
and almost all of them are clinical close-ups. A model could learn "edited photo means benign".
Conclusion: image_manipulation is not a model input, and the test results of altered images are
reported separately so a shortcut would be visible.

### image_type by diagnosis_1 (images)

{md_table(type_counts)}

Cramer's V = {type_v:.3f}. Every lesion has exactly one image of each type, so the type carries no
information about the label and is not a shortcut. It is still a strong confounder for colour
(session 2): one model sees both views of every class.

## 4. Columns that are NOT model inputs

{md_table(pd.DataFrame(NOT_INPUTS, columns=["column", "reason"]), index=False)}

Allowed inputs: the images, plus age_approx, sex and anatom_site_general for a later metadata
branch (image_type only as a view indicator).
"""
    write(config.OUTPUT_DIR / "data_quality_report.md", report)
    print(decisions.to_string())
    print(consistency.to_string(index=False))
    print(f"image_manipulation vs diagnosis_1: Cramer's V {manip_v:.3f}; image_type: {type_v:.3f}")


# ---------------------------------------------------------------- B5
def step_splits():
    images, lesions = lesion_tables()
    lesions.to_csv(config.LESIONS_CSV, index=False)
    train, val, test = splits.split_lesions(lesions, config.VAL_SIZE, config.TEST_SIZE, config.SEED)
    table = splits.assign_images(images, train, val, test)
    report = splits.verify_splits(table)

    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name in splits.SPLITS:
        part = table.loc[table["split"] == name, SPLIT_COLUMNS].sort_values(["lesion_id", "image_type"])
        part.to_csv(config.SPLITS_DIR / f"{name}.csv", index=False)
    info = {"seed": config.SEED, "created": date.today().isoformat(),
            "method": "StratifiedGroupKFold at lesion level (groups lesion_id, strata dx|diagnosis_1), "
                      "20 folds of 5%: 3 to test, 3 to val, 14 to train; images follow their lesion",
            "requested": {"val": config.VAL_SIZE, "test": config.TEST_SIZE},
            "lesions": report["sizes"]["lesions"].astype(int).to_dict(),
            "images": report["sizes"]["images"].astype(int).to_dict()}
    (config.SPLITS_DIR / "split_info.json").write_text(json.dumps(info, indent=2))

    text = (f"# Split verification\n\nSeed {config.SEED}, created {info['created']} by "
            f"`python milestone1.py splits`.\n\n## Sizes\n\n{md_table(report['sizes'])}\n\n"
            f"Lesion overlap between splits: {report['overlaps']}  \n"
            f"Lesions without exactly 2 images in one split: {report['lesions_not_paired']}\n\n"
            f"## 11-class proportions (% of lesions)\n\n{md_table(report['dx'])}\n\n"
            f"Largest deviation from the overall share: {report['dx_max_deviation_pp']} pp\n\n"
            f"## diagnosis_1 proportions (% of lesions)\n\n{md_table(report['diagnosis_1'])}\n\n"
            f"Largest deviation from the overall share: {report['diagnosis_1_max_deviation_pp']} pp\n")
    write(config.SPLITS_DIR / "split_report.md", text)
    print(report["sizes"].to_string())
    print("lesion overlap between splits:", report["overlaps"])
    print("lesions without exactly 2 images in one split:", report["lesions_not_paired"])
    print(report["dx"].to_string())
    print("max deviation, 11 classes (pp):", report["dx_max_deviation_pp"])
    print(report["diagnosis_1"].to_string())
    print("max deviation, diagnosis_1 (pp):", report["diagnosis_1_max_deviation_pp"])


# ---------------------------------------------------------------- B6
def step_preprocess():
    sizes = pd.read_csv(config.IMAGE_SIZES_CSV)
    short = sizes[["width", "height"]].min(axis=1)
    for side in (224, 256, 384, 450, 512):
        print(f"images whose shorter side is >= {side}px: {(short >= side).mean():.1%}")
    print(f"chosen input resolution: {config.IMAGE_SIZE}x{config.IMAGE_SIZE} "
          f"(Resize shorter side to {config.IMAGE_SIZE}, then CenterCrop)")

    train = pd.read_csv(config.SPLITS_DIR / "train.csv")
    start = time.time()
    mean, std = transforms.compute_normalization(train["isic_id"])
    transforms.save_normalization(mean, std, len(train))
    print(f"normalization from {len(train):,} TRAIN images ({time.time() - start:.0f}s): mean {mean}, std {std}")
    print("views: option (a), each image is a sample with its lesion's label; "
          "evaluation averages the two predictions per lesion (aggregate_predictions)")


# ---------------------------------------------------------------- B7
def step_augment():
    norm = transforms.load_normalization()
    train = pd.read_csv(config.SPLITS_DIR / "train.csv")
    image = transforms.load_rgb(paths.find_image(train["isic_id"].iloc[0]))

    evaluate = transforms.eval_transform()
    first, second = evaluate(image), evaluate(image)
    assert torch.equal(first, second), "eval_transform is not deterministic"
    print(f"eval_transform applied twice to {train['isic_id'].iloc[0]}: identical tensors {tuple(first.shape)}")

    augment = transforms.train_transform()
    seed_everything()
    print("train_transform applied twice gives different tensors:", not torch.equal(augment(image), augment(image)))

    rows, captions = [], []
    for cls in labels.PRIMARY_CLASSES:
        pool = train[(train["diagnosis_1"] == cls) & (train["image_type"] == "dermoscopic")]
        pick = pool.sample(1, random_state=config.SEED).iloc[0]
        original = transforms.load_rgb(paths.find_image(pick["isic_id"]))
        rows.append(np.asarray(original))
        captions.append(f"{cls} ({pick['dx']})\noriginal")
        for k in range(7):
            rows.append(transforms.denormalize(augment(original), norm))
            captions.append(f"augmented {k + 1}")
    viewer.image_grid(rows, captions, ncols=8,
                      title="train_transform: 1 original + 7 augmented views (Indeterminate is the rare class)",
                      filename="b7_augmentations.png")

    table = pd.DataFrame(transforms.AUGMENTATIONS)
    write(config.OUTPUT_DIR / "augmentation_table.md",
          "# Augmentations (train only)\n\nFrom `shared/transforms.py` (AUGMENTATIONS).\n\n"
          + md_table(table, index=False) + "\n")
    print(table.to_string(index=False))


# ---------------------------------------------------------------- B8
def label_histogram(loader, n_batches, classes):
    seen = []
    for k, (_, y, _) in enumerate(loader):
        if k == n_batches:
            break
        seen.append(y)
    counts = torch.bincount(torch.cat(seen), minlength=len(classes)).tolist()
    return pd.Series(counts, index=classes)


def step_loaders():
    seed_everything()
    train = pd.read_csv(config.SPLITS_DIR / "train.csv")
    weights = {}
    for col, mapping in labels.build_label_map().items():
        classes = sorted(mapping, key=mapping.get)
        w = labels.class_weights(train[col], classes)
        weights[col] = {"classes": classes,
                        "train_images": train[col].value_counts().reindex(classes).astype(int).tolist(),
                        "weights": [round(float(v), 4) for v in w],
                        "rule": "n_train_images / (n_classes * class_count), train split only"}
    write(config.CLASS_WEIGHTS_JSON, json.dumps(weights, indent=2))
    counts = train.drop_duplicates("lesion_id")["diagnosis_1"].value_counts()
    print(f"train imbalance (lesions): {counts.to_dict()}, ratio {counts.max() / counts.min():.1f} : 1")
    print("class weights diagnosis_1:", dict(zip(weights["diagnosis_1"]["classes"], weights["diagnosis_1"]["weights"])))

    loaders = make_loaders(balance="sampler")
    classes = loaders["train"].dataset.classes
    x, y, ids = next(iter(loaders["train"]))
    print(f"train batch: images {tuple(x.shape)} {x.dtype}, min {x.min():.2f}, max {x.max():.2f}; "
          f"labels {tuple(y.shape)} {y.dtype}; first ids {list(ids[:3])}")
    xv, yv, _ = next(iter(loaders["val"]))
    print(f"val batch:   images {tuple(xv.shape)} {xv.dtype}, min {xv.min():.2f}, max {xv.max():.2f}")
    print(f"batches per epoch: train {len(loaders['train'])}, val {len(loaders['val'])}, test {len(loaders['test'])}")

    sampled = label_histogram(loaders["train"], 20, classes)
    shuffled = label_histogram(make_loaders(balance="shuffle")["train"], 20, classes)
    print("labels seen in 20 train batches:")
    print(pd.DataFrame({"WeightedRandomSampler": sampled, "plain shuffle": shuffled}).to_string())

    loss = labels.weighted_loss()
    logits = torch.randn(len(y), len(classes), generator=torch.Generator().manual_seed(config.SEED))
    print(f"class-weighted cross-entropy on that batch (random logits): {loss(logits, y):.3f}")

    start = time.time()
    n = sum(len(batch[1]) for batch in loaders["train"])
    print(f"one train epoch: {n:,} images in {time.time() - start:.1f}s "
          f"(batch {config.BATCH_SIZE}, {config.NUM_WORKERS} workers)")

    shown = transforms.denormalize(x[:16])
    viewer.image_grid(list(shown), [f"{classes[t]}\n{i}" for t, i in zip(y[:16].tolist(), ids[:16])], ncols=8,
                      title="Train batch after train_transform (sampler on), labels as titles",
                      filename="b8_train_batch.png")


STEPS = {"integrity": step_integrity, "eda": step_eda, "labels": step_labels, "quality": step_quality,
         "splits": step_splits, "preprocess": step_preprocess, "augment": step_augment,
         "loaders": step_loaders}


def main(chosen):
    chosen = chosen or list(STEPS)
    unknown = [s for s in chosen if s not in STEPS]
    if unknown:
        raise SystemExit(f"unknown step(s) {unknown}; choose from {list(STEPS)}")
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with Tee(config.OUTPUT_DIR / "milestone1_log.txt"):
        print(f"\n===== milestone1 {date.today().isoformat()}  steps: {' '.join(chosen)}  seed {config.SEED}")
        for name in chosen:
            print(f"\n########## {name} ##########")
            start = time.time()
            STEPS[name]()
            print(f"[{name}: {time.time() - start:.0f}s]")


if __name__ == "__main__":
    main(sys.argv[1:])
