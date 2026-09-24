import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")

from shared import paths, data
from shared.paths import LABEL
import metadata_fields
import colour_profile
from preprocess import prepare_image, prepare_many
from batch_loader import LesionBatches
from viewer import image_grid, class_balance_chart, batch_diagnostics

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
transcript = []


def say(*parts):
    line = " ".join(str(p) for p in parts)
    print(line)
    transcript.append(line)


def main():
    df = data.load_tables()
    available = data.on_disk(df)
    say(f"{len(df):,} rows in metadata, {len(available):,} images present on disk")

    say("\n########## PART 1  metadata fields ##########")
    metadata_fields.run(df, say)

    say("\n########## PART 2  colour profile ##########")
    colour_profile.run(available, say, per_group=12)

    say("\n########## PART 3  preprocessing ##########")
    demo = available.groupby(LABEL).sample(2, random_state=5).head(4)
    raw, ids, _ = prepare_many(demo, size=None, scaling=None)
    rgb, _, _ = prepare_many(demo, size=(224, 224), scaling="unit")
    gray, _, _ = prepare_many(demo, size=(224, 224), mode="gray", scaling="unit")
    n = len(ids)
    captions = [f"raw {raw[0].shape[1]}x{raw[0].shape[0]}"] * n + ["rgb 224x224"] * n + ["gray 224x224"] * n
    image_grid(list(raw) + list(rgb) + list(gray), captions=captions, ncols=n,
               title="Original (top), resized rgb (middle), resized gray (bottom)",
               filename="s2_preprocess_before_after.png")
    _, details = prepare_image(paths.find_image(ids[0]))
    say("one preprocessed image:", details)
    _, kept, failed = prepare_many([paths.find_image(ids[0]), "does_not_exist.jpg"])
    say("batch containing a missing file, kept:", kept, "failed:", failed)

    say("\n########## PART 4  batch loader ##########")
    loader = LesionBatches(df, batch_size=12)
    say(f"{len(loader.df):,} images, {len(loader)} batches, classes: {loader.classes}")
    images, labels, ids = next(iter(loader))
    say("first batch:", images.shape, "labels:", labels.tolist())

    say("\n########## PART 5  visualiser ##########")
    image_grid(images, captions=[loader.classes[i] for i in labels],
               title="First batch from the loader", filename="s2_loader_batch.png")
    say("3 classes:", class_balance_chart(loader.df, filename="s2_balance_3class.png").to_dict())
    say("11 classes:", class_balance_chart(loader.df, "subclass", filename="s2_balance_11class.png").to_dict())
    raw4, _, _ = prepare_many([paths.find_image(i) for i in ids[:4]], size=None, scaling=None)
    batch_diagnostics(images, raw=raw4, filename="s2_batch_diagnostics.png")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "session2_results.txt").write_text("\n".join(transcript))
    print("\nFinished. Figures are in session2/figures, numbers in session2/results/session2_results.txt")


if __name__ == "__main__":
    main()
