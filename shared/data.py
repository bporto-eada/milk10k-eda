import pandas as pd

from . import paths


def load_tables():
    meta = pd.read_csv(paths.METADATA_CSV)
    labels = pd.read_csv(paths.LABELS_CSV)
    labels["subclass"] = labels[paths.SUBCLASSES].idxmax(axis=1)
    return meta.merge(labels[["lesion_id", "subclass"]], on="lesion_id", how="left")


def on_disk(df):
    present = df["isic_id"].map(paths.image_exists)
    return df[present].reset_index(drop=True)
