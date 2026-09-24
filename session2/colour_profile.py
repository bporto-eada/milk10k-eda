import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from shared import paths
from shared.paths import LABEL
from figstyle import colour, tidy, finish, CHANNEL_COLOURS
from preprocess import to_grayscale

CHANNELS = ("R", "G", "B")
STAT_COLUMNS = ["mean_R", "mean_G", "mean_B", "std_R", "std_G", "std_B", "mean_gray"]
BINS = np.arange(257)


def balanced_sample(df, per_group=12, seed=0):
    shuffled = df.sample(frac=1, random_state=seed)
    return shuffled.groupby([LABEL, "image_type"]).head(per_group).reset_index(drop=True)


def share_histogram(values):
    counts, _ = np.histogram(values, bins=BINS)
    return counts / values.size


def profile_image(path):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    gray = to_grayscale(rgb)
    record = {"hist_gray": share_histogram(gray), "mean_gray": float(gray.mean())}
    for index, name in enumerate(CHANNELS):
        channel = rgb[..., index]
        record[f"hist_{name}"] = share_histogram(channel)
        record[f"mean_{name}"] = float(channel.mean())
        record[f"std_{name}"] = float(channel.std())
    return record


def profile_table(sample):
    rows = []
    for _, r in sample.iterrows():
        try:
            record = profile_image(paths.find_image(r["isic_id"]))
        except Exception as err:
            print("skipped", r["isic_id"], err)
            continue
        rows.append({LABEL: r[LABEL], "image_type": r["image_type"], **record})
    return pd.DataFrame(rows)


def mean_curve(frame, key):
    return np.vstack(frame[key].to_list()).mean(axis=0)


def chart_histograms(t):
    classes = list(t[LABEL].value_counts().index)

    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8), sharey=True)
    for ax, key in zip(axes, ["hist_gray", "hist_R", "hist_G", "hist_B"]):
        for cls in classes:
            ax.plot(mean_curve(t[t[LABEL] == cls], key), color=colour(cls), linewidth=2, label=cls)
        ax.set_title(key.replace("hist_", "").replace("gray", "grayscale"), fontsize=10)
        ax.set_xlabel("intensity (0 to 255)")
        tidy(ax)
    axes[0].set_ylabel("share of pixels")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Average intensity histogram per class", fontsize=11)
    finish(fig, "s2_colour_histograms.png")

    kinds = sorted(t["image_type"].unique())
    fig, axes = plt.subplots(1, len(kinds), figsize=(6 * len(kinds), 3.8), sharey=True, squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        for cls in classes:
            part = t[(t[LABEL] == cls) & (t["image_type"] == kind)]
            if len(part):
                ax.plot(mean_curve(part, "hist_gray"), color=colour(cls), linewidth=2, label=cls)
        ax.set_title(kind, fontsize=10)
        ax.set_xlabel("grayscale intensity")
        tidy(ax)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Grayscale histogram per class, one panel per image type", fontsize=11)
    finish(fig, "s2_colour_histograms_by_type.png")


def chart_statistics(t):
    classes = list(t[LABEL].value_counts().index)
    fig, axes = plt.subplots(1, len(STAT_COLUMNS), figsize=(16, 3.6))
    for ax, stat in zip(axes, STAT_COLUMNS):
        box = ax.boxplot([t.loc[t[LABEL] == c, stat] for c in classes], patch_artist=True)
        for patch, c in zip(box["boxes"], classes):
            patch.set_facecolor(colour(c))
            patch.set_alpha(0.55)
        ax.set_xticks(range(1, len(classes) + 1), [c[:5] for c in classes], fontsize=7)
        ax.set_title(stat, fontsize=9)
        tidy(ax)
    fig.suptitle("Per image colour statistics, grouped by class", fontsize=11)
    finish(fig, "s2_colour_statistics.png")


def run(df, say, per_group=12):
    t = profile_table(balanced_sample(df, per_group))
    say(f"images profiled: {len(t)}")
    say(pd.crosstab(t[LABEL], t["image_type"]).to_string())

    chart_histograms(t)
    chart_statistics(t)

    say("\nAverage colour statistics per class:")
    say(t.groupby(LABEL)[STAT_COLUMNS].mean().round(1).to_string())
    say("\nAverage colour per image type and class, compare the type gap with the class gap:")
    say(t.groupby(["image_type", LABEL])[["mean_R", "mean_G", "mean_B", "mean_gray"]]
        .mean().round(1).to_string())
    return t
