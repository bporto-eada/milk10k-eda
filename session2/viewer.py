import math

import matplotlib.pyplot as plt
import numpy as np

from shared.paths import LABEL
from figstyle import PALETTE, FALLBACK, CHANNEL_COLOURS, tidy, finish


def to_displayable(img):
    a = np.asarray(img)
    if a.dtype == np.uint8:
        return a / 255.0
    a = a.astype(np.float32)
    if a.min() < 0 or a.max() > 1:
        a = (a - a.min()) / (a.max() - a.min() + 1e-6)
    return a


def image_grid(images, captions=None, ncols=6, title=None, filename="grid.png"):
    nrows = math.ceil(len(images) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.4 * nrows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.set_axis_off()
        if i >= len(images):
            continue
        shown = to_displayable(images[i])
        if shown.ndim == 2:
            ax.imshow(shown, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(shown)
        if captions is not None:
            ax.set_title(str(captions[i]), fontsize=8)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    finish(fig, filename)


def class_balance_chart(df, label_col=LABEL, filename="class_balance.png"):
    counts = df[label_col].value_counts().sort_values()
    total = counts.sum()
    colours = [PALETTE.get(c, FALLBACK) for c in counts.index]

    fig, ax = plt.subplots(figsize=(7, 0.45 * len(counts) + 1.5))
    ax.barh(counts.index.astype(str), counts.values, color=colours, height=0.6)
    for i, v in enumerate(counts.values):
        ax.text(v + total * 0.005, i, f"{v:,}  ({v / total:.1%})", va="center", fontsize=8)
    ax.set_xlim(0, counts.max() * 1.25)
    ax.set_xlabel("images")
    ax.set_title(f"Class balance for {label_col}, largest is {counts.max() / counts.min():.0f}x the smallest",
                 fontsize=10)
    ax.xaxis.grid(True, color="#e6e6e6", linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    finish(fig, filename)
    return counts.sort_values(ascending=False)


def batch_diagnostics(images, raw=None, filename="batch_check.png"):
    x = np.asarray(images, dtype=np.float32)
    k = min(4, len(x)) if raw is not None else 0
    fig = plt.figure(figsize=(2.3 * max(k, 3), 7.5))
    grid = fig.add_gridspec(3, max(k, 1), height_ratios=[1, 1, 1.4], hspace=0.35)

    for i in range(k):
        top = fig.add_subplot(grid[0, i])
        top.imshow(raw[i])
        top.set_axis_off()
        bottom = fig.add_subplot(grid[1, i])
        bottom.imshow(to_displayable(x[i]))
        bottom.set_axis_off()
        if i == 0:
            top.set_title("raw", fontsize=8, loc="left")
            bottom.set_title("processed", fontsize=8, loc="left")

    hist = fig.add_subplot(grid[2, :])
    if x.ndim == 4:
        for c, name in enumerate("RGB"):
            hist.hist(x[..., c].ravel(), bins=60, histtype="step", linewidth=2,
                      color=CHANNEL_COLOURS[name], label=name)
        hist.legend(frameon=False)
    else:
        hist.hist(x.ravel(), bins=60, color=CHANNEL_COLOURS["gray"])
    hist.set_title(f"pixel values in the batch   min {x.min():.2f}   max {x.max():.2f}   mean {x.mean():.2f}",
                   fontsize=9)
    tidy(hist)
    finish(fig, filename)
