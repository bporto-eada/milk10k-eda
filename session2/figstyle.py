from pathlib import Path

import matplotlib.pyplot as plt

FIGURES = Path(__file__).resolve().parent / "figures"
POPUP = False

PALETTE = {"Benign": "#3b6ea5", "Malignant": "#c8553d", "Indeterminate": "#588157"}
FALLBACK = "#9a9a9a"
CHANNEL_COLOURS = {"R": "#c8553d", "G": "#3a8a3a", "B": "#3b6ea5", "gray": "#555555"}


def colour(label):
    return PALETTE.get(str(label), FALLBACK)


def tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.7)
    ax.set_axisbelow(True)


def finish(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=140, bbox_inches="tight")
    if POPUP:
        plt.show()
    plt.close(fig)
