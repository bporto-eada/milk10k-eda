"""Shared figure style (moved here from session2/ so the project code can reuse it).

FIGURES is the folder finish() saves into. It defaults to outputs/figures;
session2/run_session2.py points it at session2/figures before it draws.
"""

import matplotlib.pyplot as plt

from .config import FIGURES_DIR

FIGURES = FIGURES_DIR
POPUP = False   # True: also show each figure (the notebook sets this)

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
    """Save fig as FIGURES/name, show it if POPUP is on, then close it."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=140, bbox_inches="tight")
    if POPUP:
        plt.show()
    plt.close(fig)
