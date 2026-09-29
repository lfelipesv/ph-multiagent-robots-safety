import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito colorblind-safe palette
C = {
    "blue":   "#0072B2",  # robot / condition A
    "orange": "#E69F00",  # human / condition B
    "green":  "#009E73",  # safe / good
    "red":    "#D55E00",  # unsafe / violation
    "purple": "#CC79A7",
    "sky":    "#56B4E9",
    "yellow": "#F0E442",
    "grey":   "#999999",
    "black":  "#222222",
}

SAFE_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "safe", ["#B2182B", "#F4A582", "#FDDBC7", "#D9F0D3", "#7FBF7B", "#1B7837"])


def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 150,
        "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
        "axes.labelsize": 12, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 10,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "figure.autolayout": False,
    })


# --- IEEE two-column presets ---------------------------------------------------
COL_W = 3.40   # IEEE single-column width (inches)
DBL_W = 7.16   # IEEE double-column (figure*) width (inches)

def apply_ieee():
    """Publication style sized for IEEE two-column: small fonts, thin chrome, vector out."""
    apply_style()
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.7, "grid.linewidth": 0.4, "lines.linewidth": 1.2,
        "pdf.fonttype": 42, "ps.fonttype": 42,   # editable/embeddable fonts
    })

def fig_single(h=2.3):
    return plt.subplots(figsize=(COL_W, h))

def fig_double(h=3.0, ncols=1):
    return plt.subplots(1, ncols, figsize=(DBL_W, h))


def caption(fig, text, y=-0.02):
    """Add a plain-language 'How to read this' line under a figure."""
    fig.text(0.01, y, text, ha="left", va="top", fontsize=9.5, style="italic", color="#444")


def barlabels(ax, bars, fmt="{:.0f}%", dy=None, fontsize=9.5):
    ymax = ax.get_ylim()[1]
    dy = dy if dy is not None else ymax * 0.015
    for b in bars:
        h = b.get_height()
        if h is None or (isinstance(h, float) and np.isnan(h)):
            continue
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=fontsize)


def errorbars(ax, xs, centers, los, his, color="#222", lw=1.4):
    """Draw asymmetric CI whiskers at given x positions (values in percent)."""
    yerr = np.vstack([np.array(centers) - np.array(los), np.array(his) - np.array(centers)])
    yerr = np.clip(yerr, 0, None)   # Wilson interval isn't centered on the raw p; avoid tiny negatives
    ax.errorbar(xs, centers, yerr=yerr, fmt="none", ecolor=color, elinewidth=lw, capsize=3)


def save(fig, path, cap=None, cap_y=-0.02):
    if cap:
        caption(fig, cap, y=cap_y)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
