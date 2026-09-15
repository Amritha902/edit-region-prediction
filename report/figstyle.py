"""Shared figure style. One place, so every figure in the report matches.

Palette validated with the dataviz six-checks at light surface #fcfcfb:
lightness band, chroma floor, CVD separation, normal-vision floor, contrast.
All pass. Do not substitute colours without re-running the validator.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK   = "#26241F"
MUTED = "#6B6862"
FAINT = "#C9C5BC"
SURF  = "#FCFCFB"

# categorical, fixed order — never cycled
CAT = ["#1F6FB2", "#C65A16", "#3D8B5F", "#8B4A93"]
OURS, BASE, THIRD, FOURTH = CAT

# sequential ramps, one hue light to dark
SEQ_COOL = ["#EAF2F9", "#C5DCEE", "#93BEDC", "#5A97C4", "#2B77AC", "#1F6FB2", "#14496F"]
SEQ_WARM = ["#FBEEE4", "#F5D7BE", "#EDB88C", "#E09459", "#D0742E", "#C65A16", "#8A3D0E"]

def use():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": FAINT, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": "#E8E5DE", "grid.linewidth": 0.7,
        "axes.grid": True, "axes.grid.axis": "y",
        "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.frameon": False, "legend.fontsize": 8.5,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
        "axes.titlepad": 8, "axes.titlelocation": "left",
        "figure.dpi": 190,
    })

def title(ax, t, sub=None, wrap=None):
    """Neutral title: name what is plotted. No claims, no em dashes.

    Placed in offset points from the axes' top-left, so a two-line subtitle
    never collides with the title regardless of how tall the panel is.
    """
    import textwrap as _tw
    ax.set_title("")
    n = 0
    if sub:
        if wrap: sub = "\n".join(_tw.wrap(sub, wrap))
        n = sub.count("\n") + 1
        ax.annotate(sub, (0, 1), xycoords="axes fraction", textcoords="offset points",
                    xytext=(0, 7), fontsize=8.2, color=MUTED, va="bottom", ha="left",
                    linespacing=1.4)
    ax.annotate(t, (0, 1), xycoords="axes fraction", textcoords="offset points",
                xytext=(0, 7 + 11.5 * n), fontsize=10, weight="bold", color=INK,
                va="bottom", ha="left")

def figtitle(fig, t, sub=None):
    fig.text(0.008, 0.985, t, fontsize=12.5, weight="bold", color=INK, va="top")
    if sub:
        fig.text(0.008, 0.945, sub, fontsize=9, color=MUTED, va="top")

def nogrid(ax):
    ax.grid(False)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
