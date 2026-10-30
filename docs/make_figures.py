"""Draw the README banner and results chart, light and dark, from the last run of pipeline/.

Run the pipeline first, then from inside this folder:  python make_figures.py
"""
import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch

DEFECTS = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-Full", "Scratch", "Random"]

# github's own page colours, so the pictures sit flat on the readme
THEMES = {
    "light": dict(bg="#ffffff", card="#f6f8fa", ink="#1f2328", muted="#656d76", line="#d0d7de",
                  good="#8fc59c", fail="#d6453d", forest="#2f6fde", base="#b6bec9"),
    "dark": dict(bg="#0d1117", card="#161b22", ink="#e6edf3", muted="#8d96a0", line="#30363d",
                 good="#5f9f6e", fail="#f0544b", forest="#6a9cf5", base="#4a5260"),
}

plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]


def typical_wafers(maps, labels):
    """For each defect, the single-defect wafer that looks most like the average one of its kind."""
    picks = []
    for i in range(8):
        only_this = np.where((labels[:, i] == 1) & (labels.sum(axis=1) == 1))[0]
        failed = (maps[only_this] == 2).reshape(len(only_this), -1).astype(float)
        mean = failed.mean(axis=0)
        picks.append(only_this[np.argmax(failed @ mean / (np.linalg.norm(failed, axis=1) + 1e-9))])
    return maps[picks]


def banner(wafers, t, out):
    fig = plt.figure(figsize=(16, 5.2), dpi=110, facecolor=t["bg"])
    fig.text(0.04, 0.80, "Wafer defect pattern classification", color=t["ink"], fontsize=30, weight="bold")
    fig.text(0.04, 0.70, "Name every defect pattern on a silicon wafer map, up to four at once.",
             color=t["muted"], fontsize=15)
    cmap = ListedColormap([t["card"], t["good"], t["fail"]])
    for i, (w, name) in enumerate(zip(wafers, DEFECTS)):
        x = 0.04 + i * 0.117
        card = FancyBboxPatch((x, 0.08), 0.105, 0.50, boxstyle="round,pad=0,rounding_size=0.012",
                              transform=fig.transFigure, facecolor=t["card"], edgecolor=t["line"], lw=1, zorder=-1)
        fig.patches.append(card)
        ax = fig.add_axes([x + 0.008, 0.17, 0.089, 0.39])
        ax.imshow(w, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
        ax.axis("off")
        fig.text(x + 0.0525, 0.115, name, color=t["ink"], fontsize=12.5, ha="center", weight="bold")
    fig.savefig(out, facecolor=t["bg"])
    plt.close(fig)


def results(scores, per_count, t, out):
    fig, (left, right) = plt.subplots(1, 2, figsize=(16, 6.2), dpi=110, facecolor=t["bg"],
                                      gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.28))
    for ax in (left, right):
        ax.set_facecolor(t["bg"])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(t["line"])
        ax.tick_params(colors=t["muted"], labelsize=12, length=0)

    # f1 per defect: grey dot the baseline, blue dot the forest, the line between them is what the forest adds
    rows = sorted(scores, key=lambda r: float(r["f1"]))
    y = np.arange(len(rows))
    f1 = np.array([float(r["f1"]) for r in rows])
    base = np.array([float(r["baseline_f1"]) for r in rows])
    left.hlines(y, base, f1, color=t["line"], lw=3, zorder=1)
    left.scatter(base, y, s=110, color=t["base"], zorder=2, label="Logistic regression")
    left.scatter(f1, y, s=150, color=t["forest"], zorder=3, label="Random forest")
    for yi, v, b in zip(y, f1, base):
        left.text(max(v, b) + 0.025, yi, f"{v:.2f}", va="center", color=t["ink"], fontsize=12, weight="bold")
    left.set_yticks(y, [r["defect"] for r in rows], color=t["ink"], fontsize=13)
    left.set_xlim(0, 1.08)
    left.set_xlabel("F1 on the test set", color=t["muted"], fontsize=12)
    left.grid(axis="x", color=t["line"], lw=0.6, alpha=0.6)
    left.set_axisbelow(True)
    left.set_title("F1 per defect", loc="left", color=t["ink"], fontsize=17, weight="bold", pad=40)
    left.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, labelcolor=t["muted"], fontsize=12)

    counts = [r for r in per_count if r["defects_on_wafer"] != "all"]
    share = [float(r["all_eight_right"]) * 100 for r in counts]
    bars = right.bar([r["defects_on_wafer"] for r in counts], share, color=t["forest"], width=0.62)
    for b, v, r in zip(bars, share, counts):
        right.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%", ha="center", color=t["ink"],
                   fontsize=13, weight="bold")
        right.text(b.get_x() + b.get_width() / 2, 4, f"{int(r['wafers']):,}", ha="center",
                   color=t["bg"], fontsize=10.5, alpha=0.9)
    right.set_ylim(0, 112)
    right.set_yticks([])
    right.set_xlabel("defects on the wafer  (in the bar: wafers in test)", color=t["muted"], fontsize=12)
    right.tick_params(axis="x", labelsize=13, labelcolor=t["ink"])
    right.set_title("All 8 answers right, by defects per wafer", loc="left", color=t["ink"], fontsize=17, weight="bold", pad=40)

    fig.savefig(out, facecolor=t["bg"], bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def read_csv(path):
    with open(path, encoding="utf-8") as file:
        return list(csv.DictReader(file))


split = np.load("../pipeline/output/split.npz")
wafers = typical_wafers(split["maps"], split["labels"])
scores = read_csv("../pipeline/output/test_scores.csv")
per_count = read_csv("../pipeline/output/test_per_number_of_defects.csv")

for name, theme in THEMES.items():
    banner(wafers, theme, f"banner-{name}.png")
    results(scores, per_count, theme, f"results-{name}.png")
    print(f"Saved banner-{name}.png, results-{name}.png")
