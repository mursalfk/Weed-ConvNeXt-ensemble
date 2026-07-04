r"""
make_clean_figures.py  --  clean, overlap-free schematic diagrams.

Paste this whole file into a Jupyter cell and run it. It writes seven PNGs under
OUT/ (matching the \includegraphics paths in the thesis). The font is pinned to
DejaVu Serif, which ships with matplotlib, so the output is identical on every
machine and text always fits its box. Only matplotlib and numpy are needed.

    3.1  related/approaches_taxonomy.png
    4.1  approach/ensemble_pipeline.png
    4.2  approach/architecture.png
    4.5  approach/fusion_levels.png
    5.14 dataset/dataset_funnel.png
    5.15 dataset/train_test_split.png
    6.2  progression/evaluation_integrity.png
"""
import os
import numpy as np
import matplotlib  # noqa: F401
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon

# ==================== EDIT THESE TWO, THEN RUN THE CELL ====================
OUT = "figures"          # base folder for the PNGs (matches your \includegraphics paths)
THUMB_IMG = None         # Fig 4.1 thumbnail: path to a real weed image, or None for a placeholder
# --------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],   # bundled with matplotlib -> identical everywhere
    "font.size": 11,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

INK = "#1a1a1a"; GREY = "#5f6b78"; LINE = "#7a828c"
BLUE = "#2f6fb0";  BLUEF  = "#dce8f5"
GREEN = "#2e8b57"; GREENF = "#d9efe1"
AMBER = "#c1820c"; AMBERF = "#f7e6c2"
RED = "#b03a3a";   REDF   = "#f2dede"
NEUT = "#8a939c";  NEUTF  = "#e4e8ee"


def box(ax, cx, cy, w, h, text, fc=NEUTF, ec=GREY, fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.09",
                 linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=INK,
            zorder=4, fontweight="bold" if bold else "normal")
    return (cx, cy, w, h)


def side(b, s):
    cx, cy, w, h = b
    return {"top": (cx, cy + h / 2), "bottom": (cx, cy - h / 2),
            "left": (cx - w / 2, cy), "right": (cx + w / 2, cy)}[s]


def arrow(ax, p1, p2, color=LINE, lw=1.7, rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                 lw=lw, color=color, zorder=2, shrinkA=2, shrinkB=4,
                 connectionstyle=f"arc3,rad={rad}"))


def connect(ax, b1, s1, b2, s2, **kw):
    arrow(ax, side(b1, s1), side(b2, s2), **kw)


def finish(fig, ax, path, xlim, ylim):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal"); ax.axis("off")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ======================================================================
# 4.2  architecture
# ======================================================================
def fig_architecture(path):
    rows = [
        ("Input image", "224 x 224 x 3", NEUTF, GREY, True),
        ("ConvNeXt-Large backbone\n(ImageNet, top 30% fine-tuned)", "7 x 7 x 1536", BLUEF, BLUE, True),
        ("Global average pooling", "1536", GREENF, GREEN, False),
        ("Dense + ReLU", "512", AMBERF, AMBER, False),
        ("BatchNorm + Dropout 0.5", "512", NEUTF, GREY, False),
        ("Dense + ReLU", "256", AMBERF, AMBER, False),
        ("BatchNorm + Dropout 0.3", "256", NEUTF, GREY, False),
        ("Dense + softmax", "94", REDF, RED, True),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 10.4))
    cx, w = 3.2, 5.2
    gap = 1.5
    top = len(rows) * gap
    boxes = []
    for i, (name, shape, fc, ec, bold) in enumerate(rows):
        cy = top - i * gap
        h = 1.0 if "backbone" not in name else 1.2
        b = box(ax, cx, cy, w, h, name, fc, ec, fs=10.5 if bold else 10, bold=bold)
        boxes.append(b)
        ax.text(cx + w / 2 + 0.55, cy, shape, ha="left", va="center",
                fontsize=11, family="DejaVu Sans Mono", color=INK)
    for i in range(len(boxes) - 1):
        connect(ax, boxes[i], "bottom", boxes[i + 1], "top")
    ax.text(cx, top + 0.95, "Layer", ha="center", fontsize=10.5, style="italic", color=GREY)
    ax.text(cx + w / 2 + 0.55, top + 0.95, "Output shape", ha="left", fontsize=10.5,
            style="italic", color=GREY)
    finish(fig, ax, path, (0.2, cx + w / 2 + 3.0), (top - (len(rows) - 1) * gap - 1.1, top + 1.5))


# ======================================================================
# 4.1  ensemble_pipeline   (generous boxes; edge-to-edge arrows)
# ======================================================================
def fig_ensemble_pipeline(path):
    fig, ax = plt.subplots(figsize=(12.5, 6.6))
    xin, xfold, xfus, xout = 1.6, 5.7, 9.8, 12.6
    ymid = 5.0
    tw = 2.0

    img = None
    if THUMB_IMG and os.path.exists(THUMB_IMG):
        try:
            img = plt.imread(THUMB_IMG)
        except Exception:
            img = None
    if img is not None:
        ax.imshow(img, extent=(xin - tw / 2, xin + tw / 2, ymid - tw / 2, ymid + tw / 2), zorder=3)
        ax.add_patch(Rectangle((xin - tw / 2, ymid - tw / 2), tw, tw, fill=False,
                     edgecolor=GREY, lw=1.4, zorder=4))
    else:
        box(ax, xin, ymid, tw, tw, "weed\nimage", NEUTF, GREY, fs=11)
    ax.text(xin, ymid - tw / 2 - 0.4, "weed image\n224 x 224", ha="center",
            va="top", fontsize=9.5, color=GREY)
    in_b = (xin, ymid, tw, tw)

    ys = np.linspace(8.5, 1.5, 5)
    fw, fh = 3.6, 1.05
    fus = (xfus, ymid, 3.1, 2.1)
    for i, y in enumerate(ys):
        fb = box(ax, xfold, y, fw, fh, f"ConvNeXt-Large\nfold model {i + 1}", BLUEF, BLUE, fs=10)
        connect(ax, in_b, "right", fb, "left")
        connect(ax, fb, "right", fus, "left")

    box(ax, xfus, ymid, 3.1, 2.1, "Decision-level\nfusion\n(soft vote / Borda)",
        AMBERF, AMBER, fs=10, bold=True)
    out_b = box(ax, xout, ymid, 2.4, 1.2, "predicted\nspecies", GREENF, GREEN, fs=10.5, bold=True)
    connect(ax, fus, "right", out_b, "left")

    ax.text(xfold, 9.5, "Stage 1: five ConvNeXt-Large models, one per CV fold",
            ha="center", fontsize=10, style="italic", color=GREY)
    ax.text((xfus + xout) / 2, 0.6, "Stage 2: combine the five outputs",
            ha="center", fontsize=10, style="italic", color=GREY)
    finish(fig, ax, path, (0.2, 14.0), (0.1, 10.0))


# ======================================================================
# 3.1  approaches_taxonomy   (wider boxes, short lines)
# ======================================================================
def fig_approaches_taxonomy(path):
    fig, ax = plt.subplots(figsize=(12.5, 6.6))
    W, H = 3.4, 1.2
    root = box(ax, 7.0, 9.0, 3.6, H, "Plant identification\nfrom images", NEUTF, GREY, bold=True)

    hand = box(ax, 3.4, 6.4, W, H, "Handcrafted\nfeatures", NEUTF, GREY)
    deep = box(ax, 10.6, 6.4, W, H, "Deep learning\n(CNN / Transformer)", BLUEF, BLUE, fs=10, bold=True)
    connect(ax, root, "bottom", hand, "top")
    connect(ax, root, "bottom", deep, "top")

    desc = box(ax, 1.7, 3.8, W, H, "Shape, texture,\ncolour descriptors", NEUTF, GREY, fs=9.5)
    clf = box(ax, 5.3, 3.8, W, H, "Classical\nclassifiers", NEUTF, GREY, fs=9.5)
    connect(ax, hand, "bottom", desc, "top")
    connect(ax, hand, "bottom", clf, "top")

    single = box(ax, 8.9, 3.8, W, H, "Single network\n(transfer learning)", BLUEF, BLUE, fs=9.5)
    ens = box(ax, 12.5, 3.8, W, H, "Ensemble of\nnetworks", BLUEF, BLUE, fs=9.5, bold=True)
    connect(ax, deep, "bottom", single, "top")
    connect(ax, deep, "bottom", ens, "top")

    fus = box(ax, 12.5, 1.4, W, H, "Decision-level\nfusion", GREENF, GREEN, fs=9.5)
    connect(ax, ens, "bottom", fus, "top")

    work = box(ax, 8.2, 0.2, 4.8, H, "This work:\n5-fold ConvNeXt ensemble", AMBERF, AMBER, fs=10, bold=True)
    arrow(ax, side(work, "right"), side(fus, "bottom"), color=AMBER, lw=1.7, rad=-0.18)
    finish(fig, ax, path, (-0.3, 14.5), (-0.5, 10.1))


# ======================================================================
# 4.5  fusion_levels   (both panels fully visible)
# ======================================================================
def fig_fusion_levels(path):
    fig, ax = plt.subplots(figsize=(12.0, 7.2))

    def three_models(x, y0, dy):
        return [box(ax, x, y0 - i * dy, 2.1, 0.85, f"model {i+1}", BLUEF, BLUE, fs=10) for i in range(3)]

    # top: decision-level
    ax.text(0.3, 9.5, "Decision-level fusion (used in this work)", ha="left",
            fontsize=12, style="italic", color=INK)
    m = three_models(1.9, 8.8, 1.15)
    outs = [box(ax, 5.1, 8.8 - i * 1.15, 2.0, 0.85, "class\nprobabilities", GREENF, GREEN, fs=9)
            for i in range(3)]
    for a, b in zip(m, outs):
        connect(ax, a, "right", b, "left")
    comb = box(ax, 8.3, 7.65, 2.3, 1.6, "combine\n(soft vote /\nBorda)", AMBERF, AMBER, fs=10, bold=True)
    for b in outs:
        connect(ax, b, "right", comb, "left")
    predm = box(ax, 11.3, 7.65, 2.0, 1.1, "prediction", NEUTF, GREY, fs=10, bold=True)
    connect(ax, comb, "right", predm, "left")

    # bottom: feature-level
    ax.text(0.3, 4.7, "Feature-level fusion (proposed as future work)", ha="left",
            fontsize=12, style="italic", color=INK)
    m2 = three_models(1.9, 4.0, 1.15)
    feats = [box(ax, 5.1, 4.0 - i * 1.15, 2.0, 0.85, "feature\nvector", BLUEF, BLUE, fs=9)
             for i in range(3)]
    for a, b in zip(m2, feats):
        connect(ax, a, "right", b, "left")
    concat = box(ax, 8.3, 2.85, 2.3, 1.6, "concatenate\n+ classifier", AMBERF, AMBER, fs=10, bold=True)
    for b in feats:
        connect(ax, b, "right", concat, "left")
    predf = box(ax, 11.3, 2.85, 2.0, 1.1, "prediction", NEUTF, GREY, fs=10, bold=True)
    connect(ax, concat, "right", predf, "left")

    ax.plot([0.2, 12.6], [5.75, 5.75], color="0.8", lw=1.0, ls="--", zorder=1)
    finish(fig, ax, path, (-0.1, 12.8), (1.1, 10.0))


# ======================================================================
# 5.14  dataset_funnel   (stronger colours)
# ======================================================================
def fig_dataset_funnel(path):
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    stages = [
        ("PlantCLEF 2015\nfull collection", "1,000 species", "#c9d2dd", "#5f6b78"),
        ("Present in both\ntrain and test", "654 species", "#9ec4e8", "#2f6fb0"),
        ("Agriculturally\nrelevant weeds", "94 species", "#efc670", "#c1820c"),
    ]
    yh = 1.5
    ys = [7.0, 4.6, 2.2]
    widths = [8.2, 5.6, 3.2]
    cx = 4.6
    for i, ((title, count, fc, ec), y, w) in enumerate(zip(stages, ys, widths)):
        if i < len(stages) - 1:
            w2 = widths[i + 1]; y2 = ys[i + 1]
            ax.add_patch(Polygon([(cx - w / 2, y - yh / 2), (cx + w / 2, y - yh / 2),
                                  (cx + w2 / 2, y2 + yh / 2), (cx - w2 / 2, y2 + yh / 2)],
                                 closed=True, facecolor="0.92", edgecolor="none", zorder=1))
        ax.add_patch(FancyBboxPatch((cx - w / 2, y - yh / 2), w, yh,
                     boxstyle="round,pad=0.02,rounding_size=0.08",
                     linewidth=1.8, edgecolor=ec, facecolor=fc, zorder=3))
        ax.text(cx, y + 0.2, title, ha="center", va="center", fontsize=11.5,
                color=INK, zorder=4, fontweight="bold")
        ax.text(cx, y - 0.34, count, ha="center", va="center", fontsize=10.5,
                color=ec, zorder=4, fontweight="bold")
    finish(fig, ax, path, (-0.2, 9.4), (1.0, 8.2))


# ======================================================================
# 5.15  train_test_split   (labels sit clear of the bars)
# ======================================================================
def fig_train_test_split(path):
    fig, ax = plt.subplots(figsize=(11.0, 3.8))
    total = 29947 + 2803
    tr = 29947 / total * 10.0
    te = 2803 / total * 10.0
    y, h = 1.6, 1.1
    ax.add_patch(FancyBboxPatch((0, y - h / 2), tr, h, boxstyle="round,pad=0.01,rounding_size=0.05",
                 facecolor=BLUEF, edgecolor=BLUE, linewidth=1.6, zorder=3))
    ax.add_patch(FancyBboxPatch((tr + 0.08, y - h / 2), te, h, boxstyle="round,pad=0.01,rounding_size=0.05",
                 facecolor=AMBERF, edgecolor=AMBER, linewidth=1.6, zorder=3))
    ax.text(tr / 2, y, "Training pool\n29,947 images  (91.4%)", ha="center", va="center",
            fontsize=11.5, color=INK, zorder=4, fontweight="bold")
    ax.text(tr + 0.08 + te / 2, y - h / 2 - 0.45, "Held-out test\n2,803 images (8.6%)",
            ha="center", va="top", fontsize=10, color=INK, zorder=4)
    # callout above the training bar
    cbx, cby, cbw, cbh = 3.0, y + h / 2 + 1.05, 5.6, 0.85
    ax.add_patch(FancyBboxPatch((cbx - cbw / 2, cby - cbh / 2), cbw, cbh,
                 boxstyle="round,pad=0.02,rounding_size=0.06",
                 facecolor=GREENF, edgecolor=GREEN, linewidth=1.4, zorder=3))
    ax.text(cbx, cby, "class-balanced CV subset (~5,900)\ndrawn from the training pool",
            ha="center", va="center", fontsize=10, color=INK, zorder=4)
    arrow(ax, (cbx, cby - cbh / 2), (cbx, y + h / 2 + 0.02), color=GREEN, lw=1.5)
    finish(fig, ax, path, (-0.4, 11.2), (-0.3, 4.2))


# ======================================================================
# 6.2  evaluation_integrity
# ======================================================================
def fig_evaluation_integrity(path):
    fig, ax = plt.subplots(figsize=(7.8, 5.1))
    labels = ["Validation\n(5-fold mean)", "Held-out test\n(misaligned eval)",
              "Held-out test\n(aligned eval)"]
    vals = [0.7094, 0.42, 0.6011]
    faces = [NEUTF, REDF, GREENF]; edges = [GREY, RED, GREEN]
    xs = np.arange(3)
    for x, v, fc, ec in zip(xs, vals, faces, edges):
        ax.bar(x, v, width=0.6, color=fc, edgecolor=ec, linewidth=1.6, zorder=3)
        ax.text(x, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=11, zorder=4)
    ax.annotate("", xy=(2, 0.66), xytext=(1, 0.66),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6,
                                connectionstyle="arc3,rad=-0.35"), zorder=5)
    ax.text(1.5, 0.735, "+0.18 by correcting the\nevaluation (no retraining)",
            ha="center", va="bottom", fontsize=9.5, color=INK, zorder=5)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 0.86)
    ax.set_title("Effect of correcting the evaluation pipeline\n(InceptionV3 ensemble, soft voting)",
                 fontsize=11.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True); ax.yaxis.grid(True, color="0.9", lw=0.8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ============================================================
# RUN: generates all 7 schematic figures.
# ============================================================
fig_architecture(f"{OUT}/approach/architecture.png")
fig_ensemble_pipeline(f"{OUT}/approach/ensemble_pipeline.png")
fig_approaches_taxonomy(f"{OUT}/related/approaches_taxonomy.png")
fig_fusion_levels(f"{OUT}/approach/fusion_levels.png")
fig_dataset_funnel(f"{OUT}/dataset/dataset_funnel.png")
fig_train_test_split(f"{OUT}/dataset/train_test_split.png")
fig_evaluation_integrity(f"{OUT}/progression/evaluation_integrity.png")
print("done: 7 figures under", OUT + "/")
