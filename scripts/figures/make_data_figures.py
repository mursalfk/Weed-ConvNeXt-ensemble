r"""
make_data_figures.py  --  the three data-driven figures, zero-keystroke.

Paste this whole file into a Jupyter cell and run it. It writes:
    1.1  intro/weed_montage.png     (one real image per species, from IMG_ROOT)
    5.1  eda/dist_species_raw.png   (per-species counts, parsed from the XML)
    5.3a eda/dist_genus_raw.png     (per-genus counts, single panel)
    5.3b eda/dist_family_raw.png    (per-family counts, single panel)

The counts are read straight from the PlantCLEF <Species>/<Genus>/<Family> XML
annotations, exactly as your EDA notebook did, so no pre-existing variables are
needed. Only edit the two paths under CONFIG if your layout differs. The font is
pinned to DejaVu Serif (bundled with matplotlib) so output is identical anywhere.
matplotlib, numpy, and PIL are required.
"""
import os
import numpy as np
import matplotlib  # noqa: F401
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ==================== EDIT THESE IF YOUR LAYOUT DIFFERS ====================
OUT = "figures"                              # base folder for the PNGs
PLANTCLEF = "../../../../PlantCLEF 2015 Dataset"
IMG_ROOT  = f"{PLANTCLEF}/weed_only_dataset/organized_test_dataset"  # 1.1 montage
RAW_TRAIN = f"{PLANTCLEF}/train"             # 5.1 / 5.3 counts (flat folder of .jpg + .xml)
# --------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 11,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
INK = "#1a1a1a"; GREY = "#5f6b78"
BLUE = "#2f6fb0"; GREEN = "#2e8b57"


def _save(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()
    print("wrote", path)


def taxonomy_counts(*candidate_dirs):
    """Parse Species/Genus/Family from the PlantCLEF XML files and return three
    dicts {name: image_count}. Uses the first candidate directory that contains
    .xml files (searched recursively)."""
    import xml.etree.ElementTree as ET
    for root_dir in candidate_dirs:
        if not root_dir or not os.path.isdir(root_dir):
            continue
        xmls = []
        for dp, _, files in os.walk(root_dir):
            xmls += [os.path.join(dp, f) for f in files if f.lower().endswith(".xml")]
        if not xmls:
            continue
        print(f"Parsing {len(xmls)} XML annotations from {root_dir} ...")
        sp, ge, fa = {}, {}, {}
        for i, x in enumerate(xmls):
            try:
                r = ET.parse(x).getroot()
            except Exception:
                continue
            s = (r.findtext("Species") or "Unknown").strip()
            g = (r.findtext("Genus") or "Unknown").strip()
            f = (r.findtext("Family") or "Unknown").strip()
            sp[s] = sp.get(s, 0) + 1
            ge[g] = ge.get(g, 0) + 1
            fa[f] = fa.get(f, 0) + 1
            if (i + 1) % 20000 == 0:
                print(f"  {i + 1}/{len(xmls)} ...")
        print(f"  parsed {sum(sp.values())} images: "
              f"{len(sp)} species, {len(ge)} genera, {len(fa)} families")
        return sp, ge, fa
    return None, None, None


# ======================================================================
# 5.1  single distribution (labels hidden; top-K annotated)
# ======================================================================
def clean_distribution(counts, path, unit="Species", color=BLUE, top_k=5):
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)   # tallest first
    names = [k for k, _ in items]
    values = np.array([v for _, v in items], dtype=float)
    n = len(values); mx = float(values.max())
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.bar(np.arange(n), values, width=1.0, color=color, edgecolor="none", zorder=3)
    ax.set_xlim(-1, n); ax.set_ylim(0, mx * 1.40)                        # headroom for the box
    ax.set_xticks([])
    ax.set_xlabel(f"{unit} ranked from most to least frequent (each bar is one {unit.lower()})")
    ax.set_ylabel("Number of images")
    ax.set_title(f"Per-{unit.lower()} image-count distribution\n"
                 f"{n} {unit.lower()}: min {int(values.min())}, max {int(mx)}, "
                 f"mean {values.mean():.0f}", fontsize=12)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True); ax.yaxis.grid(True, color="0.9", lw=0.8)
    lines = [f"Most populous {unit.lower()}:"] + [
        f"{names[i].replace('_', ' ')}  ({int(values[i])})" for i in range(min(top_k, n))]
    ax.text(0.985, 0.95, "\n".join(lines), transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color=INK,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="0.7"))
    _save(path)


# ======================================================================
# 5.3  genus + family, two panels in one figure
# ======================================================================
def clean_distribution_pair(counts_left, counts_right, path, unit_left="Genus",
                            unit_right="Family", color_left=GREEN, color_right="#b5651d",
                            top_k=5):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, counts, unit, color in ((axes[0], counts_left, unit_left, color_left),
                                    (axes[1], counts_right, unit_right, color_right)):
        items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        names = [k for k, _ in items]
        values = np.array([v for _, v in items], dtype=float)
        n = len(values); mx = float(values.max())
        ax.bar(np.arange(n), values, width=1.0, color=color, edgecolor="none", zorder=3)
        ax.set_xlim(-1, n); ax.set_ylim(0, mx * 1.40)
        ax.set_xticks([])
        ax.set_xlabel(f"{unit} ranked most to least frequent")
        ax.set_ylabel("Number of images")
        ax.set_title(f"Per-{unit.lower()} image counts\n"
                     f"{n} {unit.lower()}: min {int(values.min())}, max {int(mx)}, "
                     f"mean {values.mean():.0f}", fontsize=11.5)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True); ax.yaxis.grid(True, color="0.9", lw=0.8)
        lines = [f"Most populous {unit.lower()}:"] + [
            f"{names[i].replace('_', ' ')}  ({int(values[i])})" for i in range(min(top_k, n))]
        ax.text(0.975, 0.95, "\n".join(lines), transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color=INK,
                bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="0.7"))
    plt.tight_layout()
    _save(path)


# ======================================================================
# 1.1  montage of one real image per species
# ======================================================================
def weed_montage(path, img_root, species=None, ncols=4, size=300):
    default = ["Cirsium_arvense", "Convolvulus_arvensis", "Chenopodium_album",
               "Taraxacum_officinale", "Urtica_dioica", "Daucus_carota",
               "Plantago_lanceolata", "Sonchus_asper"]
    avail = sorted([d for d in os.listdir(img_root)
                    if os.path.isdir(os.path.join(img_root, d))]) if (img_root and os.path.isdir(img_root)) else []

    def find_folder(name):
        for d in avail:
            if d.lower().startswith(name.lower()):
                return d
        return None

    wanted = species or default
    resolved = [(n, find_folder(n)) for n in wanted]
    resolved = [(n, f) for n, f in resolved if f]          # keep those that exist
    if len(resolved) < len(wanted):                        # top up from whatever is available
        used = {f for _, f in resolved}
        for d in avail:
            if d not in used:
                resolved.append((d.replace("_", " "), d))
            if len(resolved) >= len(wanted):
                break
    if not resolved:                                       # no dataset: draw placeholders
        resolved = [(n, None) for n in wanted]

    nrows = int(np.ceil(len(resolved) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.7 * nrows))
    axs = np.atleast_1d(axs).ravel()
    for ax, (label, folder) in zip(axs, resolved):
        img = None
        if folder:
            fdir = os.path.join(img_root, folder)
            files = sorted(f for f in os.listdir(fdir)
                           if f.lower().endswith(("jpg", "jpeg", "png")))
            if files:
                try:
                    from PIL import Image
                    im = Image.open(os.path.join(fdir, files[0])).convert("RGB")
                    w, h = im.size; s = min(w, h)
                    im = im.crop(((w - s) // 2, (h - s) // 2,
                                  (w - s) // 2 + s, (h - s) // 2 + s)).resize((size, size))
                    img = np.asarray(im)
                except Exception as e:
                    print("skip", label, e)
        if img is not None:
            ax.imshow(img)
        else:
            ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                         facecolor="0.93", edgecolor="0.6"))
            ax.text(0.5, 0.5, "image\nunavailable", transform=ax.transAxes,
                    ha="center", va="center", color="0.5", fontsize=9)
        ax.set_title(label.replace("_", " "), fontsize=9.5, style="italic")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("0.7"); sp.set_linewidth(1.0)
    for ax in axs[len(resolved):]:
        ax.axis("off")
    plt.tight_layout()
    _save(path)


# ============================================================
# RUN: generates the 3 data figures. No variables to set.
# ============================================================
weed_montage(f"{OUT}/intro/weed_montage.png", img_root=IMG_ROOT)

species_counts, genus_counts, family_counts = taxonomy_counts(
    RAW_TRAIN, f"{PLANTCLEF}/organized_dataset")
if species_counts:
    # three separate single-panel figures, matching the thesis \includegraphics
    clean_distribution(species_counts, f"{OUT}/eda/dist_species_raw.png",
                       unit="Species", color="#2f6fb0")
    clean_distribution(genus_counts, f"{OUT}/eda/dist_genus_raw.png",
                       unit="Genus", color="#2e8b57")
    clean_distribution(family_counts, f"{OUT}/eda/dist_family_raw.png",
                       unit="Family", color="#b5651d")
else:
    print("Distributions skipped: no XML annotations found under")
    print("  ", RAW_TRAIN, "or", f"{PLANTCLEF}/organized_dataset")
    print("Set RAW_TRAIN at the top to your PlantCLEF 'train' folder and re-run.")
print("done.")
