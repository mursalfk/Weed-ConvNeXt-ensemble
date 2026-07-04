#!/usr/bin/env python3
"""
Builds appendix_species.tex with columns:
    #  |  Species  |  Common name  |  Family  |  Images

- Taxonomy (family, scientific name) is baked in from your dataset CSV.
- Common names are supplied below (verify the ones flagged in chat).
- Image counts are read from your dataset: set DATASET_DIR to the folder whose
  per-species image totals you want (point it at ONE folder that holds each
  original image once; a folder with .xml files next to images). The script
  counts one image per .xml. If DATASET_DIR is missing, the Images column is
  filled with "--" so you can still see the layout, then rerun once it is set.
"""

import os
import xml.etree.ElementTree as ET

DATASET_DIR = r"weeds_only_dataset/organized_train_dataset"   # <-- set this

# (Family, "Genus epithet Authority") straight from your CSV.
SPECIES_DATA = [
    ("Asteraceae", "Achillea millefolium L."),
    ("Apiaceae", "Aegopodium podagraria L."),
    ("Caryophyllaceae", "Agrostemma githago L."),
    ("Simaroubaceae", "Ailanthus altissima (Mill.) Swingle"),
    ("Brassicaceae", "Alliaria petiolata (M.Bieb.) Cavara & Grande"),
    ("Amaranthaceae", "Amaranthus deflexus L."),
    ("Asteraceae", "Ambrosia artemisiifolia L."),
    ("Asteraceae", "Arctium lappa L."),
    ("Asteraceae", "Arctium minus (Hill) Bernh."),
    ("Asteraceae", "Artemisia vulgaris L."),
    ("Lamiaceae", "Ballota nigra L."),
    ("Brassicaceae", "Barbarea vulgaris R.Br."),
    ("Asteraceae", "Bidens pilosa L."),
    ("Brassicaceae", "Capsella bursa-pastoris (L.) Medik."),
    ("Asteraceae", "Carduus acanthoides L."),
    ("Asteraceae", "Carduus nutans L."),
    ("Asteraceae", "Carduus pycnocephalus L."),
    ("Amaranthaceae", "Chenopodium album L."),
    ("Asteraceae", "Chondrilla juncea L."),
    ("Asteraceae", "Cichorium intybus L."),
    ("Asteraceae", "Cirsium arvense (L.) Scop."),
    ("Asteraceae", "Cirsium vulgare (Savi) Ten."),
    ("Convolvulaceae", "Convolvulus arvensis L."),
    ("Solanaceae", "Datura stramonium L."),
    ("Apiaceae", "Daucus carota L."),
    ("Boraginaceae", "Echium vulgare L."),
    ("Onagraceae", "Epilobium angustifolium L."),
    ("Asteraceae", "Erigeron canadensis L."),
    ("Euphorbiaceae", "Euphorbia cyparissias L."),
    ("Euphorbiaceae", "Euphorbia peplus L."),
    ("Papaveraceae", "Fumaria officinalis L."),
    ("Asteraceae", "Galinsoga quadriradiata Ruiz & Pav."),
    ("Rubiaceae", "Galium aparine L."),
    ("Geraniaceae", "Geranium molle L."),
    ("Geraniaceae", "Geranium robertianum L."),
    ("Asteraceae", "Hypochaeris radicata L."),
    ("Balsaminaceae", "Impatiens glandulifera Royle"),
    ("Asteraceae", "Lactuca serriola L."),
    ("Lamiaceae", "Lamium purpureum L."),
    ("Asteraceae", "Lapsana communis L."),
    ("Brassicaceae", "Lepidium draba L."),
    ("Plantaginaceae", "Linaria vulgaris Mill."),
    ("Fabaceae", "Lotus corniculatus L."),
    ("Lythraceae", "Lythrum salicaria L."),
    ("Malvaceae", "Malva neglecta Wallr."),
    ("Asteraceae", "Matricaria chamomilla L."),
    ("Fabaceae", "Medicago lupulina L."),
    ("Fabaceae", "Medicago sativa L."),
    ("Fabaceae", "Melilotus albus Medik."),
    ("Euphorbiaceae", "Mercurialis annua L."),
    ("Boraginaceae", "Myosotis arvensis Hill"),
    ("Asteraceae", "Onopordum acanthium L."),
    ("Oxalidaceae", "Oxalis corniculata L."),
    ("Papaveraceae", "Papaver rhoeas L."),
    ("Urticaceae", "Parietaria judaica L."),
    ("Plantaginaceae", "Plantago lanceolata L."),
    ("Plantaginaceae", "Plantago major L."),
    ("Portulacaceae", "Portulaca oleracea L."),
    ("Rosaceae", "Potentilla reptans L."),
    ("Lamiaceae", "Prunella vulgaris L."),
    ("Ranunculaceae", "Ranunculus acris L."),
    ("Ranunculaceae", "Ranunculus repens L."),
    ("Brassicaceae", "Raphanus raphanistrum L."),
    ("Resedaceae", "Reseda lutea L."),
    ("Polygonaceae", "Reynoutria japonica Houtt."),
    ("Fabaceae", "Robinia pseudoacacia L."),
    ("Rosaceae", "Rubus fruticosus L."),
    ("Polygonaceae", "Rumex acetosella L."),
    ("Polygonaceae", "Rumex crispus L."),
    ("Polygonaceae", "Rumex obtusifolius L."),
    ("Caryophyllaceae", "Saponaria officinalis L."),
    ("Asteraceae", "Senecio inaequidens DC."),
    ("Asteraceae", "Senecio vulgaris L."),
    ("Caryophyllaceae", "Silene vulgaris (Moench) Garcke"),
    ("Brassicaceae", "Sinapis arvensis L."),
    ("Brassicaceae", "Sisymbrium irio L."),
    ("Brassicaceae", "Sisymbrium officinale (L.) Scop."),
    ("Solanaceae", "Solanum nigrum L."),
    ("Asteraceae", "Sonchus asper (L.) Hill"),
    ("Asteraceae", "Sonchus tenerrimus L."),
    ("Caryophyllaceae", "Stellaria media (L.) Vill."),
    ("Asteraceae", "Taraxacum officinale F.H.Wigg."),
    ("Apiaceae", "Torilis arvensis (Huds.) Link"),
    ("Fabaceae", "Trifolium repens L."),
    ("Asteraceae", "Tripleurospermum inodorum Sch.Bip."),
    ("Asteraceae", "Tussilago farfara L."),
    ("Urticaceae", "Urtica dioica L."),
    ("Scrophulariaceae", "Verbascum thapsus L."),
    ("Plantaginaceae", "Veronica arvensis L."),
    ("Plantaginaceae", "Veronica persica Poir."),
    ("Fabaceae", "Vicia hirsuta (L.) Gray"),
    ("Fabaceae", "Vicia sativa L."),
    ("Apocynaceae", "Vinca minor L."),
    ("Violaceae", "Viola arvensis Murray"),
]

# scientific name -> English common name  (flagged ones noted in chat)
COMMON = {
    "Achillea millefolium": "Yarrow",
    "Aegopodium podagraria": "Ground elder",
    "Agrostemma githago": "Corncockle",
    "Ailanthus altissima": "Tree of heaven",
    "Alliaria petiolata": "Garlic mustard",
    "Amaranthus deflexus": "Perennial pigweed",
    "Ambrosia artemisiifolia": "Common ragweed",
    "Arctium lappa": "Greater burdock",
    "Arctium minus": "Lesser burdock",
    "Artemisia vulgaris": "Mugwort",
    "Ballota nigra": "Black horehound",
    "Barbarea vulgaris": "Winter-cress",
    "Bidens pilosa": "Hairy beggarticks",
    "Capsella bursa-pastoris": "Shepherd's purse",
    "Carduus acanthoides": "Spiny plumeless thistle",
    "Carduus nutans": "Musk thistle",
    "Carduus pycnocephalus": "Italian thistle",
    "Chenopodium album": "Fat hen",
    "Chondrilla juncea": "Rush skeletonweed",
    "Cichorium intybus": "Chicory",
    "Cirsium arvense": "Creeping thistle",
    "Cirsium vulgare": "Spear thistle",
    "Convolvulus arvensis": "Field bindweed",
    "Datura stramonium": "Jimsonweed",
    "Daucus carota": "Wild carrot",
    "Echium vulgare": "Viper's bugloss",
    "Epilobium angustifolium": "Rosebay willowherb",
    "Erigeron canadensis": "Canadian horseweed",
    "Euphorbia cyparissias": "Cypress spurge",
    "Euphorbia peplus": "Petty spurge",
    "Fumaria officinalis": "Common fumitory",
    "Galinsoga quadriradiata": "Shaggy soldier",
    "Galium aparine": "Cleavers",
    "Geranium molle": "Dove's-foot crane's-bill",
    "Geranium robertianum": "Herb Robert",
    "Hypochaeris radicata": "Catsear",
    "Impatiens glandulifera": "Himalayan balsam",
    "Lactuca serriola": "Prickly lettuce",
    "Lamium purpureum": "Red dead-nettle",
    "Lapsana communis": "Nipplewort",
    "Lepidium draba": "Hoary cress",
    "Linaria vulgaris": "Common toadflax",
    "Lotus corniculatus": "Bird's-foot trefoil",
    "Lythrum salicaria": "Purple loosestrife",
    "Malva neglecta": "Common mallow",
    "Matricaria chamomilla": "Scented mayweed",
    "Medicago lupulina": "Black medick",
    "Medicago sativa": "Alfalfa",
    "Melilotus albus": "White sweet-clover",
    "Mercurialis annua": "Annual mercury",
    "Myosotis arvensis": "Field forget-me-not",
    "Onopordum acanthium": "Cotton thistle",
    "Oxalis corniculata": "Creeping wood sorrel",
    "Papaver rhoeas": "Common poppy",
    "Parietaria judaica": "Pellitory-of-the-wall",
    "Plantago lanceolata": "Ribwort plantain",
    "Plantago major": "Broadleaf plantain",
    "Portulaca oleracea": "Common purslane",
    "Potentilla reptans": "Creeping cinquefoil",
    "Prunella vulgaris": "Selfheal",
    "Ranunculus acris": "Meadow buttercup",
    "Ranunculus repens": "Creeping buttercup",
    "Raphanus raphanistrum": "Wild radish",
    "Reseda lutea": "Wild mignonette",
    "Reynoutria japonica": "Japanese knotweed",
    "Robinia pseudoacacia": "Black locust",
    "Rubus fruticosus": "Blackberry",
    "Rumex acetosella": "Sheep's sorrel",
    "Rumex crispus": "Curled dock",
    "Rumex obtusifolius": "Broad-leaved dock",
    "Saponaria officinalis": "Soapwort",
    "Senecio inaequidens": "Narrow-leaved ragwort",
    "Senecio vulgaris": "Common groundsel",
    "Silene vulgaris": "Bladder campion",
    "Sinapis arvensis": "Charlock",
    "Sisymbrium irio": "London rocket",
    "Sisymbrium officinale": "Hedge mustard",
    "Solanum nigrum": "Black nightshade",
    "Sonchus asper": "Prickly sow-thistle",
    "Sonchus tenerrimus": "Slender sow-thistle",
    "Stellaria media": "Common chickweed",
    "Taraxacum officinale": "Dandelion",
    "Torilis arvensis": "Spreading hedge-parsley",
    "Trifolium repens": "White clover",
    "Tripleurospermum inodorum": "Scentless mayweed",
    "Tussilago farfara": "Coltsfoot",
    "Urtica dioica": "Stinging nettle",
    "Verbascum thapsus": "Great mullein",
    "Veronica arvensis": "Wall speedwell",
    "Veronica persica": "Common field speedwell",
    "Vicia hirsuta": "Hairy tare",
    "Vicia sativa": "Common vetch",
    "Vinca minor": "Lesser periwinkle",
    "Viola arvensis": "Field pansy",
}


def tex_escape(s):
    for a, b in [("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")]:
        s = s.replace(a, b)
    return s


def scientific(raw):
    p = raw.split()
    return f"{p[0]} {p[1]}" if len(p) > 1 else p[0]


def count_images(dataset_dir):
    """Count one image per .xml, keyed by lowercase 'genus epithet'."""
    counts = {}
    if not (dataset_dir and os.path.isdir(dataset_dir)):
        return counts
    for root, _, files in os.walk(dataset_dir):
        for fn in files:
            if not fn.lower().endswith(".xml"):
                continue
            try:
                node = ET.parse(os.path.join(root, fn)).getroot()
            except ET.ParseError:
                continue
            rec = {c.tag: (c.text or "").strip() for c in node}
            genus = rec.get("Genus", "")
            sp = rec.get("Species", "").split()
            epithet = sp[1] if len(sp) > 1 and sp[0].lower() == genus.lower() else (sp[1] if len(sp) > 1 else "")
            if genus and epithet:
                key = f"{genus} {epithet}".lower()
                counts[key] = counts.get(key, 0) + 1
    return counts


def main():
    counts = count_images(DATASET_DIR)
    have_counts = bool(counts)

    rows = []
    for family, raw in SPECIES_DATA:
        sci = scientific(raw)
        common = COMMON.get(sci, "")
        n = counts.get(sci.lower(), 0) if have_counts else "--"
        rows.append((sci, common, family, n))
    rows.sort(key=lambda r: r[0].lower())

    body = "\n".join(
        f"  {i} & \\emph{{{sci}}} & {tex_escape(common)} & {fam} & {n} \\\\"
        for i, (sci, common, fam, n) in enumerate(rows, start=1)
    )
    n_fam = len({r[2] for r in rows})
    n_gen = len({r[0].split()[0] for r in rows})

    tex = f"""\\chapter{{The Selected Weed Species}}
\\label{{app:species}}

This appendix lists the 94 weed species used in this thesis, with their English
common name, botanical family, and the number of images available per species,
as selected from the PlantCLEF~2015 collection in consultation with a domain
expert (Chapter~\\ref{{ch:datasets}}). The species span {n_fam} families and
{n_gen} genera, and are listed alphabetically by scientific name in
Table~\\ref{{tab:species-list}}.

\\begin{{center}}
\\begin{{longtable}}{{r l l l r}}
\\caption{{The 94 weed species, with common name, family, and number of images,
ordered alphabetically by scientific name.}}\\label{{tab:species-list}}\\\\
\\toprule
\\textbf{{\\#}} & \\textbf{{Species}} & \\textbf{{Common name}} & \\textbf{{Family}} & \\textbf{{Images}} \\\\
\\midrule
\\endfirsthead
\\multicolumn{{5}}{{c}}{{\\tablename\\ \\thetable{{}} -- continued from previous page}} \\\\
\\toprule
\\textbf{{\\#}} & \\textbf{{Species}} & \\textbf{{Common name}} & \\textbf{{Family}} & \\textbf{{Images}} \\\\
\\midrule
\\endhead
\\midrule
\\multicolumn{{5}}{{r}}{{Continued on next page}} \\\\
\\endfoot
\\bottomrule
\\endlastfoot
{body}
\\end{{longtable}}
\\end{{center}}
"""
    with open("appendix_species.tex", "w", encoding="utf-8") as f:
        f.write(tex)

    print(f"Rows: {len(rows)} | families: {n_fam} | genera: {n_gen}")
    print(f"Image counts: {'read from ' + DATASET_DIR if have_counts else 'NOT set (Images column = --); rerun with DATASET_DIR'}")
    missing = [r[0] for r in rows if not r[1]]
    if missing:
        print("No common name for:", missing)
    if have_counts:
        zero = [r[0] for r in rows if r[3] == 0]
        if zero:
            print("WARNING: zero images found for:", zero)


if __name__ == "__main__":
    main()
