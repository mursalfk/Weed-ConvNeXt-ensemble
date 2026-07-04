"""
extract_species_metadata.py

Builds a table of the weed species used in the thesis, with full taxonomy and
all XML metadata, and writes it to species_list_94.csv (which you then upload).

Your dataset stores an .xml file next to every image, and each XML contains
ClassId, Family, Genus, Species, and Author directly. This script therefore
reads every XML it can find and groups them by ClassId, so it recovers the
distinct species no matter how the folders are arranged (by family or by
species). It does not rely on folder names at all.

SET DATASET_DIR below, then run:   python extract_species_metadata.py
"""

import os
import csv
import xml.etree.ElementTree as ET

# --------------------------------------------------------------------------
# EDIT THIS PATH. Point it at any folder that contains all the species.
# Reading extra files is harmless (duplicates are removed by ClassId), so when
# in doubt point it at the top-level dataset folder.
# --------------------------------------------------------------------------
DATASET_DIR = r"weeds_only_dataset/organized_train_dataset"
OUTPUT_CSV  = "species_list_94.csv"
EXPECTED    = 94          # how many species you expect, used only for a warning
# --------------------------------------------------------------------------

# Taxonomic fields are constant within a species and come first in the output.
TAXO_ORDER = ["ClassId", "Family", "Genus", "Epithet", "ScientificName",
              "Author", "SpeciesRaw"]


def clean_epithet(genus, species_raw):
    """'Achillea millefolium L.' with genus 'Achillea' -> 'millefolium'."""
    parts = species_raw.split()
    if len(parts) >= 2 and parts[0].lower() == genus.lower():
        return parts[1]
    return parts[1] if len(parts) >= 2 else (parts[0] if parts else "")


def main():
    if not os.path.isdir(DATASET_DIR):
        print(f"ERROR: DATASET_DIR not found: {DATASET_DIR}")
        return

    species = {}          # ClassId -> full field dict from a representative XML
    extra_fields = set()  # any non-taxonomic tags seen (Content, Date, ...)
    n_xml = 0

    for root, _, files in os.walk(DATASET_DIR):
        for fn in files:
            if not fn.lower().endswith(".xml"):
                continue
            n_xml += 1
            try:
                node = ET.parse(os.path.join(root, fn)).getroot()
            except ET.ParseError:
                continue
            rec = {child.tag: (child.text or "").strip() for child in node}
            cid = rec.get("ClassId", "").strip()
            if not cid:
                continue
            extra_fields.update(rec.keys())
            if cid not in species:
                species[cid] = rec

    print(f"Parsed {n_xml} XML files; found {len(species)} unique species.")

    # column order: taxonomy first, then every other tag seen, alphabetically
    handled = {"ClassId", "Family", "Genus", "Species", "Author"}
    other_cols = sorted(f for f in extra_fields if f not in handled)
    columns = TAXO_ORDER + other_cols

    rows = []
    for cid, rec in species.items():
        genus = rec.get("Genus", "")
        species_raw = rec.get("Species", "")
        epithet = clean_epithet(genus, species_raw)
        row = {
            "ClassId": cid,
            "Family": rec.get("Family", ""),
            "Genus": genus,
            "Epithet": epithet,
            "ScientificName": f"{genus} {epithet}".strip(),
            "Author": rec.get("Author", ""),
            "SpeciesRaw": species_raw,
        }
        for col in other_cols:
            row[col] = rec.get(col, "")
        rows.append(row)

    # alphabetical by scientific name
    rows.sort(key=lambda r: (r["Genus"].lower(), r["Epithet"].lower()))

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} species to {OUTPUT_CSV}")
    if len(rows) != EXPECTED:
        print(f"NOTE: expected {EXPECTED} species but found {len(rows)}. "
              f"Point DATASET_DIR at a folder that holds all of them, "
              f"or adjust EXPECTED.")


if __name__ == "__main__":
    main()
