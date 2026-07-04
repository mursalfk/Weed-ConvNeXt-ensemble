# Weed Species Identification with a Cross-Validation Ensemble of ConvNeXt Models and Decision-Level Fusion

Source code for the MSc thesis *Weed Species Identification with a
Cross-Validation Ensemble of ConvNeXt Models and Decision-Level Fusion*
(Mursal Furqan Kumbhar, Sapienza University of Rome, 2025/2026).

The pipeline fine-tunes a ConvNeXt-Large backbone on each of five
cross-validation folds of a 94-species weed subset of PlantCLEF 2015, then
combines the five models by decision-level fusion (soft voting and Borda count).
This repository holds the training, evaluation, and figure-generation code so
that the reported numbers can be reproduced.

## Environment

Python 3.11.12, TensorFlow / Keras 2.15.0 (Keras 2.x API), NumPy 1.26.4,
CUDA 12.2 / cuDNN 8, on an NVIDIA RTX 4060 Laptop GPU (8 GB), Ubuntu 22.04 (WSL2).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The `convnext.preprocess_input` routine the pipeline depends on differs between
Keras 2.x and Keras 3.x, so the pinned TensorFlow 2.15 (Keras 2.15) is required
for identical results.

## Data

The experiments use the 94-species weed-relevant subset of PlantCLEF 2015. The
full collection is distributed by the LifeCLEF campaign and is not redistributed
here. The 94 selected species are listed in Appendix A of the thesis. Set the
dataset paths at the top of each notebook and script before running.

## Repository layout

```
notebooks/
  01_eda_and_dataset_prep.ipynb      EDA, imbalance analysis, label-space
                                     reduction, and test-set balancing
  02_kfold_convnext_training.ipynb   5-fold ConvNeXt-Large training + fusion
scripts/
  figures/make_clean_figures.py      7 schematic diagrams
  figures/make_data_figures.py       weed montage + raw distributions
  evaluation/evaluate_models.py       per-fold + ensemble held-out metrics
  evaluation/evaluate_tta.py          test-time augmentation results
  evaluation/find_folds.py            locate the fold directories
  evaluation/check_fold_path.py       sanity-check dataset paths
  build_species_appendix.py           build the Appendix A species table
  extract_species_metadata.py         parse species/genus/family from the XML
data/
  experimental_progression.csv        milestone accuracies (Chapter 6)
requirements.txt   LICENSE   .gitignore   README.md
```

## What maps to what in the thesis

| Thesis item | File |
|---|---|
| EDA, imbalance, duplicates, label-space reduction, test balancing (Chapter 5) | `notebooks/01_eda_and_dataset_prep.ipynb` |
| Two-phase training, 5-fold CV, decision-level fusion (Chapters 4, 6, 7) | `notebooks/02_kfold_convnext_training.ipynb` |
| Per-fold and ensemble metrics (Tables 7.1 and 7.2) | `scripts/evaluation/evaluate_models.py` |
| Test-time augmentation (Section 7.2) | `scripts/evaluation/evaluate_tta.py` |
| Development milestones (Table 6.1 / Figure 6.1) | `data/experimental_progression.csv` |
| Schematic figures 3.1, 4.1, 4.2, 4.5, 5.14, 5.15, 6.2 | `scripts/figures/make_clean_figures.py` |
| Weed montage (1.1) and raw distributions (5.1, 5.3) | `scripts/figures/make_data_figures.py` |
| Species appendix table (Appendix A) | `scripts/build_species_appendix.py`, `scripts/extract_species_metadata.py` |

## Reproducing the figures

```bash
python scripts/figures/make_clean_figures.py   # 7 schematic PNGs under figures/
python scripts/figures/make_data_figures.py    # montage + species/genus/family distributions
```

Both figure scripts pin DejaVu Serif (bundled with matplotlib) so output is
identical across machines.

## Not included (available from the author on request)

The five trained fold models (`convnext_kfold_model_1.h5` ... `_5.h5`) and the
image data are excluded because of their size (see `.gitignore`). 
## License

MIT. See `LICENSE`. Copyright (c) 2026 Mursal Furqan Kumbhar.
