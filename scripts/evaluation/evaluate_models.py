"""
evaluate_models.py

Produces every number needed for the revision (Prof. Marini, Step 1), using the
five ALREADY-TRAINED ConvNeXt-Large fold models. No retraining.

It reports:
  1. Per-fold and ensemble metrics on the ORIGINAL (un-augmented) test set, and,
     for comparison, on the full augmented test set you reported before.
  2. The held-out accuracy of a single fold model vs the five-model ensemble.
  3. McNemar's test and a paired bootstrap between soft voting and Borda count.
  4. (Optional) a re-score of the saved models on their CV validation folds, to
     confirm the reproducible CV figure.

Requirements: tensorflow, numpy, scikit-learn, scipy.

IMPORTANT: this script applies the SAME ConvNeXt preprocessing used in training
(convnext.preprocess_input on pixels in [0, 255]). Do not change that, or the
numbers will be wrong in the way Chapter 6 documents.
"""

import os
import gc
import json
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.convnext import preprocess_input

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ---------------------------------------------------------------------------
# ConvNeXt models contain a custom 'LayerScale' layer that must be supplied when
# the saved model is reloaded, otherwise load_model raises
# "Unknown layer: 'LayerScale'". Use the real class if it can be imported, and
# fall back to a compatible reimplementation if not.
# ---------------------------------------------------------------------------
try:
    from tensorflow.keras.applications.convnext import LayerScale
except Exception:
    try:
        from keras.applications.convnext import LayerScale
    except Exception:
        from tensorflow.keras import layers as _layers

        class LayerScale(_layers.Layer):
            def __init__(self, init_values, projection_dim, **kwargs):
                super().__init__(**kwargs)
                self.init_values = init_values
                self.projection_dim = projection_dim

            def build(self, _):
                self.gamma = self.add_weight(
                    name="gamma",
                    shape=(self.projection_dim,),
                    initializer=tf.keras.initializers.Constant(self.init_values),
                    trainable=True,
                )

            def call(self, x):
                return x * self.gamma

            def get_config(self):
                cfg = super().get_config()
                cfg.update({"init_values": self.init_values,
                            "projection_dim": self.projection_dim})
                return cfg

CUSTOM_OBJECTS = {"LayerScale": LayerScale}
# Some ConvNeXt versions also register a StochasticDepth layer; include it if present.
try:
    from tensorflow.keras.applications.convnext import StochasticDepth
    CUSTOM_OBJECTS["StochasticDepth"] = StochasticDepth
except Exception:
    pass

# ---------------------------------------------------------------------------
# CONFIG  -- edit these paths
# ---------------------------------------------------------------------------
# The five saved fold models, in fold order (fold 1 ... fold 5).
MODEL_PATHS = [
    "models/convnext_fold1.h5",
    "models/convnext_fold2.h5",
    "models/convnext_fold3.h5",
    "models/convnext_fold4.h5",
    "models/convnext_fold5.h5",
]

# Test directory that contains BOTH the original and the augmented test images,
# one subfolder per class. Augmented images are detected by their filename
# prefix (AUG_PREFIX). The script reports metrics on all images and, separately,
# on the originals only. Check the printed counts against your known numbers
# (about 2,803 original and 4,971 total).
TEST_DIR = "weeds_only_dataset/organized_test_dataset"
AUG_PREFIX = "aug_"

CLASS_INDICES_JSON = "class_indices.json"   # the fixed class ordering from training
IMG_SIZE = (224, 224)
NUM_CLASSES = 94
PREDICT_BATCH = 16          # safe for an 8 GB GPU with ConvNeXt-Large

# Set to False to skip the test-set evaluation (which you have already run) and
# run only the CV re-score below. Leave True to run both.
RUN_TEST_EVAL = True

# CV re-score. Point this at your per-fold VALIDATION folders (one subfolder per
# class), using {} where the fold number goes. Model i is scored on fold i's
# validation set, which reproduces the cross-validation numbers directly from the
# saved models. Set to None to skip. The script prints the resolved path and image
# count per fold, so if the pattern is wrong you will see "not found" and can fix
# it; a correct fold should report roughly 1,155 to 1,218 images.
#   examples:  "kfold_folds/fold_{}/val"   "cv/fold{}/val"   "folds/{}/val"
CV_FOLD_VAL_TEMPLATE = None
# ---------------------------------------------------------------------------


def setup_gpu():
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass


def load_class_indices(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)            # {class_name: index}
    # index -> name, and name -> index
    name_to_idx = {k: int(v) for k, v in d.items()}
    return name_to_idx


def list_test_files(test_dir, name_to_idx):
    """Return (paths, y_true, is_original) over every image, using the fixed
    class ordering from class_indices.json."""
    paths, labels, is_orig = [], [], []
    missing = []
    for cls in sorted(os.listdir(test_dir)):
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        if cls not in name_to_idx:
            missing.append(cls)
            continue
        idx = name_to_idx[cls]
        for fn in sorted(os.listdir(cls_dir)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            paths.append(os.path.join(cls_dir, fn))
            labels.append(idx)
            is_orig.append(not fn.startswith(AUG_PREFIX))
    if missing:
        print(f"  WARNING: {len(missing)} test folders not in class_indices "
              f"(ignored): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    return paths, np.array(labels), np.array(is_orig, dtype=bool)


def preprocess_all(paths):
    """Load every image once, resize, and apply ConvNeXt preprocessing.
    Holds the result in RAM (~3 GB for ~5,000 images). If RAM-limited, evaluate
    the original-only subset, or adapt to predict in chunks."""
    X = np.zeros((len(paths), IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
    for i, p in enumerate(paths):
        X[i] = img_to_array(load_img(p, target_size=IMG_SIZE))   # [0, 255]
        if (i + 1) % 1000 == 0:
            print(f"    loaded {i + 1}/{len(paths)} images")
    return preprocess_input(X)


def predict_each_model(model_paths, X):
    """Load one model at a time (fits in 8 GB), predict, free it.
    Returns an array of shape (M, N, NUM_CLASSES)."""
    probs = []
    for k, mp in enumerate(model_paths, 1):
        tf.keras.backend.clear_session()
        gc.collect()
        print(f"  [{k}/{len(model_paths)}] loading {mp}")
        model = load_model(mp, compile=False, custom_objects=CUSTOM_OBJECTS)
        p = model.predict(X, batch_size=PREDICT_BATCH, verbose=0)
        probs.append(p.astype(np.float32))
        del model
        gc.collect()
    return np.stack(probs, axis=0)


# ---------------- fusion rules ----------------
def soft_vote(prob_stack):
    return prob_stack.mean(axis=0).argmax(axis=1)


def borda_count(prob_stack):
    M, N, C = prob_stack.shape
    rank_sum = np.zeros((N, C), dtype=np.int64)
    for m in range(M):
        order = np.argsort(-prob_stack[m], axis=1)   # classes by descending prob
        ranks = np.argsort(order, axis=1)            # rank of each class (0 = top)
        rank_sum += ranks
    return rank_sum.argmin(axis=1)


# ---------------- metrics & tests ----------------
def metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)),
        average="macro", zero_division=0)
    return acc, p, r, f


def row(label, y_true, y_pred):
    acc, p, r, f = metrics(y_true, y_pred)
    print(f"  {label:<34} acc={acc:.4f}  P={p:.4f}  R={r:.4f}  F1={f:.4f}")
    return acc


def mcnemar_test(y_true, pred_a, pred_b):
    a = (pred_a == y_true)
    b = (pred_b == y_true)
    n01 = int(np.sum(a & ~b))    # A right, B wrong
    n10 = int(np.sum(~a & b))    # A wrong, B right
    n = n01 + n10
    try:
        from scipy.stats import binomtest
        pval = binomtest(min(n01, n10), n, 0.5).pvalue if n > 0 else 1.0
    except Exception:
        pval = float("nan")
    return n01, n10, pval


def bootstrap_ci(y_true, pred_a, pred_b, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    a = (pred_a == y_true).astype(np.float64)
    b = (pred_b == y_true).astype(np.float64)
    N = len(y_true)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, N, N)
        diffs[i] = a[idx].mean() - b[idx].mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def evaluate_split(name, prob_stack, y_true):
    print(f"\n--- {name} (n = {len(y_true)}) ---")
    single_accs = []
    for m in range(prob_stack.shape[0]):
        single_accs.append(row(f"single fold {m + 1}", y_true,
                               prob_stack[m].argmax(axis=1)))
    soft_pred = soft_vote(prob_stack)
    borda_pred = borda_count(prob_stack)
    soft_acc = row("ENSEMBLE soft voting", y_true, soft_pred)
    borda_acc = row("ENSEMBLE Borda count", y_true, borda_pred)

    single_accs = np.array(single_accs)
    best_single = single_accs.max()
    print(f"  single-fold mean acc = {single_accs.mean():.4f} | "
          f"best = {best_single:.4f}")
    print(f"  ensemble (soft) - mean single = {soft_acc - single_accs.mean():+.4f} | "
          f"ensemble - best single = {soft_acc - best_single:+.4f}")

    best_idx = int(single_accs.argmax())
    n01, n10, p = mcnemar_test(y_true, soft_pred,
                               prob_stack[best_idx].argmax(axis=1))
    lo, hi = bootstrap_ci(y_true, soft_pred, prob_stack[best_idx].argmax(axis=1))
    print(f"  ensemble vs best single fold: McNemar p={p:.4f} "
          f"(disc {n01}/{n10}); bootstrap 95% CI of acc diff [{lo:+.4f}, {hi:+.4f}]")

    n01, n10, p = mcnemar_test(y_true, soft_pred, borda_pred)
    lo, hi = bootstrap_ci(y_true, soft_pred, borda_pred)
    print(f"  soft vs Borda: McNemar p={p:.4f} (disc {n01}/{n10}); "
          f"bootstrap 95% CI of acc diff [{lo:+.4f}, {hi:+.4f}]")
    return {"soft": soft_acc, "borda": borda_acc,
            "single_mean": float(single_accs.mean()), "best_single": best_single}


def evaluate_cv(model_paths, template, name_to_idx):
    """Re-score each saved model on its own validation fold. Model i is scored on
    fold i's validation set (the fold it did NOT train on), which reproduces the
    cross-validation numbers directly from the saved model files. Reports
    accuracy, precision, recall, and F1 per fold, plus their mean and std, in a
    layout that maps straight onto Table 7.1."""
    print("\n=== CV re-score: each saved model on its own validation fold ===")
    print("  (reproduces the cross-validation numbers from the saved models)")
    rows = []                      # one (acc, P, R, F1) tuple per fold found
    for i, mp in enumerate(model_paths, 1):
        val_dir = template.format(i)
        if not os.path.isdir(val_dir):
            print(f"  fold {i}: '{val_dir}' not found, skipping this fold.")
            continue
        paths, y, _ = list_test_files(val_dir, name_to_idx)
        if len(paths) == 0:
            print(f"  fold {i}: no images found under '{val_dir}', skipping.")
            continue
        X = preprocess_all(paths)
        ps = predict_each_model([mp], X)          # one model at a time
        acc, p, r, f = metrics(y, ps[0].argmax(axis=1))
        print(f"  fold {i}: val_dir='{val_dir}'  n={len(y)}  "
              f"acc={acc:.4f}  P={p:.4f}  R={r:.4f}  F1={f:.4f}")
        rows.append((acc, p, r, f))
        del X
        gc.collect()

    if not rows:
        print("  No fold validation directories were found, so the CV re-score "
              "was skipped.")
        print("  Set CV_FOLD_VAL_TEMPLATE to the path pattern of your per-fold "
              "validation folders, using {} where the fold number goes.")
        return

    arr = np.array(rows)                          # (n_folds, 4)
    mean, std = arr.mean(axis=0), arr.std(axis=0)
    print("\n  --- reproducible CV summary (drop-in for Table 7.1) ---")
    print(f"  {'Fold':<6}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>9}")
    for i, (acc, p, r, f) in enumerate(rows, 1):
        print(f"  {i:<6}{acc:>10.4f}{p:>11.4f}{r:>9.4f}{f:>9.4f}")
    print(f"  {'Mean':<6}{mean[0]:>10.4f}{mean[1]:>11.4f}{mean[2]:>9.4f}{mean[3]:>9.4f}")
    print(f"  {'Std':<6}{std[0]:>10.4f}{std[1]:>11.4f}{std[2]:>9.4f}{std[3]:>9.4f}")


def main():
    setup_gpu()
    name_to_idx = load_class_indices(CLASS_INDICES_JSON)
    print(f"Loaded class ordering: {len(name_to_idx)} classes.")

    if RUN_TEST_EVAL:
        paths, y_true, is_orig = list_test_files(TEST_DIR, name_to_idx)
        n_orig, n_aug = int(is_orig.sum()), int((~is_orig).sum())
        print(f"Test images found: {len(paths)} total "
              f"({n_orig} original, {n_aug} augmented).")
        print("  -> sanity check: original should be about 2,803 and total about 4,971.")

        print("\nPreprocessing test images once (ConvNeXt preprocessing)...")
        X = preprocess_all(paths)

        print("\nPredicting with each fold model (one at a time)...")
        prob_stack = predict_each_model(MODEL_PATHS, X)
        del X
        gc.collect()

        res_orig = evaluate_split("ORIGINAL test set (honest)",
                                  prob_stack[:, is_orig, :], y_true[is_orig])
        res_full = evaluate_split("FULL augmented test set (as previously reported)",
                                  prob_stack, y_true)

        print("\n========================= SUMMARY FOR THE THESIS =========================")
        print(f"  Honest headline (ensemble soft voting, ORIGINAL test): {res_orig['soft']:.4f}")
        print(f"  Previously reported (soft voting, AUGMENTED test):     {res_full['soft']:.4f}")
        print(f"  Single-fold mean vs ensemble (ORIGINAL): "
              f"{res_orig['single_mean']:.4f} -> {res_orig['soft']:.4f} "
              f"({res_orig['soft'] - res_orig['single_mean']:+.4f})")
        print(f"  Soft vs Borda (ORIGINAL): {res_orig['soft']:.4f} vs {res_orig['borda']:.4f}")
        print("  (see the per-split blocks above for McNemar p-values and bootstrap CIs)")
        print("==========================================================================")

    if CV_FOLD_VAL_TEMPLATE:
        evaluate_cv(MODEL_PATHS, CV_FOLD_VAL_TEMPLATE, name_to_idx)
    elif not RUN_TEST_EVAL:
        print("Nothing to do: RUN_TEST_EVAL is False and CV_FOLD_VAL_TEMPLATE is None.")


if __name__ == "__main__":
    main()
