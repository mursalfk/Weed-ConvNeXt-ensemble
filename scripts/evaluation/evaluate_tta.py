# =============================================================================
# Test-Time Augmentation (TTA) -- self-contained. Paste into a Jupyter cell and
# run, or run as a script. No retraining.
#
# Answers Prof. Marini's CF-6: does a single ConvNeXt-Large model with TTA rival
# the five-model ensemble? Evaluated on the ORIGINAL 2,803-image test set.
#
# For each test image it averages the model's softmax over four label-preserving
# views (identity, horizontal flip, a mild central zoom, and zoom+flip), then
# reports each fold model with TTA (mean and best) and the five-model soft-voting
# ensemble with TTA, next to the no-TTA numbers from the earlier run.
# =============================================================================
import os, gc, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.convnext import preprocess_input
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ---- ConvNeXt stores a custom 'LayerScale' layer needed to reload the model ----
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
                    name="gamma", shape=(self.projection_dim,),
                    initializer=tf.keras.initializers.Constant(self.init_values),
                    trainable=True)

            def call(self, x):
                return x * self.gamma

            def get_config(self):
                c = super().get_config()
                c.update({"init_values": self.init_values,
                          "projection_dim": self.projection_dim})
                return c

CUSTOM_OBJECTS = {"LayerScale": LayerScale}
try:
    from tensorflow.keras.applications.convnext import StochasticDepth
    CUSTOM_OBJECTS["StochasticDepth"] = StochasticDepth
except Exception:
    pass

# ---------------------------- CONFIG ----------------------------
# These are the same paths that worked in your earlier evaluation.
MODEL_PATHS = [f"convnext_kfold_model_{i}.h5" for i in range(1, 6)]
TEST_DIR = "../../../../PlantCLEF 2015 Dataset/weed_only_dataset/organized_test_dataset"
AUG_PREFIX = "aug_"
CLASS_INDICES_JSON = "class_indices.json"
IMG_SIZE = (224, 224)
NUM_CLASSES = 94
PREDICT_BATCH = 16
ZOOM_KEEP = 0.90          # central fraction kept by the mild-zoom views

# No-TTA reference numbers on the original test set (from your earlier run).
NO_TTA_SINGLE_MEAN = 0.8242
NO_TTA_SINGLE_BEST = 0.8309
NO_TTA_ENSEMBLE_SOFT = 0.8494
# ----------------------------------------------------------------


def setup_gpu():
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass


def load_class_indices(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return {k: int(v) for k, v in d.items()}


def list_test_files(test_dir, name_to_idx):
    paths, labels, is_orig = [], [], []
    for cls in sorted(os.listdir(test_dir)):
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.isdir(cls_dir) or cls not in name_to_idx:
            continue
        idx = name_to_idx[cls]
        for fn in sorted(os.listdir(cls_dir)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(cls_dir, fn))
                labels.append(idx)
                is_orig.append(not fn.startswith(AUG_PREFIX))
    return paths, np.array(labels), np.array(is_orig, dtype=bool)


def metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)),
        average="macro", zero_division=0)
    return acc, p, r, f


# ------------------------------ TTA views ------------------------------
# Each operates on a float batch (N, H, W, 3) in [0, 255] and returns a fresh
# array (never modifies its input in place).
def _hflip(X):
    return np.ascontiguousarray(X[:, :, ::-1, :])


def _center_zoom(X, keep=ZOOM_KEEP):
    h, w = X.shape[1], X.shape[2]
    ch, cw = int(round(h * keep)), int(round(w * keep))
    top, left = (h - ch) // 2, (w - cw) // 2
    cropped = X[:, top:top + ch, left:left + cw, :]
    with tf.device("/CPU:0"):                      # keep the GPU free for the model
        resized = tf.image.resize(cropped, IMG_SIZE, method="bilinear").numpy()
    return np.ascontiguousarray(resized.astype(np.float32))


def view_makers():
    return [
        ("identity",   lambda X: X.copy()),
        ("hflip",      lambda X: _hflip(X)),
        ("zoom",       lambda X: _center_zoom(X)),
        ("zoom+hflip", lambda X: _hflip(_center_zoom(X))),
    ]


def load_raw(paths):
    X = np.zeros((len(paths), IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
    for i, p in enumerate(paths):
        X[i] = img_to_array(load_img(p, target_size=IMG_SIZE))   # [0, 255]
        if (i + 1) % 1000 == 0:
            print(f"    loaded {i + 1}/{len(paths)} images")
    return X


def predict_tta(model, X_raw):
    """Average the model's softmax over the TTA views. Returns (N, NUM_CLASSES)."""
    acc, n = None, 0
    for name, make in view_makers():
        v = make(X_raw)                            # fresh [0,255] array
        p = model.predict(preprocess_input(v), batch_size=PREDICT_BATCH,
                          verbose=0).astype(np.float32)
        acc = p if acc is None else acc + p
        n += 1
        del v
        gc.collect()
    return acc / n


def main():
    setup_gpu()
    name_to_idx = load_class_indices(CLASS_INDICES_JSON)

    paths, y, is_orig = list_test_files(TEST_DIR, name_to_idx)
    paths = [p for p, o in zip(paths, is_orig) if o]   # ORIGINAL images only
    y = y[is_orig]
    if len(paths) == 0:
        raise SystemExit(f"No images found under TEST_DIR={TEST_DIR!r}. "
                         f"Fix the path and re-run.")
    print(f"Original test images: {len(paths)} (should be about 2,803)")
    print(f"TTA views per image: {[n for n, _ in view_makers()]}")

    print("\nLoading raw images once...")
    X_raw = load_raw(paths)

    print("\nRunning TTA with each fold model (one at a time)...")
    per_fold_acc, prob_stack = [], []
    for k, mp in enumerate(MODEL_PATHS, 1):
        tf.keras.backend.clear_session()
        gc.collect()
        print(f"  [{k}/{len(MODEL_PATHS)}] loading {mp}")
        model = load_model(mp, compile=False, custom_objects=CUSTOM_OBJECTS)
        probs = predict_tta(model, X_raw)
        prob_stack.append(probs)
        acc, p, r, f = metrics(y, probs.argmax(axis=1))
        print(f"      single fold {k} + TTA:  acc={acc:.4f}  P={p:.4f}  R={r:.4f}  F1={f:.4f}")
        per_fold_acc.append(acc)
        del model
        gc.collect()

    prob_stack = np.stack(prob_stack, axis=0)
    per_fold_acc = np.array(per_fold_acc)

    ens_pred = prob_stack.mean(axis=0).argmax(axis=1)
    e_acc, e_p, e_r, e_f = metrics(y, ens_pred)

    print("\n================= TTA SUMMARY (original, un-augmented test set) =================")
    print(f"  single fold + TTA:        mean={per_fold_acc.mean():.4f}   best={per_fold_acc.max():.4f}")
    print(f"  ensemble + TTA (soft):    acc={e_acc:.4f}  P={e_p:.4f}  R={e_r:.4f}  F1={e_f:.4f}")
    print("  -------------------------------------------------------------------------------")
    print("  reference, no TTA (from your earlier run):")
    print(f"    single fold, no TTA:    mean={NO_TTA_SINGLE_MEAN:.4f}   best={NO_TTA_SINGLE_BEST:.4f}")
    print(f"    ensemble,   no TTA:     acc={NO_TTA_ENSEMBLE_SOFT:.4f}")
    print("  -------------------------------------------------------------------------------")
    print(f"  TTA effect on a single model (mean): {per_fold_acc.mean() - NO_TTA_SINGLE_MEAN:+.4f}")
    print(f"  best single + TTA  vs  ensemble no TTA: "
          f"{per_fold_acc.max() - NO_TTA_ENSEMBLE_SOFT:+.4f}")
    print("=================================================================================")


main()
