"""
find_folds.py  --  READ-ONLY. Locates the cross-validation fold directories on
disk so that CV_FOLD_VAL_TEMPLATE in evaluate_models.py can be set correctly.

This script ONLY lists directory names. It never creates, moves, writes, or
deletes anything, so it is completely safe to run.
"""
import os

# Roots to search. The first is the dataset folder you already use for the test
# set; the second is the current working directory (where the model files live).
# Add more roots if you think the folds might be somewhere else.
SEARCH_ROOTS = [
    "../../../../PlantCLEF 2015 Dataset",
    ".",
]
MAX_DEPTH = 5        # how many levels below each root to search (small = fast)


def search(root):
    root = os.path.normpath(root)
    if not os.path.isdir(root):
        print(f"[skip] root does not exist: {root}")
        return []
    print(f"[scan] {os.path.abspath(root)}   (max depth {MAX_DEPTH})")
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel in (".", "") else rel.replace("\\", "/").count("/") + 1
        if depth >= MAX_DEPTH:
            dirnames[:] = []                     # do not descend any further
        base = os.path.basename(dirpath).lower()
        if "fold" in base or base in ("val", "valid", "validation"):
            sub = sorted(d for d in dirnames if not d.startswith("."))
            hits.append((dirpath, len(sub), sub[:3]))
    return hits


def main():
    all_hits = []
    for r in SEARCH_ROOTS:
        all_hits += search(r)
    print()
    if not all_hits:
        print("No directories containing 'fold', 'val', or 'validation' were found.")
        print("The cross-validation fold folders were most likely not kept on disk")
        print("after training. In that case nothing more is needed: the thesis keeps")
        print("Table 7.1 as it stands, and the held-out result carries the argument.")
        return
    print("Candidate fold / validation directories (with subfolder counts):")
    for path, n_sub, sample in all_hits:
        flag = "   <-- looks like a validation fold (about 94 classes)" \
               if 80 <= n_sub <= 100 else ""
        print(f"  {path}")
        print(f"      {n_sub} subfolders, e.g. {sample}{flag}")
    print()
    print("If you see five validation folders each holding about 94 subfolders,")
    print("set CV_FOLD_VAL_TEMPLATE in evaluate_models.py to their shared pattern,")
    print("using {} where the fold number goes, for example:")
    print('    CV_FOLD_VAL_TEMPLATE = "<the common parent>/fold_{}/val"')
    print("then re-run evaluate_models.py with RUN_TEST_EVAL = False.")


if __name__ == "__main__":
    main()
