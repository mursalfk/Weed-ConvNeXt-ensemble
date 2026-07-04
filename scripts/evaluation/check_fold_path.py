"""
check_fold_path.py  --  READ-ONLY. Figures out exactly why the fold path is not
found, by walking down it one folder at a time. It only lists directories; it
never creates, moves, writes, or deletes anything.

Run it from the SAME folder as evaluate_models.py.
"""
import os

# Same prefix as your working TEST_DIR, so the prefix itself is known to resolve.
TEST_DIR   = "../../../../PlantCLEF 2015 Dataset/weed_only_dataset/organized_test_dataset"
FOLD1_VAL  = "../../../../PlantCLEF 2015 Dataset/kfold_folds/fold_1/val"


def isdir(path):
    return os.path.isdir(path)


print("1) Confirm the {} loop builds fold_1 .. fold_5 (it does):")
template = "../../../../PlantCLEF 2015 Dataset/kfold_folds/fold_{}/val"
for i in range(1, 6):
    print(f"     i={i} -> {template.format(i)}")

print("\n2) The test directory (known to work):")
print(f"     isdir={isdir(TEST_DIR)}   {TEST_DIR}")

print("\n3) Walking DOWN the fold path, one folder at a time:")
parts = FOLD1_VAL.split("/")
acc = ""
broke = False
for part in parts:
    acc = part if acc == "" else acc + "/" + part
    exists = isdir(acc)
    mark = "" if exists else "   <-- FIRST MISSING LEVEL"
    print(f"     isdir={exists}   {acc}{mark}")
    if not exists:
        broke = True
        parent = os.path.dirname(acc) or "."
        if isdir(parent):
            print(f"\n     What actually exists inside '{parent}':")
            for name in sorted(os.listdir(parent)):
                full = os.path.join(parent, name)
                tag = "DIR " if os.path.isdir(full) else "file"
                print(f"        [{tag}] {name}")
        else:
            print(f"     (parent '{parent}' does not exist either)")
        break

if not broke:
    print("\n     The full path exists. If evaluate_models.py still reported it as")
    print("     missing, the working directory differed between the two runs.")
else:
    print("\n=> Compare the real names listed above with 'kfold_folds', 'fold_1',")
    print("   and 'val'. Whatever the real names are, that is the correct pattern")
    print("   to put in CV_FOLD_VAL_TEMPLATE (keep {} where the fold number goes).")
