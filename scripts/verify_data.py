from pathlib import Path
import pandas as pd
from PIL import Image
import random

SEED = 42
random.seed(SEED)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = REPO_ROOT / "data" / "splits"

def check_csv(name: str):
    path = SPLITS_DIR / name
    df = pd.read_csv(path)
    print(f"\n== {name} ==")
    print("rows:", len(df))
    print("label counts:\n", df["label"].value_counts())

    missing = []
    for p in df["image_path"].sample(min(200, len(df)), random_state=SEED):
        if not (REPO_ROOT / p).exists():
            missing.append(p)
    if missing:
        print("MISSING (sample):", missing[:10])
    else:
        print("Path check (sample): OK")

    for p in df["image_path"].sample(min(3, len(df)), random_state=SEED+1):
        img_path = REPO_ROOT / p
        try:
            with Image.open(img_path) as im:
                im.verify()
        except Exception as e:
            print("BAD IMAGE:", p, "err:", e)

def main():
    for f in [
        "type_train.csv","type_val.csv","type_test.csv",
        "binary_train.csv","binary_val.csv","binary_test.csv"
    ]:
        check_csv(f)

if __name__ == "__main__":
    main()