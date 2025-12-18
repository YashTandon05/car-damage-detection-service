import os
import json
import random
from pathlib import Path
import pandas as pd

SEED = 42
random.seed(SEED)

REPO_ROOT = Path(__file__).resolve().parents[1]

CARDD_ROOT = REPO_ROOT / "data" / "raw" / "car_dd" / "CarDD_COCO"
CARDD_ANN = CARDD_ROOT / "annotations"

STANFORD_TRAIN_DIR = REPO_ROOT / "data" / "raw" / "stanford_cars" / "cars_train" / "cars_train"
STANFORD_TEST_DIR  = REPO_ROOT / "data" / "raw" / "stanford_cars" / "cars_test" / "cars_test"

SPLITS_DIR = REPO_ROOT / "data" / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

CANON_TYPES = {"dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"}

def load_coco_type_rows(split: str) -> pd.DataFrame:
    """
    Returns df with columns: image_path,label
    Derives ONE label per image from COCO annotations.
    If multiple categories exist for one image, choose the most frequent.
    """
    ann_path = CARDD_ANN / f"instances_{split}.json"
    img_dir = CARDD_ROOT / split

    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # category_id -> name
    cat_map = {c["id"]: c["name"] for c in coco.get("categories", [])}

    # image_id -> file_name
    img_map = {im["id"]: im["file_name"] for im in coco.get("images", [])}

    # image_id -> list of category names
    per_image_labels = {}
    for ann in coco.get("annotations", []):
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        name = cat_map.get(cat_id)
        if not name:
            continue
        name = name.strip()
        per_image_labels.setdefault(img_id, []).append(name)

    rows = []
    missing = 0
    unknown = 0

    for img_id, file_name in img_map.items():
        labels = per_image_labels.get(img_id, [])
        if not labels:
            missing += 1
            continue

        label = max(set(labels), key=labels.count)

        if label not in CANON_TYPES:
            unknown += 1
            continue

        full_path = img_dir / file_name
        if not full_path.exists():
            full_path = img_dir / Path(file_name).name

        if not full_path.exists():
            continue

        rel_path = full_path.relative_to(REPO_ROOT).as_posix()
        rows.append({"image_path": rel_path, "label": label})

    df = pd.DataFrame(rows)
    print(f"[CarDD {split}] rows={len(df)} | skipped(no-anns)={missing} | skipped(unknown_label)={unknown}")
    return df

def list_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])

def split_stanford_train_val(val_ratio=0.15):
    imgs = list_images(STANFORD_TRAIN_DIR)
    random.shuffle(imgs)
    n_val = int(len(imgs) * val_ratio)
    val = imgs[:n_val]
    train = imgs[n_val:]
    return train, val

def sample_to_match(population, k):
    """Sample k items from population (without replacement)."""
    if k >= len(population):
        return population
    return random.sample(population, k)

def save_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)
    print(f"Saved: {path} ({len(df)} rows)")

def main():
    # ---- Type splits (CarDD only)
    type_train = load_coco_type_rows("train")
    type_val   = load_coco_type_rows("val")
    type_test  = load_coco_type_rows("test")

    save_csv(type_train, SPLITS_DIR / "type_train.csv")
    save_csv(type_val,   SPLITS_DIR / "type_val.csv")
    save_csv(type_test,  SPLITS_DIR / "type_test.csv")

    # ---- Stanford splits (no_damage)
    stan_train_imgs, stan_val_imgs = split_stanford_train_val(val_ratio=0.15)
    stan_test_imgs = list_images(STANFORD_TEST_DIR)

    # ---- Binary splits: balance per split vs CarDD split sizes
    # For binary positives, use ALL CarDD images in each split (damage).
    bin_train_pos = type_train.copy()
    bin_val_pos   = type_val.copy()
    bin_test_pos  = type_test.copy()

    bin_train_pos["label"] = "damage"
    bin_val_pos["label"]   = "damage"
    bin_test_pos["label"]  = "damage"

    stan_train_rel = [p.relative_to(REPO_ROOT).as_posix() for p in stan_train_imgs]
    stan_val_rel   = [p.relative_to(REPO_ROOT).as_posix() for p in stan_val_imgs]
    stan_test_rel  = [p.relative_to(REPO_ROOT).as_posix() for p in stan_test_imgs]

    train_neg = sample_to_match(stan_train_rel, len(bin_train_pos))
    val_neg   = sample_to_match(stan_val_rel,   len(bin_val_pos))
    test_neg  = sample_to_match(stan_test_rel,  len(bin_test_pos))

    bin_train_neg = pd.DataFrame({"image_path": train_neg, "label": "no_damage"})
    bin_val_neg   = pd.DataFrame({"image_path": val_neg,   "label": "no_damage"})
    bin_test_neg  = pd.DataFrame({"image_path": test_neg,  "label": "no_damage"})

    binary_train = pd.concat([bin_train_pos[["image_path","label"]], bin_train_neg], ignore_index=True).sample(frac=1, random_state=SEED)
    binary_val   = pd.concat([bin_val_pos[["image_path","label"]],   bin_val_neg],   ignore_index=True).sample(frac=1, random_state=SEED)
    binary_test  = pd.concat([bin_test_pos[["image_path","label"]],  bin_test_neg],  ignore_index=True).sample(frac=1, random_state=SEED)

    save_csv(binary_train, SPLITS_DIR / "binary_train.csv")
    save_csv(binary_val,   SPLITS_DIR / "binary_val.csv")
    save_csv(binary_test,  SPLITS_DIR / "binary_test.csv")

    print("\nDone. Next: run scripts/verify_data.py then start training (binary first).")

if __name__ == "__main__":
    main()