from pathlib import Path
import json
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS = REPO_ROOT / "data" / "splits"
MODELS = REPO_ROOT / "models"
OUTDIR = REPO_ROOT / "reports" / "binary_errors"
OUTDIR.mkdir(parents=True, exist_ok=True)

TEST_CSV = SPLITS / "binary_test.csv"
META = json.loads((MODELS / "damage_detector_meta.json").read_text())
THRESH = META["threshold"]

eval_tf = transforms.Compose([
    transforms.Resize((META["image_size"], META["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

LABEL_TO_IDX = META["label_to_idx"]

@torch.no_grad()
def predict_p_damage(model, img_path: Path):
    with Image.open(img_path) as im:
        x = eval_tf(im.convert("RGB")).unsqueeze(0)
    logits = model(x)
    p = torch.softmax(logits, dim=1)[0, 1].item()
    return p

def main():
    model = torch.jit.load(str(MODELS / "damage_detector.pt")).eval()

    df = pd.read_csv(TEST_CSV)
    rows = []
    for _, r in df.iterrows():
        img_path = REPO_ROOT / r["image_path"]
        y = LABEL_TO_IDX[r["label"]]
        p = predict_p_damage(model, img_path)
        pred = 1 if p >= THRESH else 0
        rows.append((r["image_path"], y, p, pred))

    res = pd.DataFrame(rows, columns=["image_path","y_true","p_damage","y_pred"])

    # False negatives: true damage(1) predicted no_damage(0)
    fn = res[(res.y_true == 1) & (res.y_pred == 0)].sort_values("p_damage")
    print("False negatives:", len(fn))
    (OUTDIR / "fn").mkdir(exist_ok=True, parents=True)

    for i, row in enumerate(fn.itertuples(index=False)):
        src = REPO_ROOT / row.image_path
        dst = OUTDIR / "fn" / f"fn_{i}_p{row.p_damage:.3f}_" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.open(src).convert("RGB").save(dst)

    # Also save the lowest-confidence true positives (hard positives)
    hard_pos = res[(res.y_true == 1)].sort_values("p_damage").head(20)
    (OUTDIR / "hard_pos").mkdir(exist_ok=True, parents=True)
    for i, row in enumerate(hard_pos.itertuples(index=False)):
        src = REPO_ROOT / row.image_path
        dst = OUTDIR / "hard_pos" / f"pos_{i}_p{row.p_damage:.3f}_" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.open(src).convert("RGB").save(dst)

    # Save CSV for documentation
    res.to_csv(OUTDIR / "test_predictions.csv", index=False)
    print("Saved report to:", OUTDIR)

if __name__ == "__main__":
    main()