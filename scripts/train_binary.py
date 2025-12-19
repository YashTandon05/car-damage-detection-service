import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)

# -----------------------------
# Config
# -----------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

@dataclass
class TrainConfig:
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    epochs_head: int = 3
    epochs_finetune: int = 8
    lr_head: float = 1e-3
    lr_finetune: float = 2e-4
    weight_decay: float = 1e-4
    early_stop_patience: int = 3
    decision_threshold: float = 0.5
    device: str = "cpu"

CFG = TrainConfig()

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = REPO_ROOT / "data" / "splits"
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = SPLITS_DIR / "binary_train.csv"
VAL_CSV   = SPLITS_DIR / "binary_val.csv"
TEST_CSV  = SPLITS_DIR / "binary_test.csv"

LABEL_TO_IDX = {"no_damage": 0, "damage": 1}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}


# -----------------------------
# Dataset
# -----------------------------
class BinaryCSVDataset(Dataset):
    def __init__(self, csv_path: Path, transform, repo_root: Path):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.repo_root = repo_root

        needed = {"image_path", "label"}
        if not needed.issubset(self.df.columns):
            raise ValueError(f"{csv_path} must have columns {needed}, got {self.df.columns.tolist()}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img_path = self.repo_root / str(r["image_path"])
        label_str = str(r["label"])
        if label_str not in LABEL_TO_IDX:
            raise ValueError(f"Unknown label '{label_str}' in {img_path}")

        with Image.open(img_path) as im:
            im = im.convert("RGB")
            x = self.transform(im)

        y = LABEL_TO_IDX[label_str]
        return x, y


# -----------------------------
# Helpers
# -----------------------------
def set_torch_cpu_perf():
    # You can tweak threads; default is usually fine.
    # If you want to control: torch.set_num_threads(os.cpu_count())
    torch.backends.mkldnn.enabled = True

def build_transforms(image_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5), # car damage should still be damage if mirrored
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02), # helps handle lighting variation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf

@torch.no_grad()
def predict_proba(model, loader, device: str):
    model.eval()
    probs = []
    ys = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.softmax(logits, dim=1)[:, 1]  # P(damage)
        probs.append(p.cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)

def compute_metrics(y_true, p_damage, threshold: float):
    y_pred = (p_damage >= threshold).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(y_true, p_damage)
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist(),
    }

def tune_threshold(y_true, p_damage):
    # Choose threshold maximizing F1 on validation set
    best = {"threshold": 0.5, "f1": -1.0}
    for t in np.linspace(0.05, 0.95, 19):
        m = compute_metrics(y_true, p_damage, threshold=float(t))
        if m["f1"] > best["f1"]:
            best = {"threshold": float(t), "f1": m["f1"]}
    return best["threshold"]

def build_model():
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    # Replace classifier head
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)
    return model

def freeze_backbone(model):
    for p in model.features.parameters():
        p.requires_grad = False

def unfreeze_last_blocks(model, n_blocks: int = 2):
    # Unfreeze last n_blocks of feature layers
    layers = list(model.features.children())
    for layer in layers[-n_blocks:]:
        for p in layer.parameters():
            p.requires_grad = True

def train_one_epoch(model, loader, optimizer, criterion, device: str):
    model.train()
    total_loss = 0.0
    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def save_torchscript(model, out_path: Path, image_size: int):
    model.eval()
    example = torch.randn(1, 3, image_size, image_size)
    scripted = torch.jit.trace(model.cpu(), example)
    scripted.save(str(out_path))


# -----------------------------
# Main
# -----------------------------
def main():
    set_torch_cpu_perf()

    train_tf, eval_tf = build_transforms(CFG.image_size)

    train_ds = BinaryCSVDataset(TRAIN_CSV, train_tf, REPO_ROOT)
    val_ds   = BinaryCSVDataset(VAL_CSV,   eval_tf,  REPO_ROOT)
    test_ds  = BinaryCSVDataset(TEST_CSV,  eval_tf,  REPO_ROOT)

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=False)
    test_loader  = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=False)

    model = build_model().to(CFG.device)

    # ---------------- Head training ----------------
    freeze_backbone(model)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG.lr_head,
        weight_decay=CFG.weight_decay
    )

    best_f1 = -1.0
    best_state = None
    patience = 0

    print("\n== Stage 1: Train head (frozen backbone) ==")
    for epoch in range(1, CFG.epochs_head + 1):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, criterion, CFG.device)
        p_val, y_val = predict_proba(model, val_loader, CFG.device)
        metrics = compute_metrics(y_val, p_val, threshold=CFG.decision_threshold)
        dt = time.time() - t0

        print(f"Epoch {epoch}/{CFG.epochs_head} | loss={loss:.4f} | "
              f"val_f1={metrics['f1']:.3f} val_prec={metrics['precision']:.3f} val_rec={metrics['recall']:.3f} "
              f"val_auc={metrics['roc_auc']:.3f} | {dt:.1f}s")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= CFG.early_stop_patience:
                print("Early stop (head stage).")
                break

    # Restore best head-stage weights
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    # ---------------- Fine-tuning ----------------
    print("\n== Stage 2: Fine-tune last blocks ==")
    unfreeze_last_blocks(model, n_blocks=3)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG.lr_finetune,
        weight_decay=CFG.weight_decay
    )

    best_f1 = -1.0
    best_state = None
    patience = 0

    for epoch in range(1, CFG.epochs_finetune + 1):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, criterion, CFG.device)
        p_val, y_val = predict_proba(model, val_loader, CFG.device)
        metrics = compute_metrics(y_val, p_val, threshold=CFG.decision_threshold)
        dt = time.time() - t0

        print(f"Epoch {epoch}/{CFG.epochs_finetune} | loss={loss:.4f} | "
              f"val_f1={metrics['f1']:.3f} val_prec={metrics['precision']:.3f} val_rec={metrics['recall']:.3f} "
              f"val_auc={metrics['roc_auc']:.3f} | {dt:.1f}s")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= CFG.early_stop_patience:
                print("Early stop (finetune stage).")
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    # ---------------- Threshold tuning on val ----------------
    p_val, y_val = predict_proba(model, val_loader, CFG.device)
    tuned_thresh = tune_threshold(y_val, p_val)
    print(f"\nTuned threshold (max F1 on val): {tuned_thresh:.2f}")

    # ---------------- Final evaluation ----------------
    p_test, y_test = predict_proba(model, test_loader, CFG.device)
    test_metrics = compute_metrics(y_test, p_test, threshold=tuned_thresh)
    print("\n== Test metrics ==")
    for k, v in test_metrics.items():
        print(f"{k}: {v}")

    # ---------------- Save model as TorchScript ----------------
    out_model = MODELS_DIR / "damage_detector.pt"
    save_torchscript(model, out_model, CFG.image_size)
    print(f"\nSaved TorchScript model to: {out_model}")

    meta = {
        "model_name": "mobilenet_v3_small",
        "task": "binary_damage_detection",
        "image_size": CFG.image_size,
        "label_to_idx": LABEL_TO_IDX,
        "idx_to_label": IDX_TO_LABEL,
        "threshold": tuned_thresh,
        "val_size": len(val_ds),
        "test_metrics": test_metrics,
    }
    meta_path = MODELS_DIR / "damage_detector_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved metadata to: {meta_path}")

if __name__ == "__main__":
    main()