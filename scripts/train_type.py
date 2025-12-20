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

from sklearn.metrics import classification_report, confusion_matrix, f1_score


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

@dataclass
class TrainConfig:
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    epochs_head: int = 4
    epochs_finetune: int = 10
    lr_head: float = 1e-3
    lr_finetune: float = 2e-4
    weight_decay: float = 1e-4
    early_stop_patience: int = 3
    device: str = "cpu"

CFG = TrainConfig()

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = REPO_ROOT / "data" / "splits"
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = SPLITS_DIR / "type_train.csv"
VAL_CSV   = SPLITS_DIR / "type_val.csv"
TEST_CSV  = SPLITS_DIR / "type_test.csv"

TYPE_CLASSES = ["scratch", "dent", "crack", "glass shatter", "lamp broken", "tire flat"]
LABEL_TO_IDX = {c: i for i, c in enumerate(TYPE_CLASSES)}
IDX_TO_LABEL = {i: c for c, i in LABEL_TO_IDX.items()}


class TypeCSVDataset(Dataset):
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


def build_transforms(image_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.06, hue=0.02),
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


def build_model(num_classes: int):
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def freeze_backbone(model):
    for p in model.features.parameters():
        p.requires_grad = False


def unfreeze_last_blocks(model, n_blocks: int = 3):
    layers = list(model.features.children())
    for layer in layers[-n_blocks:]:
        for p in layer.parameters():
            p.requires_grad = True


def compute_class_weights(train_csv: Path):
    df = pd.read_csv(train_csv)
    counts = df["label"].value_counts().to_dict()
    total = len(df)
    weights = []
    for c in TYPE_CLASSES:
        cnt = counts.get(c, 1)
        weights.append(total / (len(TYPE_CLASSES) * cnt))
    w = torch.tensor(weights, dtype=torch.float32)
    return w, counts


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


@torch.no_grad()
def predict(model, loader, device: str):
    model.eval()
    ys, preds = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.argmax(logits, dim=1).cpu().numpy()
        preds.append(p)
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(preds)


def eval_metrics(y_true, y_pred):
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return float(macro_f1)


def save_torchscript(model, out_path: Path, image_size: int):
    model.eval()
    example = torch.randn(1, 3, image_size, image_size)
    scripted = torch.jit.trace(model.cpu(), example)
    scripted.save(str(out_path))


def main():
    train_tf, eval_tf = build_transforms(CFG.image_size)

    train_ds = TypeCSVDataset(TRAIN_CSV, train_tf, REPO_ROOT)
    val_ds   = TypeCSVDataset(VAL_CSV,   eval_tf,  REPO_ROOT)
    test_ds  = TypeCSVDataset(TEST_CSV,  eval_tf,  REPO_ROOT)

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=False)
    test_loader  = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=False)

    class_w, counts = compute_class_weights(TRAIN_CSV)
    print("Train label counts:", counts)
    print("Class weights:", class_w.tolist())

    model = build_model(num_classes=len(TYPE_CLASSES)).to(CFG.device)
    criterion = nn.CrossEntropyLoss(weight=class_w.to(CFG.device))

    # ---- Stage 1: head training
    freeze_backbone(model)
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
        yv, pv = predict(model, val_loader, CFG.device)
        mf1 = eval_metrics(yv, pv)
        dt = time.time() - t0
        print(f"Epoch {epoch}/{CFG.epochs_head} | loss={loss:.4f} | val_macro_f1={mf1:.3f} | {dt:.1f}s")

        if mf1 > best_f1:
            best_f1 = mf1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= CFG.early_stop_patience:
                print("Early stop (head stage).")
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    # ---- Stage 2: fine-tune
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
        yv, pv = predict(model, val_loader, CFG.device)
        mf1 = eval_metrics(yv, pv)
        dt = time.time() - t0
        print(f"Epoch {epoch}/{CFG.epochs_finetune} | loss={loss:.4f} | val_macro_f1={mf1:.3f} | {dt:.1f}s")

        if mf1 > best_f1:
            best_f1 = mf1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= CFG.early_stop_patience:
                print("Early stop (finetune stage).")
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    # ---- Test evaluation
    yt, pt = predict(model, test_loader, CFG.device)
    macro_f1 = eval_metrics(yt, pt)
    print("\n== Test (type) ==")
    print("macro_f1:", macro_f1)
    print("\nClassification report:")
    print(classification_report(yt, pt, target_names=TYPE_CLASSES, zero_division=0))

    cm = confusion_matrix(yt, pt)
    print("\nConfusion matrix:\n", cm)

    # ---- Save TorchScript + metadata
    out_model = MODELS_DIR / "damage_type.pt"
    save_torchscript(model, out_model, CFG.image_size)
    print("\nSaved TorchScript model to:", out_model)

    meta = {
        "model_name": "mobilenet_v3_small",
        "task": "damage_type_classification",
        "image_size": CFG.image_size,
        "classes": TYPE_CLASSES,
        "label_to_idx": LABEL_TO_IDX,
        "idx_to_label": IDX_TO_LABEL,
        "test_macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
    }
    meta_path = MODELS_DIR / "damage_type_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print("Saved metadata to:", meta_path)

if __name__ == "__main__":
    main()